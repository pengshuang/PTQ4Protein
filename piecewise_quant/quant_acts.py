import copy
import torch
import torch.nn as nn
from torch.nn import LayerNorm

from transformers import AutoTokenizer, AutoModel, EsmForProteinFolding
from transformers.models.esm.modeling_esm import *
from transformers.models.esm.modeling_esmfold import *

from .piecewise import *
from .uniform import *


class QuantAct(nn.Module):

    def __init__(self, act_bits, get_stats, minv=None, maxv=None, 
        cali_sample_size=50, cali_batch_size=1, topk=1000, quant_scheme="uniform"):
        '''
        cali_sample_size: calibration sample size, typically from random training data
        cali_batch_size: calibration sampling batch size
        topk: calibrate topk lower and upper bounds
        '''
        super(QuantAct, self).__init__()
        self.act_bits = act_bits
        self.get_stats = get_stats
        self.index = 0
        self.topk = topk
        self.sample_batches = cali_sample_size // cali_batch_size
        stats_size = (self.sample_batches, self.topk) if self.get_stats else 1
        self.register_buffer('minv', torch.zeros(stats_size))
        self.register_buffer('maxv', torch.zeros(stats_size))
        self.quant_scheme = quant_scheme

    def forward(self, x):
        if self.get_stats:
            y = x.clone()
            y = torch.reshape(y, (-1,))
            y, indices = torch.sort(y)
            topk_mins = y[:self.topk]
            topk_maxs = y[-self.topk:]
            if self.index < self.sample_batches:
                self.minv[self.index, :] = topk_mins
                self.maxv[self.index, :] = topk_maxs
                self.index += 1

        if self.act_bits > 0:
            ## uniform quantization
            if self.minv is not None:
                if self.minv >= 0.0: # activation after relu
                    self.minv *= 0.0
                    self.signed = False
                else: 
                    self.maxv = max(-self.minv, self.maxv) 
                    self.minv = - self.maxv
                    self.signed = True

            if self.quant_scheme.startswith("pwlq"):
                pw_opt = int(self.quant_scheme[-1])
                x, bkp_ratio, _ = piecewise_linear_quant(x, minv=self.minv, maxv=self.maxv, 
                                                          signed=self.signed, bits=self.act_bits, scale_bits=0.0, 
                                                          pw_opt=pw_opt, approximate=False)
            else:
                x = uniform_symmetric_quantizer(x, bits=self.act_bits, 
                    minv=self.minv, maxv=self.maxv, signed=self.signed)
                # x = uniform_affine_quantizer(x, bits=self.act_bits, 
                #         minv=self.minv, maxv=self.maxv)
            
        return x


def quant_model_acts(model, act_bits, get_stats, exclude_part=["base_model"], cali_batch_size=1, quant_scheme="uniform"):
    
    # evaluate different module of ESMFold model
    
    if type(model) in [nn.Linear, nn.Softmax, nn.functional.softmax, nn.Sigmoid(), nn.ReLU(), EsmFoldLinear, LayerNorm]:
    # if type(model) in [nn.Linear, nn.Softmax, nn.functional.softmax, nn.Sigmoid(), nn.ReLU(), EsmFoldLinear]:
    # if type(model) in [nn.Linear, nn.Sigmoid(), nn.ReLU(), EsmFoldLinear, LayerNorm]:
    # if type(model) in [nn.Softmax, nn.functional.softmax]:
    # if type(model) in [LayerNorm]:
    # if type(model) in [nn.Linear, EsmFoldLinear]:
        quant_act = QuantAct(act_bits, get_stats, cali_batch_size=cali_batch_size, quant_scheme=quant_scheme)
        return nn.Sequential(quant_act, model)
        # return nn.Sequential(model, quant_act)
    elif type(model) == nn.Sequential:
        modules = []
        for name, module in model.named_children():
            modules.append(quant_model_acts(module, act_bits, get_stats, cali_batch_size=cali_batch_size, quant_scheme=quant_scheme))
        return nn.Sequential(*modules)
    else:
        quantized_model = copy.deepcopy(model)
        for attribute in dir(model):
            if attribute in exclude_part:
                continue
            module = getattr(model, attribute)
            if isinstance(module, nn.ModuleList):
                new_module = nn.ModuleList()
                for i, m in enumerate(module.children()):
                    new_module.append(quant_model_acts(m, act_bits, get_stats, cali_batch_size=cali_batch_size, quant_scheme=quant_scheme))
                setattr(quantized_model, attribute, new_module)
            elif isinstance(module, nn.Module):
                setattr(quantized_model, attribute,
                        quant_model_acts(module, act_bits, get_stats, cali_batch_size=cali_batch_size, quant_scheme=quant_scheme))
            
        return quantized_model


def save_model_act_stats(model, save_path):
    checkpoint = model.state_dict()
    act_stats = copy.deepcopy(checkpoint)
    for key in checkpoint:
        if '.minv' not in key and '.maxv' not in key:
            del act_stats[key]
    torch.save(act_stats, save_path)
    return act_stats


def load_model_act_stats(model, load_path, act_clip_method):
    checkpoint = model.state_dict()
    act_stats = torch.load(load_path)
    for key in act_stats:
        min_or_max = 'min' if '.minv' in key else 'max'
        value = act_clip_bounds(act_stats[key], act_clip_method, min_or_max)
        # if not key.startswith("trunk.structure_module"):
        #     key = key.replace('module.', '')
        checkpoint[key][0] = value
    model.load_state_dict(checkpoint)
    return model


def act_clip_bounds(stats, act_clip_method, min_or_max):
    if act_clip_method.startswith('top'):
        topk = int(act_clip_method.split('_')[1]) 
        stats = stats[:, :topk] if min_or_max == 'min' else stats[:, -topk:]
        values, indices = torch.median(stats, 1)
        res = torch.mean(values)
        if res.item() <= 0 and min_or_max == "max":
            stats = stats[:, -5:]
            values, indices = torch.median(stats, 1)
            res = torch.mean(values)
        return res
    elif act_clip_method.startswith('clip'):
        clip_coef = float(act_clip_method.split('_')[1])
        clip_value = torch.min(stats) if min_or_max == 'min' else torch.max(stats)
        return clip_coef * clip_value
    else:
        raise RuntimeError("Please implement for activation clip method: %s !!!" % act_clip_method) 