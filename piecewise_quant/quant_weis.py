import torch
import numpy as np

from .piecewise import *
from .uniform import *


def quant_checkpoint(model, weight_layers, wei_quant_scheme="uni"):
    
    checkpoint = model.state_dict()

    all_quant_error, all_quant_num = 0, 0
    all_tail_num = 0
    
    for each_layer in weight_layers:
        each_layer_weights = checkpoint[each_layer].clone()

        quant_error, quant_num, layer_tail_num = 0, 0, 0
        output_channel_num = each_layer_weights.size()[0]
        
        if len(each_layer_weights.size()) > 1:
            for c in range(output_channel_num):
                w = each_layer_weights[c, :].clone().cpu()
                qw, err, tail_num = quant_weights(w, wei_quant_scheme)

                each_layer_weights[c, :] = qw
                quant_error += err
                quant_num += len(qw.reshape(-1, 1))
                layer_tail_num += tail_num
        else:
            w = each_layer_weights.clone().cpu()
            qw, err, tail_num = quant_weights(w, wei_quant_scheme)
            if np.isnan(err):
                continue
            each_layer_weights = qw
            quant_error += err
            quant_num += len(qw.reshape(-1, 1))
            layer_tail_num += tail_num
        
        all_quant_num += quant_num
        all_quant_error += quant_error
        all_tail_num += layer_tail_num
        
        checkpoint[each_layer] = each_layer_weights
        if np.isnan(quant_error):
            continue
    
    rmse = np.sqrt(all_quant_error / all_quant_num)
    print('\ntotal quant RMSE: %.4e' % rmse)
    
    return checkpoint, rmse


def quant_weights(w, wei_quant_scheme, bias_corr=False):
    '''
    Quantize a tensor of weights 
    '''
    bkp_ratio = 0.0
    if wei_quant_scheme.startswith('uni'):
        qw = uniform_symmetric_quantizer(w, bits=8)
    else:
        try:
            pw_opt = int(wei_quant_scheme[-1])
        except:
            print('PWLQ options: \n  pw-1 with overlapping regions; pw-2 with non-overlapping regions')
            raise RuntimeError("Please specify an option for PWLQ, like 'pw-2' !!!")
        
        # piecewise linear quantization (PWLQ)
        qw, bkp_ratio, _ = piecewise_linear_quant(w, 
                                bits=8, scale_bits=0.0, 
                                break_point_approach="norm", pw_opt=pw_opt, 
                                approximate=False)
    
    # bias correction
    if bias_corr:
        mean_w, std_w = torch.mean(w), torch.std(w)
        mean_diff = mean_w - torch.mean(qw)
        std_ratio = torch.div(std_w, torch.std(qw) + 1e-12)
        qw = torch.mul(qw + mean_diff, std_ratio)
    
    err = float(torch.sum(torch.mul(qw - w, qw - w)))

    abs_max = torch.max(torch.abs(w))
    break_point = abs_max * bkp_ratio
    tail_num = np.sum(torch.abs(w).detach().numpy() > float(break_point))

    return qw, err, tail_num