import torch
import numpy as np


def uniform_symmetric_quantizer(x, bits=8.0, minv=None, maxv=None, signed=True, 
                                scale_bits=0.0, num_levels=None, scale=None, simulated=True):
    if minv is None:
        maxv = torch.max(torch.abs(x))
        minv = - maxv if signed else 0

    if signed:
        maxv = np.max([-float(minv), float(maxv)])
        minv = - maxv
    else:
        minv = 0
    
    if num_levels is None:
        num_levels = 2 ** bits

    if scale is None:
        scale = (maxv - minv) / (num_levels - 1)

    if scale_bits > 0:
        scale_levels = 2 ** scale_bits
            
    ## clamp
    x = torch.clamp(x, min=float(minv), max=float(maxv))
        
    x_int = torch.round(x / scale)
    
    if signed:
        x_quant = torch.clamp(x_int, min=-num_levels/2, max=num_levels/2 - 1)
        if not (minv == - maxv):
            print("===================DEBUG=======================")
            print(x)
            print(minv, maxv)
        assert(minv == - maxv)
    else:
        x_quant = torch.clamp(x_int, min=0, max=num_levels - 1)
        if not (minv == 0 and maxv > 0):
            print("===================DEBUG=======================")
            print(x)
            print(minv, maxv)
        assert(minv == 0 and maxv > 0)
        
    x_dequant = x_quant * scale
    
    return x_dequant if simulated else x_quant


def uniform_affine_quantizer(x, bits=8.0, minv=None, maxv=None, offset=None, include_zero=True,
                            scale_bits=0.0, num_levels=None, scale=None, simulated=True):
    if minv is None:
        maxv = torch.max(x)
        minv = torch.min(x)
        if include_zero:
            if minv > 0:
                minv = 0
            elif maxv < 0:
                maxv = 0
    
    if num_levels is None:
        num_levels = 2 ** bits
    
    if not scale:
        scale = (maxv - minv) / (num_levels - 1)

    if not offset:
        offset =  minv

    if scale_bits > 0:
        scale_levels = 2 ** scale_bits
        scale = torch.round(torch.mul(scale, scale_levels)) / scale_levels
        offset = torch.round(torch.mul(offset, scale_levels)) / scale_levels

    x = torch.clamp(x, min=float(minv), max=float(maxv))
    x_int = torch.round((x - offset) / scale)
    x_quant = torch.clamp(x_int, min=0, max=num_levels - 1)
    x_dequant = x_quant * scale + offset
    
    return x_dequant if simulated else x_quant
