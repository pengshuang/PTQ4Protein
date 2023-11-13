import os
import copy
from tqdm import tqdm

import torch
import torch.nn as nn

import numpy as np 
import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import SingleLetterAlphabet

from transformers import AutoTokenizer, AutoModel, EsmForProteinFolding

import sys
sys.path.append('../../')
from piecewise_quant import *


# 1. loading EsmForProteinFolding model

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=False)

model = model.cuda('cuda:1')
model.esm = model.esm.float()


# 2. quantizing model weights

quant_layers = []

for key in model.state_dict().keys():
    key_size = model.state_dict()[key].size()
    if key.startswith("esm.encoder.layer") or key.startswith("trunk.block") or key.startswith("trunk.structure_module"):
        quant_layers.append(key)
        

checkpoint, rmse = quant_checkpoint(model, quant_layers)
model.load_state_dict(checkpoint)
del checkpoint
print("RMSE: {}".format(rmse))

# 3. saving quantized model

os.makedirs('../output/quant_weis/', exist_ok=True)
torch.save(model, "../output/quant_weis/quant_weis_8b.pt")
