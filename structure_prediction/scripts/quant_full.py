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
model = model.cpu()

# 2. quantizing model weights

quant_layers = []
for key in model.state_dict().keys():
    if key.startswith("esm.encoder.layer") or key.startswith("trunk.block") or key.startswith("trunk.structure_module"):
        quant_layers.append(key)
    
checkpoint, rmse = quant_checkpoint(model, quant_layers)
model.load_state_dict(checkpoint)
del checkpoint

# 3. quantizing model activations

model = quant_model_acts(model, 0, True, exclude_part=["base_model"])
model = model.cuda("cuda:2")

seq_fasta = list(SeqIO.parse("../data/casp14.fasta", "fasta"))
seq_list = [seq.seq.__str__() for seq in seq_fasta][:50]
key_list = [seq.id.__str__().split("_")[0] for seq in seq_fasta][:50]

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
ecoli_tokenized = tokenizer(seq_list, padding=False, add_special_tokens=False)['input_ids']

outputs = []
with torch.no_grad():
    for input_ids in tqdm(ecoli_tokenized):
        input_ids = torch.tensor(input_ids, device='cuda:2').unsqueeze(0)
        output = model(input_ids)
        outputs.append({key: val.cpu() for key, val in output.items()})


os.makedirs('../output/stats/', exist_ok=True)
act_stats_save_path = '../output/stats/act_stats_full_50.pth'
act_dict = save_model_act_stats(model, act_stats_save_path)


new_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=False)
new_model = new_model.cpu()
new_model = quant_model_acts(new_model, 8, False, exclude_part=["base_model"], cali_batch_size=50, quant_scheme="pwlq-3")
load_model_act_stats(new_model, act_stats_save_path, act_clip_method="top_5")


# 4. saving quantized model

os.makedirs('../output/quant_full/', exist_ok=True)
torch.save(new_model, "../output/quant_full/quant_8b_full.pt")
