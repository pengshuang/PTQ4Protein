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


model = torch.load("../output/quant_acts/quant_act_8b.pt")
model.cuda("cuda:0")

seq_fasta = list(SeqIO.parse("../data/sequences_cameo.fasta", "fasta"))
seq_list = [seq.seq.__str__() for seq in seq_fasta]
key_list = [seq.id.__str__().split("_")[0] for seq in seq_fasta]

tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
ecoli_tokenized = tokenizer(seq_list, padding=False, add_special_tokens=False)['input_ids']

outputs = []
os.makedirs('../output/pred_quant_act_pdb_v1/', exist_ok=True)
with torch.no_grad():
    for input_ids in tqdm(ecoli_tokenized):
        input_ids = torch.tensor(input_ids, device='cuda:0').unsqueeze(0)
        output = model(input_ids)
        outputs.append({key: val.cpu() for key, val in output.items()})
        
pdb_list = [convert_outputs_to_pdb(output) for output in outputs]
for identifier, pdb in zip(key_list, pdb_list):
    with open(f"../output/pred_quant_act_pdb_v1/{identifier}.pdb", "w") as f:
        f.write("".join(pdb))
        

from TMscore import TMscore
from tqdm import tqdm

real_pdbs = os.listdir("../data/cameo_real_pdb")
pred_quant_pdbs = os.listdir("../output/pred_quant_act_pdb_v1")

tmscore = TMscore("TMscore")
tmscore_list = []
lddt_list = []

for a, b in tqdm(zip(real_pdbs, pred_quant_pdbs)):
    tmscore(os.path.join("../data/cameo_real_pdb", a), os.path.join("../output/pred_quant_act_pdb_v1", b))
    score = tmscore.get_tm_score()
    if score is not None:
        tmscore_list.append(tmscore.get_tm_score())

tmscore_pred = sum(tmscore_list) / len(tmscore_list)   
print("Average TMScore = {}/{} = {}".format(sum(tmscore_list), len(tmscore_list), tmscore_pred))
