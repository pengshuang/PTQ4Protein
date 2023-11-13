## PTQ4Protein

### Introduction
Pytorch Code for our paper at IEEE BIBM 2023: **Exploring Post-Training Quantization of Protein
Language Models** [[arXiv]](https://arxiv.org/abs/2310.19624)

### Requirements

The code was verified on Python-3.8.15, PyTorch-1.13.1, Transformers-4.27.1

```
pip install -r requirements.txt
```

### Usage
Check **PTQ4Protein** at **piecewise_quant/piecewise.py**


#### Task 1: Protein Structure Prediction (Supervised)

- Entering Task Dir

```
cd structure_prediction/scripts
```

- Quantizing ESMFold Model

```
# only quantize model weights
python quant_weis.py

# only quantize model activations
python quant_acts.py

# quantize both model weights and activations
python quant_full.py
```

- Evaluating Quantized ESMFold Model

```
# evaluate quantization of model weights
python eval_quant_weis.py

# evaluate quantization of model activations
python eval_quant_acts.py

# evaluate quantization of both model weights and activations
python eval_quant_full.py

```

Evaluation results would be printed on the command-line and prediction results would be saved at ``../data/output/`` dir.



#### Task 2: Protein Contact Prediction (Unsupervised)

- Entering Task Dir

```
cd contact_prediction/scripts
```

- Quantizing ESM2 Model and Evaluating Quantized Model

```
# only quantize model weights
python quant_weis.py

# only quantize model activations
python quant_acts.py

# quantize both model weights and activations
python quant_full.py
```

Evaluation results would be printed on the command-line.

### Cite Our Work
```
@inproceedings{peng2023protein,
  title={Exploring Post-Training Quantization of Protein Language Models},
  author={Peng, Shuang and Yang, Fei and Sun, Ning and Chen, Sheng and Jiang, Yanfeng and Pan, Aimin},
  booktitle={2023 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2023},
  organization={IEEE}
}
```

### Reference

The work of PWLQ has given us great inspiration. Here is the code and paper of PWLQ.

https://github.com/jun-fang/PWLQ

PyTorch Code for our paper at ECCV 2020 (**oral** presentation): **Post-Training Piecewise Linear Quantization for Deep Neural Networks** [[Paper]](https://github.com/jun-fang/PWLQ/blob/master/paper/2949.pdf) [[arXiv]](https://arxiv.org/abs/2002.00104) 
