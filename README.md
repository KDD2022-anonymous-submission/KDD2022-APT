# Data-Active Pre-training of Graph Neural Networks

## About

This project is the implementation of the KDD'22 paper "Data-Active Pre-training of Graph Neural Networks"

## Dependencies
The script has been tested running under Python 3.7.10, with the following packages installed (along with their dependencies):

- `PyTorch ≥ 1.4.0`
- `0.5 > DGL ≥ 0.4.3`
- `pip install -r requirements.txt`
- `conda install -c conda-forge rdkit=2019.09.2`

In addition, CUDA 10.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution

## File folders

`splits`: **need to unzipped**, contains the split data of "Cora, Pubmed, Cornell and Wisconsin".

`dataset`: contains the data of "DD242, DD68, DD687".

`scripts`: contains all the scripts for running code.

`gcc&utils`: contains the code of model.

## Usage: How to run the code
We divide it into 3 steps (1) Pre-training/Finetuning (2) Evaluating (3) Analyze the performance.

### 1.Pre-training / Fine-tuning

Pre-training datasets is stored in `data.bin`. And the datasets can be download through [website](https://drive.google.com/file/d/1kbOciSHXSOAFV7X1CuL_nm9_2sxKeDfU/view).

**1.1 Pretraining**

```bash
python train_al.py \
  --gap <period for graph selection> \
  --graph <budget of input graphs> \
  --exp <exp file> \
  --model-path <saved file> \
  --tb-path <tensorboard file> \
  --dgl_file <dataset in bin format> \
  --moco
```
**Demo:**	

```bash
python train_al.py \
  --gap 6 \
  --graph 7 \
  --exp Pretrain \
  --model-path saved \
  --tb-path tensorboard  \
  --dgl_file data.bin \
  --moco 
```
**1.2 Fine-tuning**

**Finetune APT on all downstream datasets in the background:**

```
nohup bash scripts/evaluate_generate.sh <saved file> > <log file> 2>&1 &
```

**Demo:**

```
nohup bash scripts/evaluate_generate.sh saved > result.out 2>&1 &
```

### 2.Evaluating

**2.1 Evaluate without Fine-tuning on all downstream datasets in the background:**

```
nohup bash evaluate.sh <load path> <gpu> > <log file> 2>&1 &
```

**Demo:**

```
nohup bash scripts/evaluate.sh saved 0 > log.out 2>&1 &
```


**2.2 Evaluate after Fine-tuning on all downstream datasets in the background:**

```
nohup bash evaluate_finetune.sh <load path> <gpu> > <log file> 2>&1 &
```

**Demo:**

```
nohup bash scripts/evaluate_finetune.sh saved 0 > log.out 2>&1 &
```

### 3.Analyze the performance

Analyze the performance from log file generated in `2.Evaluating` phase and save in csv format file.

```
python cope_result.py --file <log file>
```

**Demo:**
```
python cope_result.py --file test.out
```

## Acknowledgements
Part of this code is inspired by Qiu et al.'s [GCC: Graph Contrastive Coding](https://github.com/THUDM/GCC).

