# Data-Active Pre-training of Graph Neural Networks

Code for KDD'22 Data-Active Pre-training of Graph Neural Networks

##File folders

`splits`: need to unzipped, contains the split data of "Cora, Pubmed, Cornell and Wisconsin".

`dataset`: contains the data of "DD242, DD68, DD687".

`scripts`: contains all the scripts for running code.

`gcc&utils`: contains the code of model.

## How to run the code
We divide it into 3 steps (1) Pre-training/Finetuning (2) Evaluating (3) Analyze the performance

#### 1.Pre-training / Fine-tuning
**Pretraining**

Pre-training datasets is stored in `data.bin`

```bash
python train_al.py \
  --gap 6 <period for graph selection> \
  --graph <budget of input graphs> \
  --rate <rewiring rate> \
  --exp <exp file> \
  --model-path <saved file> \
  --tb-path <tensorboard file> \
  --dgl_file <dataset in bin format> \
  --moco
```
For example:

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
**Fine-tuning**

**Finetune APT on all downstream datasets in the background:**

```
nohup bash scripts/evaluate_generate.sh <saved file> > <log file> 2>&1 &
```

For example:

```
nohup bash scripts/evaluate_generate.sh saved > result.out 2>&1 &
```

#### 2.Evaluating

**Evaluate without Fine-tuning on all downstream datasets in the background:**

```
nohup bash evaluate.sh <load path> <gpu> > <log file> 2>&1 &
```

For example:

```
nohup bash scripts/evaluate.sh saved 0 > log.out 2>&1 &
```


**Evaluate after Fine-tuning on all downstream datasets in the background:**

```
nohup bash evaluate_finetune.sh <load path> <gpu> > <log file> 2>&1 &
```

For example:

```
nohup bash scripts/evaluate_finetune.sh saved 0 > log.out 2>&1 &
```

#### 3.Analyze the performance

Analyze the performance from log file generated in `Evaluating` phase and save in csv file.

```
python cope_result.py --file <log file>
```

