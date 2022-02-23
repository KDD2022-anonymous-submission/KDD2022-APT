## Data-Active Pre-training of Graph Neural Networks

To run the code, we divide it into 3 steps (1) pre-training/finetuning (2) generate embedding (3) node/graph classification

#### 1.Pre-training
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
#### 2.Finetunine
Finetune APT on single downstream datasets:

```bash
bash scripts/finetune.sh <load_path> <gpu> usa_airport
```
Finetune APT on all downstream datasets

```bash
nohup bash evaluate_finetune.sh <saved file> >result.out 2>&1 &
```

#### 2.Generate embeddings
We can 
```
bash scripts/generate.sh <gpu> <load_path> 
```

For example:

```bash
bash scripts/generate.sh 0 saved/GCC_perturb_16384_0.001_self/current.pth brazil_airport 
```

#### 3.Node/graph Classification

Node classification:

```bash
bash scripts/node_classification/ours.sh <load_path> <hidden_size> brazil_airport
```

Graph classification:

```bash
bash scripts/graph_classification/ours.sh <load_path> <hidden_size> imdb-binary
```
