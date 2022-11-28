PATTERN="2fwl"
# 2wl, 2wl_l, 2fwl, 2fwl_l

DATASET="Citeseer"
# Cora, Citeseer, Pubmed
# ogbl-ddi ?

SEED=60

# python 2WLtest.py \
#   --pattern $PATTERN \
#   --dataset $DATASET \
#   --device 2 \
#   --seed $SEED \
#   --subgraph path

# python 2WLtest.py \
#   --pattern $PATTERN \
#   --dataset $DATASET \
#   --device 1 \
#   --seed $SEED \
#   --subgraph cycle

# python 2WLtest.py \
#   --pattern $PATTERN \
#   --dataset $DATASET \
#   --device 2 \
#   --seed $SEED \
#   --subgraph incoming

# python 2WLtest.py \
#   --pattern $PATTERN \
#   --dataset $DATASET \
#   --device 3 \
#   --seed $SEED \
#   --subgraph outgoing

python 2WLtest.py \
  --pattern $PATTERN \
  --dataset $DATASET \
  --device 3 \
  --seed $SEED \
  --subgraph alpha