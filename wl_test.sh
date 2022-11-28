PATTERN="2wl"
# 2wl, 2wl_l, 2fwl, 2fwl_l

DATASET="Cora"
# Cora, Citeseer, Pubmed
# ogbl-ddi ?

SEED=0

python 2WLtest.py \
  --pattern $PATTERN \
  --dataset $DATASET \
  --device $1 \
  --seed $SEED \