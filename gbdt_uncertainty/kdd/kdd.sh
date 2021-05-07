#!/bin/bash

# Download KDD dataset
URLs=(
    "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz"
    "http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz"
)

FILEs=(
    "kddcup.data.gz"
    "corrected.gz"
)


for ((i=0;i<${#URLs[@]};++i)); do
        url=${URLs[i]}
        wget "$url"
done

for ((i=0;i<${#FILEs[@]};++i)); do
        file=${FILEs[i]}
        gunzip $file
done

sort kddcup.data | uniq -c > tmp
awk -F ' ' '{print $1}' tmp > train_counts.txt
awk -F ' ' '{print $2}' tmp > kdd_train_compressed.csv

sort corrected | uniq -c > tmp
awk -F ' ' '{print $1}' tmp > corrected_counts.txt
awk -F ' ' '{print $2}' tmp > corrected_compressed.csv

rm tmp

git clone https://github.com/yandex-research/GBDT-uncertainty.git

mkdir model

python GBDT-uncertainty/gbdt_uncertainty/kdd/train_sglb.py ./ model/ --n_models 2 --lr 0.15 --n_iters 100 --n_objects 10000
python gGBDT-uncertainty/gbdt_uncertainty/kdd/eval_sglb.py ./ model --n_models 2 > model/results.txt