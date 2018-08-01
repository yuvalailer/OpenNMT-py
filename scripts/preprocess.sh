#!/usr/bin/env bash

DATA=41

echo "converting *.data to *.txt ..."

for s in "train dev test"
do
    zcat ../../../project/data/test.data | jq '.text' > test.article.txt

done


python preprocess.py -train_src data/news/train.article.txt \
 -train_tgt data/news/train.sum.txt \
 -valid_src data/news/dev.article.txt \
 -valid_tgt data/news/dev.sum.txt \
 -save_data data/news/NEWS \
 -src_seq_length 10000 \
 -dynamic_dict \
 -share_vocab \
 -max_shard_size 524288000