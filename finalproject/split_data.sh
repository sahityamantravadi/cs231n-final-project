#!/bin/bash

data_dir='/home/smantra/finalproject/data'
eval_dir='/home/smantra/finalproject/eval'
test_dir='/home/smantra/finalproject/test'
total=($(wc -l <(ls $data_dir)))
num_val=200
num_test=100
num_train=(($total - $num_val - $num_test))

echo 'Splitting '$total' examples into '$num_train' training, '$num_val' validation, and '$num_test' training examples.'

val_files=$(ls $data_dir | shuf -n $num_val)

for v in $val_files;
do
    echo $v
    vfile=$data_dir'/'$v
    mv $vfile $eval_dir
done

test_files=$(ls $data_dir | shuf -n $num_test)
for t in $test_files;
do
    echo $t
    tfile=$data_dir'/'$t
    mv $tfile $test_dir
done
