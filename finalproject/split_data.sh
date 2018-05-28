#!/bin/bash

data_dir='/home/smantra/finalproject/data'
eval_dir='/home/smantra/finalproject/eval'
total=($(wc -l <(ls $data_dir)))
num_val=220

val_files=$(ls $data_dir | shuf -n $num_val)

for v in $val_files;
do
    echo $v
    vfile=$data_dir'/'$v
    mv $vfile $eval_dir
done

#for d in $data_site_dir/*;
#do
#  if [ -d "$d" ]; then
#    echo "$d"
#    for d2 in $d/*;
#    do
#        if [ -d "$d2" ]; then
#            cp $d2/anat/*T1w.nii.gz $data_dir
#        fi
#    done
#  fi
#done
