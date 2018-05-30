#!/bin/bash

# TO DOWNLOAD
# aws s3 ls s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS/ --recursive --include "*T1w*"
# aws s3 cp s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS/ . --recursive --exclude "*" --include "*T1w*"

# aws s3 ls s3://fcp-indi/data/Projects/ABIDE/
# aws s3 ls s3://fcp-indi/data/Projects/ABIDE/RawDataBIDS/

data_site_dir='/home/smantra/finalproject/data_by_site'
data_dir='/home/smantra/finalproject/data'

for d in $data_site_dir/*;
do
  if [ -d "$d" ]; then
    echo "$d"
    for d2 in $d/*;
    do
        if [ -d "$d2" ]; then
            cp $d2/anat/*T1w.nii.gz $data_dir
        fi
    done
  fi
done
