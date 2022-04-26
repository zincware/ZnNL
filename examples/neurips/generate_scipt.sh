#!/usr/bin/env bash

# Generate test files for the neurips data

base_file='mnist_parent.py'

for i in 10 15 20 25 30 35 40 50 60
do
  mkdir ${i}_analysis
  sed "s/DS_SIZE/${i}/g" ${base_file} > ${i}_analysis/analysis.py
done
