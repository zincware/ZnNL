#!/usr/bin/env bash

# Generate test files for the neurips data


for i in 20 30 40 50 60 70 80 90 100
do
  cp submit.sh ${i}_analysis/submit.sh
  cd ${i}_analysis
  sbatch submit.sh
  cd ../
done
