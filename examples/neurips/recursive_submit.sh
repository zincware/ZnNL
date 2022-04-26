#!/usr/bin/env bash

# Generate test files for the neurips data


for i in 10 15 20 25 30 35 40 50 60
do
  cp submit.sh ${i}_analysis/submit.sh
  cd ${i}_analysis
  sbatch submit.sh
  cd ../
done
