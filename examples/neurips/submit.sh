#! /bin/bash

# Input script for mdsuite analysis run.

### ---------------------------------------- ###
### Input parameters for resource allocation ###
### ---------------------------------------- ###

#SBATCH -J neurips22_RND
#SBATCH -N 1

### ----------------------------------------- ###
### Input parameters for logging and run time ###
### ----------------------------------------- ###

#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stovey@icp.uni-stuttgart.de

### -------------------- ###
### Modules to be loaded ###
### -------------------- ###

module load gcc/8.4.0     	 # gcc compiler
module load miniconda3

source /home/${USER}/.bashrc

conda activate neurips

### ------------------------------------- ###
### Change into working directory and run ###
### ------------------------------------- ###

cd $SLURM_SUBMIT_DIR  # change into working directory
python analysis.py > output.txt
