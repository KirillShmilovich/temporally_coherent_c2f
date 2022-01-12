#!/bin/sh

#SBATCH --job-name=ADP
#SBATCH --out=ADP.out
#SBATCH --partition=fela
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu 
#SBATCH --cpus-per-task=10   # set this to the desired number of threads
#SBATCH --mem=17000

# LOAD MODULES
module load cuda/10.1
module load python 
#/project2/andrewferguson/Kirill/conda_env/bin/conda activate /project2/andrewferguson/Kirill/conda_env

# DO COMPUTE WORK
/project2/andrewferguson/Kirill/conda_env/bin/python -u train_ADP_restart.py
