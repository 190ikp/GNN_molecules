#!/bin/bash
#PBS -l nodes=1:gold6258r:ppn=1
#PBS -N gnn
#PBS -j oe
#PBS -o output.log

cd ${PBS_O_WORKDIR}

PYTHON=/home/u75549/anaconda3/envs/torch/bin/python
${PYTHON} preprocess.py
${PYTHON} main.py --epochs 10
