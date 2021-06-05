#!/bin/bash
#PBS -l nodes=1:gold6258r:ppn=1
#PBS -N gnn
#PBS -j oe
#PBS -o output.log

cd ${PBS_O_WORKDIR}

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
TOTAL_CORES=28

export $KMP_SETTING
echo -e "### using $KMP_SETTING"

export OMP_NUM_THREADS=$TOTAL_CORES
echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"

PYTHON=/home/u75549/anaconda3/envs/torch/bin/python

${PYTHON} preprocess.py
${PYTHON} main.py --epochs 10
