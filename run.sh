#!/bin/bash
#PBS -l nodes=1:gold6258r:ppn=1
#PBS -N gnn
#PBS -j oe
#PBS -o output.log

cd ${PBS_O_WORKDIR}

source activate torch

python preprocess.py
python main.py --epochs 10
