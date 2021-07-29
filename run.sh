#!/bin/bash
#PBS -l nodes=1:gold6128:ppn=1
#PBS -N gnn
#PBS -j oe
#PBS -o output.log

cd "${PBS_O_WORKDIR}" || exit

CPUS=$(grep "physical id" /proc/cpuinfo | sort | uniq | wc -l)
CORES=$(grep "core id" /proc/cpuinfo | sort | uniq | wc -l)
TOTAL_CORES=$(("${CPUS}" * "${CORES}"))

echo "CPUS=${CPUS} CORES=${CORES} TOTAL_CORES=${TOTAL_CORES}"
export OMP_NUM_THREADS=${TOTAL_CORES}
echo -e "### using OMP_NUM_THREADS=${TOTAL_CORES}"

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_SETTING
echo -e "### using ${KMP_SETTING}"

PYTHON=${HOME}/.conda/envs/torch/bin/python

${PYTHON} preprocess.py
${PYTHON} main.py --epochs 5
