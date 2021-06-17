#!/bin/bash
<<<<<<< HEAD
#PBS -l nodes=1:gold6258r:ppn=1
=======
#PBS -l nodes=1:gold6128:ppn=1
>>>>>>> ba02d5a95e8ff9b48159ff8de0b821f39bd9174f
#PBS -N gnn
#PBS -j oe
#PBS -o output.log

cd ${PBS_O_WORKDIR}

<<<<<<< HEAD
KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
TOTAL_CORES=28

export ${KMP_SETTING}
echo -e "### using ${KMP_SETTING}"

export OMP_NUM_THREADS=${TOTAL_CORES}
echo -e "### using OMP_NUM_THREADS=${TOTAL_CORES}"

PYTHON=${HOME}/anaconda3/envs/torch/bin/python

${PYTHON} preprocess.py
${PYTHON} main.py --epochs 10
=======
CPUS=$(cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l)
CORES=$(cat /proc/cpuinfo | grep "core id" | sort | uniq | wc -l)
TOTAL_CORES=$(expr ${CPUS} \\* ${CORES})

echo "CPUS=${CPUS} CORES=${CORES} TOTAL_CORES=${TOTAL_CORES}"
export OMP_NUM_THREADS=${TOTAL_CORES}
echo -e "### using OMP_NUM_THREADS=${TOTAL_CORES}"

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export ${KMP_SETTING}
echo -e "### using ${KMP_SETTING}"

PYTHON=${HOME}/.conda/envs/torch/bin/python

${PYTHON} preprocess.py
${PYTHON} main.py --epochs 5
>>>>>>> ba02d5a95e8ff9b48159ff8de0b821f39bd9174f
