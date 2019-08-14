#!/bin/bash
#SBATCH -J ggnn-pre
#SBATCH -o ggnn-pre-result.txt
#SBATCH -t 01:01:01
#SBATCH -N 1 -n 2
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=64GB
#SBATCH -A alipour

function load_packages {
    module load CMake/3.12.2
    module load Anaconda3
    module load GCC/7.2.0-2.29
    export CC=/project/cacds/apps/easybuild/software/GCCcore/7.2.0/bin/gcc
    export CXX=/project/cacds/apps/easybuild/software/GCCcore/7.2.0/bin/gcc
    module load CUDA/9.0.176
    export CUDA_TOOLKIT_ROOT_DIR=/project/cacds/apps/easybuild/software/CUDA/9.0.176
}

function main_fn {
    cd GGNN-Discriminator/
    python main.py "../Graphs-Pre/" --output "../Run/ggnn-pre-log.txt"
}

cd /project/alipour/rabin/tool/fork-invariant-validation/
load_packages
main_fn
