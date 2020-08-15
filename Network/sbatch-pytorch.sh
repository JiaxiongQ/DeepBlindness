#!/bin/bash
#SBATCH --job-name=deepblur-cuhk-mask
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --partition=aliyun1
#SBATCH --framework=pytorch
#SBATCH --workdir=/tmp
#SBATCH --output=./_log_sbatch/%j-%N.out
#SBATCH --error=./_log_sbatch/%j-%N.err

# If you'd like to install other python packages, e.g. gym, run
#   pip install --user numpy

# You can test by running
#  pip install --user gym
#  python -c "import gym" -> no error


source /home/qiujiaxiong/anaconda3/bin/activate pytorch0.4
export PYTHONPATH=/home/qiujiaxiong/anaconda3/envs/pytorch0.4/lib/python3.6/site-packages
NPNUM=8
CONDA_CMD='python /share2/public/fail_safe/kitti/DeepBlur/main.py'

bash -c "source /home/qiujiaxiong/anaconda3/bin/activate pytorch0.4; export PYTHONPATH=/home/qiujiaxiong/anaconda3/envs/pytorch0.4/lib/python3.6/site-packages; ${CONDA_CMD}"



