#!/bin/bash
#SBATCH -J mcs_ga_build      # name of job
#SBATCH -A eecs              # sponsored account
#SBATCH -p dgx2              # partition or queue
#SBATCH -o dvis_data.out     # output file
#SBATCH -e dvis_data.err     # error file
#SBATCH --ntasks-per-node=8 # num CPUs
#SBATCH --gres=gpu:1         # request 1 GPU
#SBATCH --time 3-12:00:00    # 3.5 days

export MAX_TIME=36e4
srun -N1 -n1 sleep $MAX_TIME &
module load gcc/6.5
module load cuda
nvidia-smi
# had to run this to set permissions for sharing:
# chmod -R 775 /nfs/hpc/share/$USER
cd /nfs/hpc/share/$USER

export CREATE_ENV=false
# if conda not setup
if ! [ -d "miniconda3" ]; then
  export CREATE_ENV=true
  mkdir -p miniconda3 &>/dev/null
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh &>/dev/null
fi
bash miniconda3/miniconda.sh -b -u -p miniconda3 &>/dev/null
miniconda3/bin/conda init bash &>/dev/null
if $CREATE_ENV; then
  conda create -n mcs_opics python=3.6.8 &>/dev/null
fi
conda activate mcs_opics
# redundancy because the above can fail
source miniconda3/bin/activate mcs_opics
python -V 

if ! [ -d "mcs_opics" ]; then
  # get git repo, cd to it, select specific branch
  git clone https://github.com/MCS-OSU/mcs_opics.git &>/dev/null
fi
cd mcs_opics
git stash
git checkout component/interactive-env
git pull &>/dev/null

export OUR_XPID=`nvidia-smi | grep Xorg | awk '{print $5}'`
export DISPLAY=`pgrep Xorg | xargs ps | grep $OUR_XPID | awk '{print $6}'`
echo "GPU's DISPLAY id"; printenv DISPLAY

# setup all the python env + dependencies
alias pip=pip3
alias python=python3
# pip install --upgrade pip &> /dev/null
# pip install --upgrade setuptools &> /dev/null
# pip install --upgrade wheel &> /dev/null
mkdir ./tmp -p 

if ! pip list | grep machine-common-sense; then
  # /tmp on cn-gpu2 server can be full
  TMPDIR=/scratch/MCS/tmp python -m pip install --cache-dir /scratch/MCS/tmp --build /scratch/MCS/tmp git+https://github.com/NextCenturyCorporation/MCS@0.4.1 &> /dev/null
fi

pip install -r requirements.txt &> /dev/null
pip show machine_common_sense
bash setup_torch.sh &> /dev/null
bash setup_unity.sh &> /dev/null
bash setup_vision.sh &> /dev/null
bash get_dataset.sh &> /dev/null
export PYTHONPATH=$PWD

# Data Collecting
# 66, 132, 198, 264, 330, 396, 462, 528, 594 and 660
# python vision/generateData/simple_task_multi.py 0 66 &
# python vision/generateData/simple_task_multi.py 132 66 & 
# python vision/generateData/simple_task_multi.py 198 66 &
# python vision/generateData/simple_task_multi.py 264 66 &
# python vision/generateData/simple_task_multi.py 330 66 &
# python vision/generateData/simple_task_multi.py 396 66 &
# python vision/generateData/simple_task_multi.py 462 66 &
# python vision/generateData/simple_task_multi.py 528 66 &
# python vision/generateData/simple_task_multi.py 660 66 &
# python vision/generateData/simple_task_multi.py 726 66 &
# python vision/generateData/simple_task_multi.py 792 66 &
# python vision/generateData/simple_task_multi.py 858 66 &
# python vision/generateData/simple_task_multi.py 924 66 &
# sleep 66
# nvidia-smi # let's see how the GPU is doing!
# increase prob that this next 1 finishes last
# python vision/generateData/simple_task_multi.py 990 86

# $? stores the exit code of the most recently finished process
# if [[ $? = 0 ]]; then
    # will check output for this to confirm success!
    # echo 69420
# fi

# ensure everything is easily shared
# chmod -R 775 /nfs/hpc/share/$USER/mcs_opics

# Training
###########################################################################################
chmod -R 775 /nfs/hpc/share/$USER/mcs_opics

echo "installing training libraries"
pip install cython &> /dev/null
pip install opencv-python pillow pycocotools matplotlib &> /dev/null 
pip install scikit-image &> /dev/null
pip install scipy==1.2.0 &> /dev/null
pip install tensorboardX &> /dev/null
echo "finished installing training libraries"

# From main mcs_opics
cd visionmodule
mkdir -p cd data/dataset/mcsvideo/interaction_scenes/

cp -r /nfs/hpc/share/$USER/diego_bkup_2 data/dataset/mcsvideo/interaction_scenes/
cp /nfs/hpc/share/$USER/diego_bkup_2/train.txt data/dataset/mcsvideo/interaction_scenes/
cp /nfs/hpc/share/$USER/diego_bkup_2/eval.txt data/dataset/mcsvideo/interaction_scenes/


nvidia-smi
touch logs_ms.txt
python train.py --scripts=mcsvideo3_inter_unary_pw.json >> logs_ms.txt
nvidia-smi 

chmod -R 775 /nfs/hpc/share/$USER/mcs_opics/visionmodule/Result/


########################################################################################

# kill the process keeping this slurm job open
pgrep sleep | xargs ps | grep $MAX_TIME | xargs echo | cut -d ' ' -f 1 | xargs kill
