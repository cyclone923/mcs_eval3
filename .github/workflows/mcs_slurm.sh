#!/bin/bash
#SBATCH -J mcs_ga_build  # name of job
#SBATCH -A eecs          # sponsored account
#SBATCH -p gpu           # partition or queue
#SBATCH -o mcs.out       # output file
#SBATCH -e mcs.err       # error file
#SBATCH --gres=gpu:1     # request 1 GPU
#SBATCH --nodelist=cn-gpu2
 
MAX_TIME = 1e4 # ~ 3hrs
srun -N1 -n1 sleep $MAX_TIME &
module load gcc/6.5
module load cuda
nvidia-smi
cd /scratch/MCS

# if conda not setup
if ! [ -d "miniconda3" ]; then
 mkdir -p miniconda3 &>/dev/null
 wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh &>/dev/null
 bash miniconda3/miniconda.sh -b -u -p miniconda3 &>/dev/null
 miniconda3/bin/conda init bash &>/dev/null
 conda create -n mcs_opics python=3.6.8 &>/dev/null
fi
conda activate mcs_opics &>/dev/null
python -V

# get git repo, cd to it, select specific branch
rm -fr mcs_opics
git clone https://github.com/MCS-OSU/mcs_opics.git &>/dev/null
cd mcs_opics 
git checkout test.yaml-rob 
git pull &>/dev/null

export OUR_XPID=`nvidia-smi | grep Xorg | awk '{print $5}'`
export DISPLAY=`pgrep Xorg | xargs ps | grep $OUR_XPID | awk '{print $6}'`
echo "GPU's DISPLAY id"; printenv $DISPLAY

# setup all the python env + dependencies
pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade wheel
mkdir ./tmp -p 
# /tmp on cn-gpu2 server can be full
TMPDIR=/scratch/MCS/tmp python -m pip install --cache-dir /scratch/MCS/tmp --build /scratch/MCS/tmp git+https://github.com/NextCenturyCorporation/MCS@0.3.8
pip install -r requirements.txt &> /dev/null
pip show machine_common_sense
bash setup_torch.sh &> /dev/null
bash setup_unity.sh &> /dev/null
bash setup_vision.sh &> /dev/null
export PYTHONPATH=$PWD

python get_gravity_scenes.py &> /dev/null
cp gravity_scenes/gravity_support_ex_01.json different_scenes
# agent code in this branch explodes
rm different_scenes/preference_0001_01.json
python eval.py

# will check output for this to confirm success!
echo 69420

# kill the process keeping this slurm job open
pgrep sleep | xargs ps | grep $MAX_TIME | xargs echo | cut -d ' ' -f 1 | xargs kill
