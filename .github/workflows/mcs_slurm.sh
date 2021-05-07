#!/bin/bash
#SBATCH -J mcs_ga_build  # name of job
#SBATCH -A eecs          # sponsored account
#SBATCH -p gpu           # partition or queue
#SBATCH -o mcs.out       # output file
#SBATCH -e mcs.err       # error file
#SBATCH --gres=gpu:1     # request 1 GPU
#SBATCH --nodelist=cn-gpu2

export MAX_TIME=1e4 # ~ 3hrs
srun -N1 -n1 sleep $MAX_TIME &
module load gcc/6.5
module load cuda
nvidia-smi
cd /scratch/MCS

module load anaconda
conda activate mcs_opics
if [ $? -eq 0 ]; then
  :
else
  # Create the environment and activate
  conda env create -n mcs_opics python=3.6.8
  conda activate mcs_opics
fi
python -V

if ! [ -d "mcs_opics" ]; then
  # get git repo, cd to it, select specific branch
  git clone https://github.com/MCS-OSU/mcs_opics.git &>/dev/null
fi
cd mcs_opics
git stash
git checkout develop
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
export PYTHONPATH=$PWD

python get_gravity_scenes.py &> /dev/null
cp gravity_scenes/gravity_support_ex_01.json different_scenes
# agent code in this branch explodes
rm different_scenes/preference_0001_01.json

# real magic!
python eval.py
# $? stores the exit code of the most recently finished process
if [[ $? = 0 ]]; then
    # will check output for this to confirm success!
    echo 69420
fi

# kill the process keeping this slurm job open
pgrep sleep | xargs ps | grep $MAX_TIME | xargs echo | cut -d ' ' -f 1 | xargs kill
