#!/bin/bash
#SBATCH -J mcs_data_collection # name of job
#SBATCH -A eecs                # sponsored account
#SBATCH -p gpu                 # partition or queue
#SBATCH -o dvis_data.out       # output file
#SBATCH -e dvis_data.err       # error file
#SBATCH --ntasks-per-node=5    # num CPUs
#SBATCH --nodelist=cn-gpu2
#SBATCH --gres=gpu:1           # request 1 GPU
#SBATCH --time 1-00:12:00      # 1 day


export MAX_TIME=1e5 # ~ 28 hrs
srun -N1 -n1 sleep $MAX_TIME &

module load gcc/6.5
module load cuda/10.1
nvidia-smi

cd /nfs/hpc/share/$USER

conda activate mcs_opics
if [ $? -eq 0 ]; then
  :
else
  # Create the environment and activate
  conda create --yes -n mcs_opics python=3.6.8
  conda activate mcs_opics
fi
python -V 

if ! [ -d "mcs_opics" ]; then
  # get git repo, cd to it, select specific branch
  git clone https://github.com/MCS-OSU/mcs_opics.git
fi
cd mcs_opics
git stash
git checkout component/interactive-env.diego
git pull &>/dev/null

export OUR_XPID=`nvidia-smi | grep Xorg | awk '{print $5}'`
export DISPLAY=`pgrep Xorg | xargs ps | grep $OUR_XPID | awk '{print $6}'`
echo "GPU's DISPLAY id"; printenv DISPLAY

mkdir .tmp -p 
if ! pip list | grep machine-common-sense; then
  # /tmp on cn-gpu2 server can be full
  TMPDIR=.tmp python -m pip install --cache-dir .tmp --build ./tmp git+https://github.com/NextCenturyCorporation/MCS@0.4.1
fi

pip install --user -r requirements.txt
pip show numpy
pip show machine_common_sense
#bash setup_torch.sh
#bash setup_unity.sh &> /dev/null
#bash setup_vision.sh
export PYTHONPATH=$PWD





# Data Collection

cp -r /nfs/hpc/share/$USER/diego_bkup_2/retrieval_scenes_e4/ interaction_scenes/

# 66, 132, 198, 264, 330, 396, 462, 528, 594 and 660
python vision/generateData/simple_task_multi.py 0 66 &
python vision/generateData/simple_task_multi.py 132 66 & 
python vision/generateData/simple_task_multi.py 198 66 &
python vision/generateData/simple_task_multi.py 264 268 &
# python vision/generateData/simple_task_multi.py 330 66 &
# python vision/generateData/simple_task_multi.py 396 66 &
# python vision/generateData/simple_task_multi.py 462 66 &
# python vision/generateData/simple_task_multi.py 528 66 &
# python vision/generateData/simple_task_multi.py 660 66 &
# python vision/generateData/simple_task_multi.py 726 66 &
# python vision/generateData/simple_task_multi.py 792 66 &
# python vision/generateData/simple_task_multi.py 858 66 &
# python vision/generateData/simple_task_multi.py 924 66 &
sleep 66
nvidia-smi # let's see how the GPU is doing!
# increase prob that this next 1 finishes last
# python vision/generateData/simple_task_multi.py 990 86

$? stores the exit code of the most recently finished process
if [[ $? = 0 ]]; then
    will check output for this to confirm success!
    echo 69420
fi

# ensure everything is easily shared
chmod -R 775 /nfs/hpc/share/$USER/mcs_opics

# Training
###########################################################################################
# chmod -R 775 /nfs/hpc/share/$USER/mcs_opics

# echo "installing training libraries"
# pip install cython &> /dev/null
# pip install opencv-python pillow pycocotools matplotlib &> /dev/null 
# pip install scikit-image &> /dev/null
# pip install scipy==1.2.0 &> /dev/null
# pip install tensorboardX &> /dev/null
# echo "finished installing training libraries"

# # From main mcs_opics
# cd visionmodule
# mkdir -p cd data/dataset/mcsvideo/interaction_scenes/

# cp -r /nfs/hpc/share/$USER/diego_bkup_2 data/dataset/mcsvideo/interaction_scenes/
# cp /nfs/hpc/share/$USER/diego_bkup_2/train.txt data/dataset/mcsvideo/interaction_scenes/
# cp /nfs/hpc/share/$USER/diego_bkup_2/eval.txt data/dataset/mcsvideo/interaction_scenes/

# touch logs_ms.txt
# python train.py --scripts=mcsvideo3_inter_unary_pw.json >> logs_ms.txt

# chmod -R 775 /nfs/hpc/share/$USER/mcs_opics/visionmodule/Result/

########################################################################################

ps -ef | grep nvidia-smi | grep -v grep | awk '{print $2}' | xargs kill
# kill the process keeping this slurm job open
pgrep sleep | xargs ps | grep $MAX_TIME | xargs echo | cut -d ' ' -f 1 | xargs kill

