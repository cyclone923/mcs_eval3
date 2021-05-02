pip install cython &> /dev/null
pip install opencv-python pillow pycocotools matplotlib &> /dev/null 
pip install scikit-image &> /dev/null
pip install scipy==1.2.0 &> /dev/null
pip install tensorboardX &> /dev/null

# From main mcs_opics
cd visionmodule
mkdir -p cd data/dataset/mcsvideo/interaction_scenes/

cp -r /nfs/hpc/share/$USER/diego_bkup_2 data/dataset/mcsvideo/interaction_scenes/
cp /nfs/hpc/share/$USER/diego_bkup_2/train.txt data/dataset/mcsvideo/interaction_scenes/
cp /nfs/hpc/share/$USER/diego_bkup_2/eval.txt data/dataset/mcsvideo/interaction_scenes/

python train.py --scripts=mcsvideo3_inter_unary_pw.json