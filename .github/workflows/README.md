## PERMISSIONS:
Have your major advisor email Rob Yelle (robert.yelle@oregonstate.edu) asking to give you permission to the High Performance Computing (HPC) cluster. 

## INTERACTIVE SHEEL WITH GPU:
```
ssh $USER@flip.engr.oregonstate.edu 
ssh $USER@submit-b.hpc.engr.oregonstate.edu 
module load slurm 
# request an interactive bash shell on cn-gpu2 server with gpu(s) 
srun -A eecs -p gpu --nodelist=cn-gpu2 --pty bash --gres=gpu:1 # can be > 1 
```  
## SETUP GITHUB ACTIONS RUNNER:

2 minute github action tutorial. Useful for setting up your own runner or workflow.
https://youtu.be/GHVSRc1BYCc%20Github%20Actions%20Tutorial 

```
ssh $USER@flip.engr.oregonstate.edu 
ssh $USER@pelican04.eecs.oregonstate.edu 
ssh-keygen -t rsa -b 4096 -C "$USER@pelican03.eecs.oregonstate.edu" 
cat .ssh/id_rsa.pub | ssh $USER@submit-b.hpc.engr.oregonstate.edu 'cat >> .ssh/authorized_keys' 
tmux 
```
Follow the steps of this video to setup a github action runner.
Disconnect the current tmux server by pressing Ctrl+b followed by d. Your runner will stay open after logging out. Nohup can be used to a similar effect: 

```nohup ./run.sh &```

## RESTRICTED GUI WITH GPU:
-X11 all the way thru version. Can't see unity render but can ease development by opening saving images to the file system & view them with dolphin & eog: 
 
```
dolphin & # opens file explorer 
eog selfie.png # view just 1 image
```

## MISC RESOURCES:
Useful slurm (ie sbatch, srun, etc) info: Link 1 Link 2 


 

 
