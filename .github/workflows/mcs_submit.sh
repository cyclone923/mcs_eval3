rm -fr mcs_opics &> /dev/null
git clone https://github.com/MCS-OSU/mcs_opics.git &> /dev/null
cd mcs_opics
git checkout test.yaml-rob &> /dev/null
git pull &> /dev/null

# makes 'module load' work
source /etc/profile
module load slurm

SJOBID=$(sbatch --parsable /tmp/mcs_slurm.sh)

# wait for gpu job to finish
until [ `sacct -j $SJOBID | grep COMPLETED | wc -l` -gt 0 -o `sacct -j $SJOBID | grep CANCELLED | wc -l` -gt 0 ]
do
  sleep 5
done

# copy output to github actions logs
echo '==== STDOUT ===='
cat ../mcs.out
echo '==== STDERR ===='
cat ../mcs.err

# check if the build failed/passed
if ! grep 69420 ../mcs.out; 
  then exit 404; 
fi
