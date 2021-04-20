# makes 'module load' work
source /etc/profile
module load slurm

RES=$(sbatch /tmp/mcs_slurm.sh)
SJOBID=${RES##*}
# wait for gpu job to finish
until [ `sacct -j $SJOBID | grep COMPLETED | wc -l` -gt 0 -o `sacct -h $SJOBID | grep CANCELLED | wc -l` -gt 0 ]
do
  sleep 5
done

# copy output to github actions logs
echo '==== STDOUT ===='
cat mcs.out
echo '==== STDERR ===='
cat mcs.err

# check if the build failed/passed
if ! cat mcs.out | grep 69420; 
  then exit 404; 
fi
