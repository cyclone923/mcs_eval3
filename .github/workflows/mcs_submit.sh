#!/bin/bash

source /etc/profile
module load slurm

chmod +x /tmp/mcs_slurm.sh
SJOBID=$(sbatch --parsable /tmp/mcs_slurm.sh)

echo $SJOBID

# wait for gpu job to finish
until [ `sacct -j $SJOBID | grep COMPLETED | wc -l` -gt 0 -o `sacct -j $SJOBID | grep CANCELLED | wc -l` -gt 0 ]
do
  sleep 5
done

# copy output to github actions logs
echo '==== STDOUT ===='
cat mcs.out
echo '==== STDERR ===='
cat mcs.err
 
# check if the build failed/passed
if ! grep 69420 mcs.out; 
  then exit 404; 
fi
