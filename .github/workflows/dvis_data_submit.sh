#!/bin/bash

SJOBID=$(sbatch --parsable /tmp/dvis_data_slurm.sh)

# wait for gpu job to finish
until [ `sacct -j $SJOBID | grep COMPLETED | wc -l` -gt 0 -o `sacct -j $SJOBID | grep CANCELLED | wc -l` -gt 0 ]
do
  sleep 5
done

# copy output to github actions logs
echo '==== STDOUT ===='
cat dvis_data.out
echo '==== STDERR ===='
cat dvis_data.err
 
# check if the build failed/passed
if ! grep 69420 dvis_data.out;
  then exit 404;
fi
