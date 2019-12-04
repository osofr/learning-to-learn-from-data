#!/bin/bash
## **********************************************************************
## 2. Train offsets for CRs, given pre-trained precedure for the centers
## nepochs specifies the total number of iterations for the training procedure
## **********************************************************************
## * Requires running on GPU *
## **********************************************************************
## NOTE: Make sure this script is executable
## **********************************************************************
## Restarts from latest checkpoint everytime Flux fails
## **********************************************************************

icount=0
reloadEvery=1000
nepochs=160000
maxicount=$(($nepochs / $reloadEvery - 1))
echo reloadEvery: $reloadEvery
echo nepochs: $nepochs
echo maxicount: $maxicount

while [ $icount -le $maxicount ]
do
  echo icount: $icount
  ./02_main_train_script_offsetsLU.jl $icount $reloadEvery $nepochs >> 02_main_train_script_offsetsLU.out

  ((icount++))
  sleep 5
done

echo All done
