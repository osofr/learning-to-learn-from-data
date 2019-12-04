#!/usr/bin/julia
## **********************************************************************
## 2. Train offsets for CRs, given pre-trained precedure for the centers
## Training will continue for the number of iterations specified by nepochsOff
## **********************************************************************
## NOTE: DO NOT RUN THIS SCRIPT DIRECTLY FOR ACTUAL TRAINING OF THE OFFSETS
## To train offsets CR procedure, run bash script: 02_main_train_script_offsetsLU.sh
## This will allow to automatically resume training from the latest check-point at each failure
## **********************************************************************
## * Requires running on GPU *
## **********************************************************************
## NOTE: Make sure this script is executable
## Use Bigger LSTM(100) for offsets L(X)/U(X)
## Using the best centers LSTM C(X)=[μ,σ] over 100K iterations (from epoch/iteration 72,200 @ max risk 0.034127563f0) as input
## **********************************************************************

icount = parse(Int, ARGS[1])
reloadEvery = parse(Int, ARGS[2])
nepochs = parse(Int, ARGS[3])

println("icount.jl: $icount; $(typeof(icount))")
println("reloadEvery.jl: $reloadEvery; $(typeof(reloadEvery))")
println("nepochs.jl: $nepochs; $(typeof(nepochs))")

include("maximinNN1.jl")
main = maximinNN1.main

success = 0
fail = 0
while (success == 0 && fail <= 10)
  try
    println("------------------------------------------------------------------")
    println("...re-attempting to run #: $fail...")
    println("------------------------------------------------------------------")
    res = main("--gpu 1 --seed 3554 --seedreload $(icount+20) --xdim 50 --reloadEvery $reloadEvery --iepoch $icount
    --kcent 2.0 --losstypecent L2
    --nepochs1Cx 0 --nepochs2Cx 0 --nepochsOff $nepochs --maxRiskEvery 50
    --alpha 0.05 --Tlossη_α 1.5 --Tlossη_β 4.0 --Tlossη1 0.10 --Tlossη2 0.05 --nbatch 500 --SGAnbatch 500 --lambdaMC 50
    --Pirange [[-10.0f0, 10.0f0], [0.5f0, 10.0f0]]
    --hiddenPi 25 --lrPi 0.001 --β1Pi 0.50
    --nstepsTcent 3 --nstepsPicent 1 --fixPiTcent 1
    --mTmodel LSTM2x --mTreduce sum --hiddenT 100 --numLSTMcells 1 --lrT 0.0005 --β1T 0.90
    --SGAnstarts1 20 --SGAnstarts2 70 --SGAramp_maxepoch 100000 --SGAnruns 0
    --loaddir ./saved_models_centersCx_pretrained --loadfile 072200.bson
    --saveEvery 50 --outdir ./saved_models_offsetsLU")
    success += 1
  catch
    fail += 1
  end
end
