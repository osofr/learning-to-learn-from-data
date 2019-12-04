#!/usr/bin/julia
## **********************************************************************
## 1. Main script to pre-train LSTM estimator of the centers C(X)=[μ,σ] for 100K iterations (nepochs2Cx).
##    To be used for constructing the final estimator of the Confidence Region (CR).
## **********************************************************************
## * Requires running on GPU *
## **********************************************************************
## NOTE: Make sure this script is executable
## **********************************************************************
## 1: pre-train mTᵏcent against uniform prior for 100 alternating iterations (nepochs1Cx)
## 2: each iteration consists of 1 update on Piᵏ and 3 updates on  Tᵏ (1-3)
## 3: Tᵏ is parametrized by 2 connected LSTMs(50)
## 4: bigger Piᵏ.cent: MLP with single hidden layer(25)
## 5: 4 new inputs to Piᵏ.cent: mTᵏcent (C(X)) at true γ [-10,5.25] and [+10,5.25]
## **********************************************************************
## 6: mTᵏoff LSTM includes a new input, random η (same η feeds with each Xᵢ as (X₁,η), (X₂,η), (Xn,η)).
##    η is sampled from βdistribution(1.0,4.0). New η sampled for each dataset in batch (nbatch total random η).
##    η value is used when evaluating mTᵏoff quantile loss.
## **********************************************************************
## 7: mTᵏoff: Switching from SGA random grid search (update mTᵏoff based on single worst γ)
##    to updating mTᵏoff based on a sample of γs from the uniform prior (Piᵏ is fixed at uniform prior)
##    TODO: Add the function to run SGA for coverage either at set constant \eta or at randomly drawn \eta
## **********************************************************************
## 8: mTᵏoff: Quantile loss is re-scaled by the C(X) L2 loss at that γ₀
## **********************************************************************
## UNTRANSFORED LSTM (mTᵏcent / mTᵏoff) OUTPUTS:
## 1) C(X) LSTM output is as is, just empirical mean, no σ(.) final layer.
## 2) (L(X),U(X)) is as is, empirical mean (no l1 norms, abs() final layer)
## **********************************************************************

include("maximinNN1.jl")
main = maximinNN1.main
res = main("--gpu 1 --seed 5435 --xdim 50
--losstypecent L2 --kcent 2.0
--nepochs1Cx 100 --nepochs2Cx 100000 --nepochsOff 0 --maxRiskEvery 50
--alpha 0.05 --Tlossη_α 2.0 --Tlossη_β 4.0 --Tlossη1 0.20 --Tlossη2 0.50 --nbatch 500 --SGAnbatch 500 --lambdaMC 50
--Pirange [[-10.0f0, 10.0f0], [0.5f0, 10.0f0]]
--nstepsTcent 3 --nstepsPicent 1 --fixPiTcent 1
--mTmodel LSTM2x --mTreduce sum --hiddenT 50 --numLSTMcells 1 --lrT 0.0008 --β1T 0.90
--hiddenPi 25 --lrPi 0.001 --β1Pi 0.50
--SGAnstarts1 15 --SGAnstarts2 50 --SGAramp_maxepoch 20000 --SGAnruns 0
--saveEvery 10 --outdir ./saved_models_centersCx_pretrained")
