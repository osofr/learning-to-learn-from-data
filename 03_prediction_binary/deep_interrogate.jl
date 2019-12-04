#!/usr/bin/julia

# cd ~/funcGrad.jl/examples/ex2/
# julia

include("maximinNN1.jl")
main = maximinNN1.main

include("findMinMaxEpoch.jl")
findMMEpoch = findMinMaxEpoch.findMMEpoch

################################################################################
# linear_logistic_2d_n50: 2d linear-logistic regression at n=50

epoch = findMMEpoch("linear_logistic_2d_n50_borders")

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./linear_logistic_2d_n50_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./linear_logistic_2d_n50_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################
# linear_logistic_10d_n50_borders: 10d linear-logistic regression at n=50

epoch = findMMEpoch("linear_logistic_10d_n50_borders")

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-2.0, 2.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-2.0, 2.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################
# linear_logistic_10d_n50_nointercept_borders: 10d linear-logistic regression at n=50

epoch = findMMEpoch("linear_logistic_10d_n50_nointercept_borders")

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.0, 0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.0, 0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.0, 0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.0, 0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################
# linear_logistic_10d_n50_pm1_borders: 10d linear-logistic regression at n=50

epoch = findMMEpoch("linear_logistic_10d_n50_pm1_borders")

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_pm1_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_pm1_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################
# NN3_2d_n50_mixed: 2d NN with 1 hidden layer (3 nodes) at n=50.

files=broadcast(x->string(x)*"_maxRisk_5starts.jld2",3510:10:4000)
epoch = findMMEpoch("NN3_2d_n50_mixed";files=files)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN3_2d_n50_mixed/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN3_2d_n50_mixed/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################
# NN33_2d_n50_mixed: 2d NN with 1 hidden layer (3 nodes) at n=50.

files=broadcast(x->string(x)*"_maxRisk_5starts.jld2",3010:10:3500)
epoch = findMMEpoch("NN33_2d_n50_mixed";files=files)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN33_2d_n50_mixed/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 3 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN33_2d_n50_mixed/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 3 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################
# NN3_2d_n50_mixed_glmInit: 2d NN with 1 hidden layer (3 nodes) at n=50. linear model coefficients provided to T

files=broadcast(x->string(x)*"_maxRisk_5starts.jld2",1250:50:1500)
epoch = findMMEpoch("NN3_2d_n50_mixed_glmInit";files=files)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN3_2d_n50_mixed_glmInit/ --saveEvery 25 --maxRiskEvery 50 --hiddenT 50 --innerSGAnsteps 0 --verbose 1 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 --maxRiskInit 1 --glmInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN3_2d_n50_mixed_glmInit/ --saveEvery 25 --maxRiskEvery 50 --hiddenT 50 --innerSGAnsteps 0 --verbose 1 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 --maxRiskInit 1 --glmInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################
# NN33_2d_n50_mixed_glmInit: 2d NN with 1 hidden layer (3 nodes) at n=50. linear model coefficients provided to T

files=broadcast(x->string(x)*"_maxRisk_5starts.jld2",1250:50:1500)
epoch = findMMEpoch("NN33_2d_n50_mixed_glmInit";files=files)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN33_2d_n50_mixed_glmInit/ --saveEvery 25 --maxRiskEvery 50 --hiddenT 50 --innerSGAnsteps 0 --verbose 1 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 3 --maxRiskInit 1 --glmInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN33_2d_n50_mixed_glmInit/ --saveEvery 25 --maxRiskEvery 50 --hiddenT 50 --innerSGAnsteps 0 --verbose 1 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 3 --maxRiskInit 1 --glmInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))



################################################################################

# cd4_linear_logistic_2d_n50_pm05_borders: 2d linear-logistic regression at n=50, model bounds 0.5

epoch = findMMEpoch("cd4_linear_logistic_2d_n50_pm05_borders")

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_cd4 --odim 3 --n 50 --outdir ./cd4_linear_logistic_2d_n50_pm05_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 100 --repsPerGamma 500 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_cd4 --odim 3 --n 50 --outdir ./cd4_linear_logistic_2d_n50_pm05_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 100 --repsPerGamma 500 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################

# cd4_linear_logistic_2d_n50_pm1_borders: 2d linear-logistic regression at n=50, model bounds 1.0

epoch = findMMEpoch("cd4_linear_logistic_2d_n50_pm1_borders")

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_cd4 --odim 3 --n 50 --outdir ./cd4_linear_logistic_2d_n50_pm1_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]] --psirange [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 100 --repsPerGamma 500 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_cd4 --odim 3 --n 50 --outdir ./cd4_linear_logistic_2d_n50_pm1_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]] --psirange [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 100 --repsPerGamma 500 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################

# cd4_linear_logistic_2d_n50_pm2_borders: 2d linear-logistic regression at n=50, model bounds 2.0

epoch = findMMEpoch("cd4_linear_logistic_2d_n50_pm2_borders")

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_cd4 --odim 3 --n 50 --outdir ./cd4_linear_logistic_2d_n50_pm2_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-2.0, 2.0], [-2.0, 2.0], [-1.0, 1.0]] --psirange [[-2.0, 2.0], [-2.0, 2.0], [-1.0, 1.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 100 --repsPerGamma 500 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_cd4 --odim 3 --n 50 --outdir ./cd4_linear_logistic_2d_n50_pm2_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-2.0, 2.0], [-2.0, 2.0], [-1.0, 1.0]] --psirange [[-2.0, 2.0], [-2.0, 2.0], [-1.0, 1.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 100 --repsPerGamma 500 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################
# linear_logistic_10d_n50_nointercept_borders_rho3: 10d linear-logistic regression at n=50

epoch = findMMEpoch("linear_logistic_10d_n50_nointercept_borders_rho3")

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_rho3 --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders_rho3/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_rho3 --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders_rho3/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################
# linear_logistic_10d_n50_nointercept_borders_rho6: 10d linear-logistic regression at n=50

epoch = findMMEpoch("linear_logistic_10d_n50_nointercept_borders")

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_rho6 --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders_rho6/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_rho6 --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders_rho6/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))

################################################################################
# linear_logistic_10d_n50_nointercept_borders_rho9: 10d linear-logistic regression at n=50

epoch = findMMEpoch("linear_logistic_10d_n50_nointercept_borders")

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_rho9 --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders_rho9/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --udim 3 --hiddenPi 20 20 --nbatchPi 250 --interrogationStarts 50 --numInterDS 10 --interrogateEpoch "*string(epoch))

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_rho9 --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders_rho9/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --innerSGAnsteps 0 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --verbose 0 --ntest 10 --hiddenPi 0 --interrogationStarts 250 --numInterDS 25 --interrogationngridBorder 1 --interrogationngrid 1 --interrogationSE 1 --interrogateEpoch "*string(epoch))
