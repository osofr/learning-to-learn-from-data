#!/usr/bin/julia

# cd ~/funcGrad.jl/examples/ex2/
# julia

include("maximinNN1.jl")
main = maximinNN1.main

################################################################################
# Setting i (Table 1 in paper)
# linear_logistic_2d_n50: 2d linear-logistic regression at n=50

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./linear_logistic_2d_n50_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --ntest 100 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 2")


# ################################################################################
# # Code for 10d linear-logistic regression at n=50
# Setting vi (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-2.0, 2.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --ntest 100 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 50")

    # ################################################################################
# # Code for 10d linear-logistic regression at n=50, intercept in pm1
# Setting v (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_pm1_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-1.0,1.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-1.0,1.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --ntest 100 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 50")


  # ################################################################################
# # Code for 10d linear-logistic regression at n=50, intercept fixed at zero
# Setting iv (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --ntest 100 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 50")

################################################################################

# NN3_2d_n50_mixed: 2d NN with 1 hidden layer (3 nodes) at n=50, some draws from border.
# Setting ii (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN3_2d_n50_mixed/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 --maxRiskInit 1 --ntest 150 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 50 --innerSGAnstartsMixed 50")

################################################################################
# NN33_2d_n50_mixed: 2d NN with 2 hidden layers (3 nodes) at n=50.
# Setting iii (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN33_2d_n50_mixed/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 3 --maxRiskInit 1 --ntest 150 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 50 --innerSGAnstartsMixed 50")

################################################################################
################################################################################


# CD4 Examples


################################################################################

# cd4_linear_logistic_2d_n50: 2d linear-logistic regression at n=50
# Setting x (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_cd4 --odim 3 --n 50 --outdir ./cd4_linear_logistic_2d_n50_pm05_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 1000 --repsPerGamma 500 --nT 2 --hiddenReg 0 --maxRiskInit 1 --ntest 100 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 2")

# Setting xi (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_cd4 --odim 3 --n 50 --outdir ./cd4_linear_logistic_2d_n50_pm1_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]] --psirange [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 1000 --repsPerGamma 500 --nT 2 --hiddenReg 0 --maxRiskInit 1 --ntest 100 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 2")

# Setting xii (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_cd4 --odim 3 --n 50 --outdir ./cd4_linear_logistic_2d_n50_pm2_borders/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-2.0, 2.0], [-2.0, 2.0], [-1.0, 1.0]] --psirange [[-2.0, 2.0], [-2.0, 2.0], [-1.0, 1.0]] --niter 20 --gpu 1 --nbatchT 200 --innerSGAntest 1000 --repsPerGamma 500 --nT 2 --hiddenReg 0 --maxRiskInit 1 --ntest 100 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 2")

################################################################################
# NN3_2d_n50_mixed_glmInit: 2d NN with 1 hidden layer (3 nodes) at n=50, some draws from border.
# 	glmInit=1 gives the estimator access to coefficient estimates from linear model

# aware variant of Setting ii (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN3_2d_n50_mixed_glmInit/ --saveEvery 25 --maxRiskEvery 50 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 --maxRiskInit 1 --ntest 150 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 50 --innerSGAnstartsMixed 50 --glmInit 1")

################################################################################
# NN33_2d_n50_mixed_glmInit: 2d NN with 2 hidden layers (3 nodes) at n=50.
# 	glmInit=1 gives the estimator access to coefficient estimates from linear model

# aware variant of Setting iii (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit --odim 3 --n 50 --outdir ./NN33_2d_n50_mixed_glmInit/ --saveEvery 25 --maxRiskEvery 50 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-2.0, 2.0]] --psirange [[-2.0, 2.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 3 3 --maxRiskInit 1 --ntest 150 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 50 --innerSGAnstartsMixed 50 --glmInit 1")




  # ################################################################################
# linear_logistic_10d_n50_nointercept_borders_rho3: 10d linear-logistic regression at n=50, intercept fixed at zero, correlation between predictors equals 0.3
# Setting vii (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_rho3 --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders_rho3/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --ntest 100 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 50")

# linear_logistic_10d_n50_nointercept_borders_rho6: 10d linear-logistic regression at n=50, intercept fixed at zero, correlation between predictors equals 0.6
# Setting viii (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_rho6 --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders_rho6/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --ntest 100 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 50")

# linear_logistic_10d_n50_nointercept_borders_rho9: 10d linear-logistic regression at n=50, intercept fixed at zero, correlation between predictors equals 0.9
# Setting ix (Table 1 in paper)

@time main("--nepochs 100000 --seed 54321 --truepsi Ψident --name logit_rho9 --odim 11 --n 50 --outdir ./linear_logistic_10d_n50_nointercept_borders_rho9/ --saveEvery 5 --maxRiskEvery 10 --hiddenT 50 --hiddenPi 0 --innerSGAnsteps 0 --verbose 1 --parsrange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --psirange [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [0.0,0.0]] --niter 20 --gpu 1 --nbatchT 150 --innerSGAntest 1000 --repsPerGamma 1000 --nT 2 --hiddenReg 0 --maxRiskInit 1 --ntest 100 --SGAnstarts 200 --SGAnruns 0 --SGAnbatch 1000 --innerSGAnstartsBorder 50")
