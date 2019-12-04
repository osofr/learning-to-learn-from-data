# Write design matrices from data sets with collinear
# predictors (mtcars and Mandel) to csv files so that
# they can be read into Julia.
# Columns of design matrices are scaled to have empirical
# mean zero and variance one

library("glmnet")
library("parallel")
library("doParallel")
library("carData")
registerDoParallel(cores=95)
# setwd("set-to-current-dir")


############################################################
# mtcars
data(mtcars)
dat_mt = subset(mtcars,select=c(disp,cyl))
dat_mt = scale(dat_mt)


############################################################
# Mandel
data(Mandel)  
dat_mandel = subset(Mandel,select=c(x1,x2))
dat_mandel = scale(dat_mandel)


############################################################
nbeta = 10000  ## number of random draws of beta
mtest = 5000   ## number of test dataset for each beta draw
set.seed(12345)

############################################################
## generate test betas. Approach A (1).
Z1 = rnorm(nbeta, mean = 0, sd = 1)
Z2 = rnorm(nbeta, mean = 0, sd = 1)
Z3 = rnorm(nbeta, mean = 0, sd = 1)
Znorm = sqrt(Z1^2 + Z2^2 + Z3^2)
beta1.A1 = 10 * Z1 / Znorm
beta2.A1 = 10 * Z2 / Znorm
beta.A1 = rbind(beta1.A1, beta2.A1)
l2beta.A1 = sqrt(beta.A1[1,]^2 + beta.A1[2,]^2)
max(l2beta.A1)


############################################################
## generate test betas. Approach B (2).
Z1 = rnorm(nbeta, mean = 0, sd = 1)
Z2 = rnorm(nbeta, mean = 0, sd = 1)
U = runif(nbeta)
Znorm = sqrt(Z1^2 + Z2^2)
beta1.A2 = 10 * sqrt(U) * Z1 / Znorm
beta2.A2 = 10 * sqrt(U) * Z2 / Znorm
beta.A2 = rbind(beta1.A2, beta2.A2)
l2beta.A2 = sqrt(beta.A2[1,]^2 + beta.A2[2,]^2)
max(l2beta.A2)


############################################################
## evaluate MSE wrt beta1 (first coefficient) in LM and Ridge regression
## loop over a given dataset of predictors (dat), for a given true beta value
evalMSEbeta1 = function(dat, betaj, mtest) {
  ndat = nrow(dat)
  l2beta = sqrt(sum(betaj^2))
  daty = data.frame(dat)
  ## generate a sample Y for a single dataset (to define initial cv.glmnet lambdas)
  daty$y = cbind(dat)%*%betaj + rnorm(ndat)
  ## for ridge, start with a vector of initial lambdas (+ adding small lambda values initially not considered)
  r_reg = cv.glmnet(dat,daty$y,alpha=0,nfolds=10,intercept=FALSE)

  lambda_CV = r_reg$glmnet.fit$lambda
  lambda_CV = c(lambda_CV, 0.001, rev(seq(0.000, 0.5, 0.002)^3))

  outLMRidge = sapply(1:mtest,function(i){
    daty = data.frame(dat)
    daty$y = cbind(dat)%*%betaj + rnorm(ndat)
    coef_lm = coef(lm(y~-1+.,daty))
    l2_LM = sqrt(sum(coef_lm^2))
    names(coef_lm) = c("b1_LM", "b2_LM")
    r_reg = cv.glmnet(dat,daty$y,alpha=0,nfolds=10,intercept=FALSE,lambda=lambda_CV)
    coef_r = coef(r_reg)[2:3,]
    ## extract beta ests that are closest to l2-norm = 10
    betas = r_reg$glmnet.fit$beta
    idx_l2norm10 = which.min(abs(sqrt(betas[1,]^2 + betas[2,]^2) - 10))
    coef_l2norm10 = betas[, idx_l2norm10]
    names(coef_l2norm10) = names(coef_r) = c("b1_Ridge", "b2_Ridge")
    l2_Ridge = sqrt(sum(coef_r^2))
    ## if needed, replace LM or Ridge betas with coef_l2norm10
    if (l2_LM > 10.0) {
      coef_lm[] = coef_l2norm10
    }
    if (l2_Ridge > 10.0) {
      coef_r[] = coef_l2norm10
    }
    return(c(coef_lm, coef_r))
  })

  ## test it works (all L2 norms must be <= 10.0)
  # l2_LM = sqrt(outLMRidge[1,]^2 + outLMRidge[2,]^2)
  # print(l2_LM)
  # l2_R = sqrt(outLMRidge[3,]^2 + outLMRidge[4,]^2)
  # print(l2_R)

  MSE_LM = mean((outLMRidge["b1_LM",]-betaj[1])^2)
  MSE_Ridge = mean((outLMRidge["b1_Ridge",]-betaj[1])^2)
  cat("betaj:", betaj, "; l2norm:", l2beta, "MSE_LM:", round(MSE_LM,3), "; MSE_Ridge:", round(MSE_Ridge,3), "\n")
  return(c(MSE_LM = MSE_LM, MSE_Ridge = MSE_Ridge))
}

############################################################
## Iterate over two beta samples (A1 / A2)
sample_and_eval = function(dat, datname, beta, betaname) {
  ## Sampling method A1
  res_bias <- foreach(j = 1:nbeta, .packages = c("parallel", "glmnet")) %dopar% {
    cat("sim betaj:", j, "\n")
    betaj = beta[,j,drop=FALSE]
    evalMSEbeta1(dat, betaj, mtest)
  }
  res_dat = do.call("rbind", res_bias)
  maxLMidx = which.max(res_dat[,1])
  maxRidgeidx = which.max(res_dat[,2])
  res_dat = cbind(t(beta), res_dat)

  ## save the risks per beta in a separate dataset
  saveRDS(data.frame(res_dat), file = paste0("./res_",betaname,"_",datname,".Rdata"))

  ## evaluate the Bayes risk (over beta draws) 
  BayesRisk = colMeans(res_dat)[3:4]
  names(BayesRisk) = c("BayesRisk_LM", "BayesRisk_Ridge")

  ## re-evaluate the worst risk wrt one beta (to de-bias the max risk estimate)
  betajworstLM = t(res_dat[maxLMidx, c(1,2), drop=FALSE])
  maxRiskLM = evalMSEbeta1(dat, betajworstLM, 5000)
  maxRiskLM = maxRiskLM[1]
  print("MAX LM risk before de-biasing"); print(res_dat[maxLMidx, 3])
  print("MAX LM risk after de-biasing"); print(maxRiskLM)

  betajworstRidge = t(res_dat[maxRidgeidx, c(1,2), drop=FALSE])
  maxRiskRidge = evalMSEbeta1(dat, betajworstRidge, 5000)
  maxRiskRidge = maxRiskRidge[2]
  print("MAX Risdge risk before de-biasing"); print(res_dat[maxRidgeidx, 4])
  print("MAX Risdge risk after de-biasing"); print(maxRiskRidge)

  ## Save all the risks in csv and return
  Risks = rbind(rbind(data.frame(BayesRisk), maxRisk_LM = maxRiskLM), maxRisk_Ridge = maxRiskRidge)
  write.csv(Risks, file = paste0("./res_",betaname,"_",datname,".csv"))

  print(paste0("RISK for ", datname, " ", betaname, " beta samples"))
  print(Risks)

  return(Risks)
}


#############################################################
## Sampling using mtcars, A1 beta sample
Risks_mtA1 = sample_and_eval(dat_mt, "mtcars", beta.A1, "A1")
#############################################################
## Sampling using mtcars, A2 beta sample
Risks_mtA2 = sample_and_eval(dat_mt, "mtcars", beta.A2, "A2")
############################################################
## Sampling using mandel, A1 beta sample
Risks_mandelA1 = sample_and_eval(dat_mandel, "mandel", beta.A1, "A1")
############################################################
## Sampling using mandel, A1 beta sample
Risks_mandelA2 = sample_and_eval(dat_mandel, "mandel", beta.A2, "A2")
############################################################
## Final risks
print("RISK for mtcars, A1 beta samples")
print(Risks_mtA1)
print("RISK for mtcars, A2 beta samples")
print(Risks_mtA2)
print("RISK for mandel, A1 beta samples")
print(Risks_mandelA1)
print("RISK for mandel, A2 beta samples")
print(Risks_mandelA2)
