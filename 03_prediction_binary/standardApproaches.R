library(nloptr)
library(glmnet)


makeDatasets = function(num.reps,p,n){
	# number of MC repetitions for evaluation
	m = 1000
	lapply(1:num.reps,function(i){
		x = matrix(rnorm(n*p),ncol=p)
		u = runif(n)
		xp = matrix(rnorm(m*p),ncol=p)
		return(list(x=x,u=u,xp=xp))
	})
}

makeCD4Datasets = function(num.reps,n){
	# number of MC repetitions for evaluation
	m = 1000
	lapply(1:num.reps,function(i){
		x = matrix(rnorm(n*2),ncol=2)
		x[,1] = 2*(x[,1]>0)-1
		u = runif(n)
		xp = matrix(rnorm(m*2),ncol=2)
		xp[,1] = 2*(xp[,1]>0)-1
		return(list(x=x,u=u,xp=xp))
	})
}

# Correlated predictors
my_sqrtm = function(A){
  e <- eigen(A)
  V <- e$vectors
  V %*% diag(sqrt(e$values)) %*% t(V)
}

makeDatasets_rho = function(num.reps,p,n,rho){
  # number of MC repetitions for evaluation
  m = 1000
  Sigma = matrix(rho,p,p)
  diag(Sigma) = 1
  SigmaRoot = my_sqrtm(Sigma)
  lapply(1:num.reps,function(i){
    x = matrix(rnorm(n*p),ncol=p)%*%SigmaRoot
    u = runif(n)
    xp = matrix(rnorm(m*p),ncol=p)%*%SigmaRoot
    return(list(x=x,u=u,xp=xp))
  })
}


# Gaussian covariates, p=2
makeDatasets2 = function(num.reps,n){makeDatasets(num.reps,2,n)}
# Gaussian covariates, p=10
makeDatasets10 = function(num.reps,n){makeDatasets(num.reps,10,n)}

# Gaussian covariates, p=10, rho = 0.3
makeDatasets10_rho3 = function(num.reps,n){makeDatasets_rho(num.reps,10,n,0.3)}

# Gaussian covariates, p=10, rho = 0.6
makeDatasets10_rho6 = function(num.reps,n){makeDatasets_rho(num.reps,10,n,0.6)}

# Gaussian covariates, p=10, rho = 0.9
makeDatasets10_rho9 = function(num.reps,n){makeDatasets_rho(num.reps,10,n,0.9)}


# truebeta is the true parameter value
# estimator maps from an observed (x,y) pair and a holdout set of covariates xp to a prediction for yp
# ds is the list of data sets and holdout sets to be used
findRisk = function(beta0,estimator,Qbar,makeDS,num.ds,n,lb.est,ub.est,par=TRUE){
	f = function(i){
		ds = makeDS(1,n)
		preds = estimator(
			ds[[1]]$x,											# x
			as.numeric(ds[[1]]$u<Qbar(ds[[1]]$x,beta0)),		# y
			ds[[1]]$xp,											# xp
			lb.est,
			ub.est)
		Qb = Qbar(ds[[1]]$xp,beta0)
		out2 = -mean(Qb * log(preds/Qb) + (1-Qb) * log((1-preds)/(1-Qb)))
		return(out2)
	}
	if(par){
		exportDoMPI(cl,varlist=ls(),envir=environment())
		out = unlist(foreach(i=1:num.ds, .packages=c('nloptr'), .verbose=FALSE, .errorhandling='pass') %dopar% {f(i)})
	} else {
		require(nloptr)
		out = sapply(1:num.ds,f)
	}
	return(list(risk=-mean(out),riskVar=var(out)))
}

glm.est = function(x,y,xp,...){
	pmax(pmin(predict(glm(y~.,data=data.frame(x),family=binomial),newdata=data.frame(xp),type="response"),0.9),0.1)
}

glm.interaction.est = function(x,y,xp,...){
	pmax(pmin(predict(glm(y~.^2,data=data.frame(x),family=binomial),newdata=data.frame(xp),type="response"),0.9),0.1)
}

glmnet.est = function(x,y,xp,...){
	require(glmnet)
	return(
		tryCatch({ # errors occur when very very few 0 or 1 events -- in these rare cases, revert to glm
			cvob1 = cv.glmnet(x,y,family="binomial")
			return(pmax(pmin(predict(cvob1,newx=xp,s="lambda.min",type="response"),0.9),0.1))
		}, error = function(e){
			pmax(pmin(predict(glm(y~.,data=data.frame(x),family=binomial),newdata=data.frame(xp),type="response"),0.9),0.1)
		}))
}

erm.est = function(Qbar,x,y,xp,lb,ub,n.start=5){#,param="lin"){
	f = function(cc){
		preds = Qbar(x,cc)
		-mean(y*log(preds) + (1-y)*log(1-preds))
	}
	# if(param=="lin"){
	# 	grad_f = function(cc){
	# 		b = c(cbind(x,1)%*%cbind(cc))
	# 		(rbind(plogis(b)*(exp(b)*(y-1 + y - 0.1))/(0.1+exp(b))) %*% cbind(x,1))/nrow(x)
	# 	}
	# } else if(param=="nn"){
	# 	# To be filled in
	# }
	init.grid.size = max(100,2*n.start)
	cc.init.mat = matrix(runif(length(lb)*init.grid.size,rep(lb,each=init.grid.size),rep(ub,each=init.grid.size)),nrow=init.grid.size)
	risks = apply(cc.init.mat,1,f)
	opt.out = apply(cc.init.mat[order(risks)[1:n.start],,drop=FALSE],1,function(cc.init){
		#opt = optim(cc.init,f,method="L-BFGS-B",lower=lb,upper=ub)
		#return(c(opt$value,opt$par))
		opt = nloptr(
			x0=cc.init,
			eval_f=f,
			lb=lb,
			ub=ub,
			# opts=list("algorithm"="NLOPT_LD_LBFGS"),
			# eval_grad_f=grad_f)
			# opts=list("algorithm"="NLOPT_LN_COBYLA"))
			opts=list("algorithm"="NLOPT_LN_BOBYQA","xtol_rel" = 1.0e-5))
		return(c(opt$objective,opt$solution))
	})
	cc.final = opt.out[-1,which.min(opt.out[1,])]
	Qbar(xp,cc.final)
}

erm.lin.est = function(x,y,xp,lb,ub){erm.est(Qbar.lin,x,y,xp,lb,ub,n.start=1)}
erm.nn.est = function(x,y,xp,lb,ub){erm.est(Qbar.nn,x,y,xp,lb,ub,n.start=1)}

erm.nn2.est = function(x,y,xp,lb,ub){erm.est(Qbar.nn2,x,y,xp,lb,ub,n.start=1)}


gen.rand = function(n,lb,ub){
	matrix(rep(lb,n),ncol=n) + matrix(runif(n*length(lb)),ncol=n)*matrix(rep(ub-lb,n),ncol=n)
}

gen.rand.border = function(n,lb,ub){
	matrix(rep(lb,n),ncol=n) + matrix(runif(n*length(lb))>1/2,ncol=n)*matrix(rep(ub-lb,n),ncol=n)
}

## findsHardRisk
# Inputs
# estimator : a function mapping from a training data set to predictions on a validation data set
# Qbar : a parameterization for the conditional expectation
# lb.model : model lower bounds for parameters of the conditional expectation
# ub.model : model upper bounds for parameters of the conditional expectation
# makeDS : a function that can generate the desired number of data sets
# lb.model : assumed lower bounds (by the estimator) for parameters of the conditional expectation
# ub.model : assumed upper bounds (by the estimator) for parameters of the conditional expectation
# numDS : Number of data sets used to compute Bayes risk
findHardRisk = function(estimator,Qbar,lb.model,ub.model,makeDS,lb.est=lb.model,ub.est=ub.model,numBayesRep=5e3,interrogationStarts=10,numInterDS=10,n=50,numRectShrink=8,SGAnstartsBorder=1,SGAnstarts=1,ntestFun=function(j){2500+50*(j+1)},mFinal=5000){	

	unifMat = gen.rand(numBayesRep,lb.model,ub.model)
	exportDoMPI(cl,varlist=ls(),envir=environment())
	unifBayesAll = foreach(i=1:ncol(unifMat), .packages=c('nloptr'), .verbose=FALSE, .errorhandling='pass') %dopar% {
		findRisk(unifMat[,i],estimator,Qbar,makeDS,1,n,lb.est,ub.est,par=FALSE)$risk}
	unifBayesRisk = mean(-unlist(unifBayesAll))
	print(paste0("Uniform Bayes risk: ",unifBayesRisk))
	maxRisk = -Inf
	hardestGamma = NA

	for(i in 1:interrogationStarts){
	    SGAnstarts_total = SGAnstartsBorder+SGAnstarts
	    currHardestGammaArray = gen.rand(SGAnstarts,lb.model,ub.model)
	    # print(currHardestGammaArray)
	    # print(paste0("initCurrMaxRisk (biased up): ",max(risks)))
	    for(j in 0:numRectShrink){		# NOTE: 1:8 was used to interrogate our estimator, here 1:7 is used (should yield a slightly worse optimizer, but will also run slightly faster...)
	    	print(paste0("Start ",i,", Step ",j))
	    	rectMaxWidths = (ub.model-lb.model)*(0.95^j)/2
	    	rectLBs = pmax(lb.model,currHardestGammaArray-rectMaxWidths)
	    	rectUBs = pmin(ub.model,currHardestGammaArray+rectMaxWidths)
	    	newInits = cbind(gen.rand(SGAnstarts,rectLBs,rectUBs),gen.rand.border(SGAnstartsBorder,rectLBs,rectUBs),currHardestGammaArray)
	    	risks = apply(newInits,2,function(gamma){-findRisk(gamma,estimator,Qbar,makeDS,ntestFun(j),n,lb.est,ub.est)$risk})
	    	currHardestGammaArray = newInits[,which.max(risks)]
	    	# print(currHardestGammaArray)
	    	print(paste0("currMaxRisk (biased up): ",max(risks)))
	    }
	    # print(currHardestGammaArray)
	    currMaxRisk = -findRisk(currHardestGammaArray,estimator,Qbar,makeDS,mFinal*numInterDS,n,lb.est,ub.est)$risk
	    print(paste0("currMaxRisk: ",currMaxRisk))
	    if(currMaxRisk>maxRisk){
	    	maxRisk = currMaxRisk
	    	hardestGamma = currHardestGammaArray
	    }
	}
	# The current max risk estimate will be biased up, so check risk again to get unbiased estimate of risk at current gamma
	numEvalReps = mFinal*2*numInterDS
	tmp = findRisk(hardestGamma,estimator,Qbar,makeDS,numEvalReps,n,lb.est,ub.est)
	maxRisk = -tmp$risk
	se = sqrt(tmp$riskVar/numEvalReps)
	print(paste0("maxRisk: ",maxRisk))

	return(list(maxRisk=maxRisk,hardestGamma=hardestGamma,unifBayesRisk=unifBayesRisk,se=se))#,borderBayesRisk=borderBayesRisk))
}

# Linear-logistic (scaled to [0.1,0.9])
Qbar.lin = function(x,beta0) 0.1 + 0.8*plogis(cbind(x,1)%*% cbind(beta0))

# NN with 1 hidden layer
Qbar.nn = function(x,beta0) {
	0.1 + 0.8 * plogis(cbind(tanh(cbind(x,1)%*%matrix(beta0[1:9],nrow=3,ncol=3)),1)%*%cbind(beta0[10:13]))
}

# NN with 2 hidden layers
Qbar.nn2 = function(x,beta0) {
	0.1 + 0.8 * plogis(cbind(tanh(cbind(tanh(cbind(x,1)%*%matrix(beta0[1:9],nrow=3,ncol=3)),1)%*%matrix(beta0[10:21],nrow=4,ncol=3)),1)%*%cbind(beta0[22:25]))
}

