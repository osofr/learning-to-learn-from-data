## Read in args from job submission
args=(commandArgs(trailingOnly=TRUE))

if(length(args)==0){
	warning("No arguments supplied.")
} else {
	for(i in 1:length(args)) eval(parse(text=args[[i]]))
}

source("standardApproaches.R")
library(doMPI)

set.seed(54321)

cl <- startMPIcluster()
registerDoMPI(cl)
# Also pass Qbar's/estimators (beginning in "Qbar" and ending in ".est")
exportDoMPI(cl,varlist=ls()[(substr(ls(),0,4)=="Qbar") | (substr(ls(),nchar(ls())-4+1,nchar(ls()))==".est")])
exportDoMPI(cl,varlist=c("findRisk","makeDatasets","makeCD4Datasets","makeDatasets_rho","my_sqrtm"))


##########################################################################################
##########################################################################################
# linear logistic, p=2
##########################################################################################

settings = list(
# NN, p=2
	list(
		est = erm.nn.est,
		Qbar = Qbar.nn,
		lb.model = rep(-2,13),
		ub.model = rep(2,13),
		lb.est = rep(-2,13),
		ub.est = rep(2,13),
		makeDS = makeDatasets2,
		wd = "NN3_2d_n50_mixed",
		prefix="erm"
	),
# linear logistic, p=2
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = rep(-2,3),
		ub.model = rep(2,3),
		lb.est = rep(-2,3),
		ub.est = rep(2,3),
		makeDS = makeDatasets2,
		wd = "linear_logistic_2d_n50_borders",
		prefix="glm"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = rep(-2,3),
		ub.model = rep(2,3),
		lb.est = rep(-2,3),
		ub.est = rep(2,3),
		makeDS = makeDatasets2,
		wd = "linear_logistic_2d_n50_borders",
		prefix="erm"
	),
# linear logistic, p=10
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),2),
		ub.model = c(rep(0.5,10),2),
		lb.est = -c(rep(0.5,10),2),
		ub.est = c(rep(0.5,10),2),
		makeDS = makeDatasets10,
		wd = "linear_logistic_10d_n50_borders",
		prefix="glm"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),2),
		ub.model = c(rep(0.5,10),2),
		lb.est = -c(rep(0.5,10),2),
		ub.est = c(rep(0.5,10),2),
		makeDS = makeDatasets10,
		wd = "linear_logistic_10d_n50_borders",
		prefix="erm"
	),
	list(
		est = glmnet.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),2),
		ub.model = c(rep(0.5,10),2),
		lb.est = -c(rep(0.5,10),2),
		ub.est = c(rep(0.5,10),2),
		makeDS = makeDatasets10,
		wd = "linear_logistic_10d_n50_borders",
		prefix="glmnet"
	),
# # cd4 NN, p=2
# 	list(
# 		est = erm.nn.est,
# 		Qbar = Qbar.nn,
# 		lb.model = -c(rep(2,12),1),
# 		ub.model = c(rep(2,12),1),
# 		lb.est = -c(rep(2,12),1),
# 		ub.est = c(rep(2,12),1),
# 		makeDS = makeCD4Datasets,
# 		wd = "cd4_NN3_2d_n50_borders",
# 		prefix="erm"
# 	),
# cd4 linear logistic, p=2, estimator assumes bounds of +/-[0.5,0.5,1]
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(0.5,0.5,1),
		ub.model = c(0.5,0.5,1),
		lb.est = -c(0.5,0.5,1),
		ub.est = c(0.5,0.5,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm05_borders",
		prefix="glm"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(0.5,0.5,1),
		ub.model = c(0.5,0.5,1),
		lb.est = -c(0.5,0.5,1),
		ub.est = c(0.5,0.5,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm05_borders",
		prefix="erm"
	),
# cd4 linear logistic, p=2, estimator assumes bounds of +/-[1,1,1]
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(1,1,1),
		ub.model = c(1,1,1),
		lb.est = -c(1,1,1),
		ub.est = c(1,1,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm1_borders",
		prefix="glm"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(1,1,1),
		ub.model = c(1,1,1),
		lb.est = -c(1,1,1),
		ub.est = c(1,1,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm1_borders",
		prefix="erm"
	),
# cd4 linear logistic, p=2, estimator assumes bounds of +/-[2,2,1]
	# Truth has bounds +/- [2,2,1]
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(2,2,1),
		ub.model = c(2,2,1),
		lb.est = -c(2,2,1),
		ub.est = c(2,2,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm2_borders",
		prefix="glm"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(2,2,1),
		ub.model = c(2,2,1),
		lb.est = -c(2,2,1),
		ub.est = c(2,2,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm2_borders",
		prefix="erm"
	),
# cd4 linear logistic, p=2, estimator assumes bounds of +/-[0.5,0.5,1]
	# Truth has bounds +/- [1,1,1]
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(1,1,1),
		ub.model = c(1,1,1),
		lb.est = -c(0.5,0.5,1),
		ub.est = c(0.5,0.5,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm05_borders",
		prefix="glm_1"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(1,1,1),
		ub.model = c(1,1,1),
		lb.est = -c(0.5,0.5,1),
		ub.est = c(0.5,0.5,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm05_borders",
		prefix="erm_1"
	),
	# Truth has bounds +/- [2,2,1]
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(2,2,1),
		ub.model = c(2,2,1),
		lb.est = -c(0.5,0.5,1),
		ub.est = c(0.5,0.5,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm05_borders",
		prefix="glm_2"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(2,2,1),
		ub.model = c(2,2,1),
		lb.est = -c(0.5,0.5,1),
		ub.est = c(0.5,0.5,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm05_borders",
		prefix="erm_2"
	),
# cd4 linear logistic, p=2, estimator assumes bounds of +/-[1,1,1]
	# Truth has bounds +/- [0.5,0.5,1]
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(0.5,0.5,1),
		ub.model = c(0.5,0.5,1),
		lb.est = -c(1,1,1),
		ub.est = c(1,1,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm1_borders",
		prefix="glm_05"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(0.5,0.5,1),
		ub.model = c(0.5,0.5,1),
		lb.est = -c(1,1,1),
		ub.est = c(1,1,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm1_borders",
		prefix="erm_05"
	),
	# Truth has bounds +/- [2,2,1]
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(2,2,1),
		ub.model = c(2,2,1),
		lb.est = -c(1,1,1),
		ub.est = c(1,1,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm1_borders",
		prefix="glm_2"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(2,2,1),
		ub.model = c(2,2,1),
		lb.est = -c(1,1,1),
		ub.est = c(1,1,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm1_borders",
		prefix="erm_2"
	),
# cd4 linear logistic, p=2, estimator assumes bounds of +/-[2,2,1]
	# Truth has bounds +/- [0.5,0.5,1]
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(0.5,0.5,1),
		ub.model = c(0.5,0.5,1),
		lb.est = -c(2,2,1),
		ub.est = c(2,2,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm2_borders",
		prefix="glm_05"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(0.5,0.5,1),
		ub.model = c(0.5,0.5,1),
		lb.est = -c(2,2,1),
		ub.est = c(2,2,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm2_borders",
		prefix="erm_05"
	),
	# Truth has bounds +/- [1,1,1]
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(1,1,1),
		ub.model = c(1,1,1),
		lb.est = -c(2,2,1),
		ub.est = c(2,2,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm2_borders",
		prefix="glm_1"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(1,1,1),
		ub.model = c(1,1,1),
		lb.est = -c(2,2,1),
		ub.est = c(2,2,1),
		makeDS = makeCD4Datasets,
		wd = "cd4_linear_logistic_2d_n50_pm2_borders",
		prefix="erm_1"
	),
	# NN (2 hidden layers), p=2
	list(
		est = erm.nn2.est,
		Qbar = Qbar.nn2,
		lb.model = rep(-2,25),
		ub.model = rep(2,25),
		lb.est = rep(-2,25),
		ub.est = rep(2,25),
		makeDS = makeDatasets2,
		wd = "NN33_2d_n50_mixed",
		prefix="erm"
	),
	# p=10, varying bounds on intercept
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),0),
		ub.model = c(rep(0.5,10),0),
		lb.est = -c(rep(0.5,10),0),
		ub.est = c(rep(0.5,10),0),
		makeDS = makeDatasets10,
		wd = "linear_logistic_10d_n50_nointercept_borders",
		prefix="erm"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),1),
		ub.model = c(rep(0.5,10),1),
		lb.est = -c(rep(0.5,10),1),
		ub.est = c(rep(0.5,10),1),
		makeDS = makeDatasets10,
		wd = "linear_logistic_10d_n50_pm1_borders",
		prefix="erm"
	),
	# p=10, varying bounds on intercept, non-MLE methods
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),0),
		ub.model = c(rep(0.5,10),0),
		lb.est = -c(rep(0.5,10),0),
		ub.est = c(rep(0.5,10),0),
		makeDS = makeDatasets10,
		wd = "linear_logistic_10d_n50_nointercept_borders",
		prefix="glm"
	),
	list(
		est = glmnet.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),0),
		ub.model = c(rep(0.5,10),0),
		lb.est = -c(rep(0.5,10),0),
		ub.est = c(rep(0.5,10),0),
		makeDS = makeDatasets10,
		wd = "linear_logistic_10d_n50_nointercept_borders",
		prefix="glmnet"
	),
	list(
		est = glm.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),1),
		ub.model = c(rep(0.5,10),1),
		lb.est = -c(rep(0.5,10),1),
		ub.est = c(rep(0.5,10),1),
		makeDS = makeDatasets10,
		wd = "linear_logistic_10d_n50_pm1_borders",
		prefix="glm"
	),
	list(
		est = glmnet.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),1),
		ub.model = c(rep(0.5,10),1),
		lb.est = -c(rep(0.5,10),1),
		ub.est = c(rep(0.5,10),1),
		makeDS = makeDatasets10,
		wd = "linear_logistic_10d_n50_pm1_borders",
		prefix="glmnet"
	),
	# p=10, rho=3
	list(
		est = glmnet.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),0),
		ub.model = c(rep(0.5,10),0),
		lb.est = -c(rep(0.5,10),0),
		ub.est = c(rep(0.5,10),0),
		makeDS = makeDatasets10_rho3,
		wd = "linear_logistic_10d_n50_nointercept_borders_rho3",
		prefix="glmnet"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),0),
		ub.model = c(rep(0.5,10),0),
		lb.est = -c(rep(0.5,10),0),
		ub.est = c(rep(0.5,10),0),
		makeDS = makeDatasets10_rho3,
		wd = "linear_logistic_10d_n50_nointercept_borders_rho3",
		prefix="erm"
	),
	# p=10, rho=6
	list(
		est = glmnet.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),0),
		ub.model = c(rep(0.5,10),0),
		lb.est = -c(rep(0.5,10),0),
		ub.est = c(rep(0.5,10),0),
		makeDS = makeDatasets10_rho6,
		wd = "linear_logistic_10d_n50_nointercept_borders_rho6",
		prefix="glmnet"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),0),
		ub.model = c(rep(0.5,10),0),
		lb.est = -c(rep(0.5,10),0),
		ub.est = c(rep(0.5,10),0),
		makeDS = makeDatasets10_rho6,
		wd = "linear_logistic_10d_n50_nointercept_borders_rho6",
		prefix="erm"
	),
	# p=10, rho=3
	list(
		est = glmnet.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),0),
		ub.model = c(rep(0.5,10),0),
		lb.est = -c(rep(0.5,10),0),
		ub.est = c(rep(0.5,10),0),
		makeDS = makeDatasets10_rho9,
		wd = "linear_logistic_10d_n50_nointercept_borders_rho9",
		prefix="glmnet"
	),
	list(
		est = erm.lin.est,
		Qbar = Qbar.lin,
		lb.model = -c(rep(0.5,10),0),
		ub.model = c(rep(0.5,10),0),
		lb.est = -c(rep(0.5,10),0),
		ub.est = c(rep(0.5,10),0),
		makeDS = makeDatasets10_rho9,
		wd = "linear_logistic_10d_n50_nointercept_borders_rho9",
		prefix="erm"
	)
)

# Run the simulation across the settings
for(i in settingInd){
	setting = settings[[i]]
	print("=======================================")
	print(setting$wd)
	print(setting$prefix)

	# to get results -- way fewer interrogations than we use on our own estimator
	out = with(setting,findHardRisk(est,Qbar,lb.model,ub.model,makeDS,lb.est=lb.est,ub.est=ub.est,numBayesRep=5000,numInterDS=2,interrogationStarts=3,n=50,numRectShrink=100,SGAnstartsBorder=1,SGAnstarts=1,ntestFun=function(j){2500+50*(j+1)},mFinal=5000))

	# for testing
	# out = with(setting,findHardRisk(est,Qbar,lb.model,ub.model,makeDS,lb.est=lb.est,ub.est=ub.est,numBayesRep=7,numInterDS=2,interrogationStarts=1,n=50,numRectShrink=1,SGAnstartsBorder=1,SGAnstarts=1,ntestFun=function(j){5+1*(j+1)},mFinal=5))

	save.dir = file.path(setting$wd, "standard_approaches")
	dir.create(save.dir, showWarnings = FALSE)
	write.table(out$maxRisk,file=file.path(save.dir,paste0(setting$prefix,"_maxRisk.txt")),quote=FALSE,row.names=FALSE,col.names=FALSE)
	write.table(out$hardestGamma,file=file.path(save.dir,paste0(setting$prefix,"_hardestGamma.txt")),quote=FALSE,row.names=FALSE,col.names=FALSE)
	write.table(out$unifBayesRisk,file=file.path(save.dir,paste0(setting$prefix,"_unifBayesRisk.txt")),quote=FALSE,row.names=FALSE,col.names=FALSE)
	write.table(out$se,file=file.path(save.dir,paste0(setting$prefix,"_se.txt")),quote=FALSE,row.names=FALSE,col.names=FALSE)
}

closeCluster(cl)
mpi.finalize()

q(save='no')
