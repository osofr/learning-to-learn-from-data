
num.ds = 1e6

source("standardApproaches.R")

# Hardest gamma for NN3, from the file NN3_2d_n50_mixed/risks/3930_maxRisk_250starts.jld2
beta_NN3 = c(
	c(
		rbind(
			cbind(c(-0.327238, -1.81789), c(-1.84999, 0.136529), c(1.98951, 0.612417)),
			c(-1.58926, 1.24234, 1.00674))),
	c(
		rbind(
			cbind(c(1.96761, 2.0, 1.98274)),
			c(-0.024825)))
)

# find the risk at this gamma
NN3 = findRisk(
	beta=beta_NN3,
	estimator=erm.nn.est,
	Qbar=Qbar.nn,
	makeDS=makeDatasets2,
	num.ds=num.ds,
	n=50,
	lb.est=rep(-2,13),
	ub.est=rep(2,13),
	par=FALSE)

# write to file
write.table(-NN3$risk,file="NN3_2d_n50_mixed/standard_approaches/erm_minimax_least_favorable_maxRisk.txt",quote=FALSE,row.names=FALSE,col.names=FALSE)
write.table(sqrt(NN3$riskVar/num.ds),file="NN3_2d_n50_mixed/standard_approaches/erm_minimax_least_favorable_se.txt",quote=FALSE,row.names=FALSE,col.names=FALSE)
write.table(beta_NN3,file="NN3_2d_n50_mixed/standard_approaches/erm_minimax_least_favorable_hardestGamma.txt",quote=FALSE,row.names=FALSE,col.names=FALSE)



# Hardest gamma for NN33, from the file NN33_2d_n50_mixed/risks/3280_maxRisk_250starts.jld2
beta_NN33 = c(
	c(
		rbind(
			cbind(c(0.368417, 1.98696), c(-0.951603, -1.833), c(-0.554679, -1.07589)),
			c(-1.89501, 0.0535888, -1.44501))),
	c(
		rbind(
			cbind(c(1.80488, 1.91069, -1.91917), c(-1.88147, -0.430678, -0.377298), c(1.74349, 1.75239, -1.66754)),
			c(-0.452579, -1.44129, -0.537124))),
	c(
		rbind(
			cbind(c(-1.83732,1.11424,-1.90643)),
			c(-1.20174)))
)

# find the risk at this gamma
NN33 = findRisk(
	beta=beta_NN33,
	estimator=erm.nn2.est,
	Qbar=Qbar.nn2,
	makeDS=makeDatasets2,
	num.ds=num.ds,
	n=50,
	lb.est=rep(-2,25),
	ub.est=rep(2,25),
	par=FALSE)

# write to file
write.table(-NN33$risk,file="NN33_2d_n50_mixed/standard_approaches/erm_minimax_least_favorable_maxRisk.txt",quote=FALSE,row.names=FALSE,col.names=FALSE)
write.table(sqrt(NN33$riskVar/num.ds),file="NN33_2d_n50_mixed/standard_approaches/erm_minimax_least_favorable_se.txt",quote=FALSE,row.names=FALSE,col.names=FALSE)
write.table(beta_NN33,file="NN33_2d_n50_mixed/standard_approaches/erm_minimax_least_favorable_hardestGamma.txt",quote=FALSE,row.names=FALSE,col.names=FALSE)


