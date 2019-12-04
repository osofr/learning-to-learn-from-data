# functions used in data analyses

# make validation indices
make_val_inds = function(wd,ds_name,response_var,num_vals=1000){
	val_inds = do.call(rbind,lapply(1:num_vals,function(i){
		cont = TRUE
		out = NA
		while(cont){
			out = sample(1:nrow(processed_data),nrow(processed_data)-50)
			cont = (mean(processed_data[[response_var]][out])%in%c(0,1))
		}
		return(out)
	}))

	write.csv(val_inds,file=file.path(wd,paste0("processed_",ds_name,"_data"),"val_inds.csv"),quote=FALSE,row.names=FALSE)

	return(TRUE)
}

# evaluate performance of the different methods
eval_performance = function(wd,ds_name,response_var,methods,finite_pop=FALSE){
	processed_data = read.csv(file.path(wd,paste0("processed_",ds_name,"_data"),"processed_data.csv"))
	val_inds = read.csv(file.path(wd,paste0("processed_",ds_name,"_data"),"val_inds.csv"))

	val_inds_list = lapply(1:nrow(val_inds),function(i){as.numeric(val_inds[i,])})

	########################################

	truth = lapply(1:nrow(val_inds),function(i){processed_data[[response_var]][as.numeric(val_inds[i,])]})

	########################################
	# cross-entropy

	cross_entropy = function(preds,truth,val_inds_list,finite_pop){
		if(!finite_pop){
			n = length(unique(unlist(val_inds_list)))
			indiv_loss = aggregate(
				data.frame(loss=unlist(lapply(1:length(truth),function(i){
					curr_preds = preds[[i]]
					(-(truth[[i]]*log(curr_preds) + (1-truth[[i]])*log(1-curr_preds))) * (n/(length(val_inds_list)*length(val_inds_list[[i]])))
					}))),
				list(unlist(val_inds_list)),
				sum)$loss

			est = mean(indiv_loss)
			# accounts for randomness in selection of full sample (selected from infinite population)
			# doesn't account for randomness in selected CV folds
			se = sd(indiv_loss)/sqrt(length(indiv_loss))
		} else {
			risk_fold = sapply(1:length(truth),function(i){
				curr_preds = preds[[i]]
				-mean(truth[[i]]*log(curr_preds) + (1-truth[[i]])*log(1-curr_preds))
				})
			est = mean(risk_fold)
			se = sd(risk_fold)/sqrt(length(risk_fold))
		}

		return(c(est,est-qnorm(0.975)*se,est+qnorm(0.975)*se))
	}

	########################################
	# auc and ci

	my_ci.cvAUC = function(preds,truth,val_inds_list,finite_pop){
		require("ROCR")
		require("cvAUC")
		n = max(unlist(val_inds_list))
		auc_folds = sapply(1:length(preds),function(i){
			performance(prediction(preds[[i]],truth[[i]]),"auc")@y.values[[1]]
			})
		auc_out = mean(auc_folds)
		
		if(!finite_pop){
			# accounts for randomness in selection of full sample (selected from infinite population)
			# doesn't account for randomness in selected CV folds
			se = ci.cvAUC(preds,truth)$se * sqrt(length(unlist(preds)))/sqrt(n)
		} else {
			# doesn't account for randomness in selection of full sample (finite population)
			# does account for randomness in selected CV folds (random draws from this finite population)
			se = sd(auc_folds)/sqrt(length(auc_folds))
		}


		return(c(auc_out,auc_out-qnorm(0.975)*se,auc_out+qnorm(0.975)*se))
	}

	########################################
	# Existing methods

	glm_preds = lapply(val_inds_list,function(val_ind){
		pmax(pmin(predict(
							glm(
								paste0(response_var," ~ ."),
								processed_data[-val_ind,],
								family = binomial()
								),
							newdata = processed_data[val_ind,],
							type="response"),0.999),0.001)
		})
	glm_auc = my_ci.cvAUC(glm_preds,truth,val_inds_list,finite_pop)
	glm_crossent = cross_entropy(glm_preds,truth,val_inds_list,finite_pop)

	library("glmnet")
	glmnet_preds = lapply(val_inds_list,function(val_ind){
		cvob1 = tryCatch(cv.glmnet(as.matrix(processed_data[-val_ind,][,names(processed_data)!=response_var]),processed_data[-val_ind,][[response_var]],family="binomial"),
			error=function(e){ # if there's an error, running again with user-specified sequence of lambdas should fix it
			# https://stackoverflow.com/questions/40145209/cv-glmnet-fails-for-ridge-not-lasso-for-simulated-data-with-coder-error
				warning("glmnet gave error on first try, so provided user-specified sequence of lambdas")
				cv.glmnet(as.matrix(processed_data[-val_ind,][,names(processed_data)!=response_var]),processed_data[-val_ind,][[response_var]],family="binomial",lambda=exp(seq(log(0.0001), log(20), length.out=100)))
			})
		pmin(pmax(predict(cvob1,newx=as.matrix(processed_data[val_ind,][,names(processed_data)!=response_var]),s="lambda.min",type="response")[,1],0.001),0.999)
		})
	glmnet_auc = my_ci.cvAUC(glmnet_preds,truth,val_inds_list,finite_pop)
	glmnet_crossent = cross_entropy(glmnet_preds,truth,val_inds_list,finite_pop)

	# ERMs implementing the same methods used above
	# Relies on functions in 

	# warning("If update standardApproaches.R, check if need to update this RCurl url.")
	# library(RCurl)

	# script <- getURL("https://raw.githubusercontent.com/osofr/funcGrad.jl/master/examples/ex2/standardApproaches.R?token=AdarRIcm0VBNmScALEoPFaYzoTaJmRfyks5bkWa_wA%3D%3D", ssl.verifypeer = FALSE)

	# eval(parse(text = script))

	source("standardApproaches.R")

	erm = lapply(methods,function(method){
		# define settings used when fitting our adversarial algorithms
		# (to be provided to the ERMs)
		if(method=="cd4_linear_logistic_2d_n50_pm05_borders"){
			lb = c(-0.5,-0.5,-1.0)
			ub = -lb
			Qbar_curr = Qbar.lin
		} else if(method=="cd4_linear_logistic_2d_n50_pm1_borders") {
			lb = c(-1.0,-1.0,-1.0)
			ub = -lb
			Qbar_curr = Qbar.lin
		} else if(method=="cd4_linear_logistic_2d_n50_pm2_borders") {
			lb = c(-2.0,-2.0,-1.0)
			ub = -lb
			Qbar_curr = Qbar.lin
		} else if(method=="linear_logistic_2d_n50_borders"){
			lb = rep(-2,3)
			ub = -lb
			Qbar_curr = Qbar.lin
		} else if(method%in%c("NN3_2d_n50_mixed","NN3_2d_n50_mixed_glmInit")) {
			lb = rep(-2,13)
			ub = -lb
			Qbar_curr = Qbar.nn
		} else if(method%in%c("NN33_2d_n50_mixed","NN33_2d_n50_mixed_glmInit")) {
			lb = rep(-2,25)
			ub = -lb
			Qbar_curr = Qbar.nn2
		} else if(method=="linear_logistic_10d_n50_nointercept_borders"){
			lb = c(rep(-0.5,10),0.0)
			ub = -lb
			Qbar_curr = Qbar.lin
		} else if(method=="linear_logistic_10d_n50_pm1_borders") {
			lb = c(rep(-0.5,10),-1.0)
			ub = -lb
			Qbar_curr = Qbar.lin
		} else if(method=="linear_logistic_10d_n50_borders") {
			lb = c(rep(-0.5,10),-2.0)
			ub = -lb
			Qbar_curr = Qbar.lin
		}
		erm_preds = lapply(val_inds_list,function(val_ind){

			curr_dat = processed_data
			# process data so that columns are standardized based on training data
			# for cd4 analyses, recode binary var to be in +/- 1
			if(method%in%c("cd4_linear_logistic_2d_n50_pm05_borders","cd4_linear_logistic_2d_n50_pm1_borders","cd4_linear_logistic_2d_n50_pm2_borders")){
				mu_bmi = mean(curr_dat$bmi[-val_ind])
				std_bmi = sd(curr_dat$bmi[-val_ind])
				curr_dat$bmi = (curr_dat$bmi-mu_bmi)/std_bmi
				curr_dat$sexbin = 2*curr_dat$sexbin - 1
			} else {
				mu = colMeans(curr_dat[-val_ind,names(curr_dat)!=response_var])
				std = apply(curr_dat[-val_ind,names(curr_dat)!=response_var],2,sd)
				for(i in which(names(curr_dat)!=response_var)){
					curr_dat[,i] = (curr_dat[,i]-mu[i])/std[i]
				}
			}

			erm.est(
				Qbar_curr,
				as.matrix(curr_dat[-val_ind,names(curr_dat)!=response_var]),
				curr_dat[-val_ind,][[response_var]],
				as.matrix(curr_dat[val_ind,names(curr_dat)!=response_var]),
				lb=lb,
				ub=ub,
				n.start=5)
		})
		erm_auc = my_ci.cvAUC(erm_preds,truth,val_inds_list,finite_pop)
		erm_crossent = cross_entropy(erm_preds,truth,val_inds_list,finite_pop)
		return(list(auc=erm_auc,crossent=erm_crossent))
		})

	existing_aucs = do.call(rbind,lapply(erm,function(xx){xx$auc}))
	existing_crossents = do.call(rbind,lapply(erm,function(xx){xx$crossent}))

	rownames(existing_aucs) <- rownames(existing_crossents) <- paste0("erm_",methods)

	existing_aucs = rbind(existing_aucs,"glm"=glm_auc,"glmnet"=glmnet_auc)
	existing_crossents = rbind(existing_crossents,"glm"=glm_crossent,"glmnet"=glmnet_crossent)

	########################################
	# Adversarially learned methods

	new_aucs = do.call(rbind,lapply(methods,function(meth){
		preds = read.csv(file.path(wd,paste0(ds_name,"_preds"),paste0(meth,".csv")))
		preds = lapply(1:nrow(preds),function(i){as.numeric(preds[i,])})
		my_ci.cvAUC(preds,truth,val_inds_list,finite_pop)
		}))

	new_crossents = do.call(rbind,lapply(methods,function(meth){
		preds = read.csv(file.path(wd,paste0(ds_name,"_preds"),paste0(meth,".csv")))
		preds = lapply(1:nrow(preds),function(i){as.numeric(preds[i,])})
		cross_entropy(preds,truth,val_inds_list,finite_pop)
		}))

	rownames(new_aucs) <- rownames(new_crossents) <- methods

	########################################
	# Combining it all

	all_aucs = rbind(new_aucs,existing_aucs)
	all_crossents = rbind(new_crossents,existing_crossents)

	colnames(all_aucs) = c("auc","lb","ub")
	colnames(all_crossents) = c("crossent","lb","ub")

	write.csv(all_aucs,file.path(wd,paste0(ds_name,"_preds"),"auc.csv"))
	write.csv(all_crossents,file.path(wd,paste0(ds_name,"_preds"),"crossent.csv"))

	return(list(auc=all_aucs,crossent=all_crossents))
}
