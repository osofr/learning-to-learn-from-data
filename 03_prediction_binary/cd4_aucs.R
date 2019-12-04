
# calculate aucs for all methods

set.seed(2)

library(AUC)
library(cvAUC)

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

wd = "./03_prediction_binary"
ds_name = "cd4"
response_var = "any_response"
methods = c(
	"cd4_linear_logistic_2d_n50_pm05_borders"="cd4_linear_logistic_2d_n50_pm05_borders",
	"cd4_linear_logistic_2d_n50_pm1_borders"="cd4_linear_logistic_2d_n50_pm1_borders",
	"cd4_linear_logistic_2d_n50_pm2_borders"="cd4_linear_logistic_2d_n50_pm2_borders")
finite_pop = FALSE


####################################################################################

source(file.path(wd,"data_analysis_funs.R"))

####################################################################################

out = eval_performance(wd,ds_name,response_var,methods,finite_pop=finite_pop)

out$auc[order(out$auc[,"auc"],decreasing=TRUE),]

out$crossent[order(out$crossent[,"crossent"]),]
