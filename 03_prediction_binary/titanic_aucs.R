
# calculate aucs for all methods

set.seed(2)

library(AUC)
library(cvAUC)

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

wd = "./03_prediction_binary"
ds_name = "titanic"
response_var = "survived"
methods = c(
	"linear_logistic_10d_n50_nointercept_borders"="linear_logistic_10d_n50_nointercept_borders",
	"linear_logistic_10d_n50_pm1_borders"="linear_logistic_10d_n50_pm1_borders",
	"linear_logistic_10d_n50_borders"="linear_logistic_10d_n50_borders")
finite_pop = TRUE


####################################################################################

source(file.path(wd,"data_analysis_funs.R"))

####################################################################################

out = eval_performance(wd,ds_name,response_var,methods,finite_pop=finite_pop)

out$auc[order(out$auc[,"auc"],decreasing=TRUE),]

out$crossent[order(out$crossent[,"crossent"]),]
