# Runs the analysis of titanic data in processed_titanic_data folder.

cd("./03_prediction_binary")

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

# models to use
fldrs = [
	"linear_logistic_2d_n50_borders",
	"NN3_2d_n50_mixed",
	"NN33_2d_n50_mixed",
	"NN3_2d_n50_mixed_glmInit",
	"NN33_2d_n50_mixed_glmInit"]

# name of data set
# Note: the code treats the "cd4" dataset specially (because sexbin is binary and should be coded +/- 1)
ds_name = "titanic2d"

# name of response variable
response_var = "survived"

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

include("run_data_analysis.jl")


