# Runs the analysis of titanic data in processed_titanic_data folder.

cd("./03_prediction_binary")

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

# models to use
fldrs = [
	"linear_logistic_10d_n50_nointercept_borders",
	"linear_logistic_10d_n50_pm1_borders",
	"linear_logistic_10d_n50_borders"]

# name of data set
# Note: the code treats the "cd4" dataset specially (because sexbin is binary and should be coded +/- 1)
ds_name = "titanic"

# name of response variable
response_var = "survived"

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

include("run_data_analysis.jl")


