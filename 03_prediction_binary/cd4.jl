# Runs the analysis of cd4 data in processed_cd4_data folder.

cd("./03_prediction_binary")

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

# models to use
fldrs = [
	"cd4_linear_logistic_2d_n50_pm05_borders",
	"cd4_linear_logistic_2d_n50_pm1_borders",
	"cd4_linear_logistic_2d_n50_pm2_borders"]

# name of data set
# Note: the code treats the "cd4" dataset specially (because sexbin is binary and should be coded +/- 1)
ds_name = "cd4"

# name of response variable
response_var = "any_response"

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

include("run_data_analysis.jl")


