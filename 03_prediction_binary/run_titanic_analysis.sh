#!/bin/bash

# NOTE: THE INTERMEDIATE OUTPUTS OF THIS ANALYSIS ARE TOO LARGE TO PUT ON
# GITHUB, BUT THE FINAL AUC/CROSS-ENTROPY TABLES ARE STORED HERE

# process the data in R

R CMD BATCH ./process_titanic_data.R

# evaluate our models in julia

julia titanic.jl

# evaluate our performance in R

R CMD BATCH ./titanic_aucs.R
