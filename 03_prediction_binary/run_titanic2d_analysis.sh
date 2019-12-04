#!/bin/bash

# NOTE: THE INTERMEDIATE OUTPUTS OF THIS ANALYSIS ARE TOO LARGE TO PUT ON
# GITHUB, BUT THE FINAL AUC/CROSS-ENTROPY TABLES ARE STORED HERE

# process the data in R

R CMD BATCH ./process_titanic2d_data.R

# evaluate our models in julia

julia titanic2d.jl

# evaluate our performance in R

R CMD BATCH ./titanic2d_aucs.R
