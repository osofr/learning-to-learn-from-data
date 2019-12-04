#!/bin/bash

# NOTE: THE INTERMEDIATE OUTPUTS OF THIS ANALYSIS ARE TOO LARGE TO PUT ON
# GITHUB, BUT THE FINAL AUC/CROSS-ENTROPY TABLES WILL BE STORED HERE

# process the data in R
# (This is commented out because the data is stored on the SCHARP filesystem,
#  and so this is run there rather than on the rhino cluster.
#  See trials/vaccine/p070/adhoc/ALuedtke_2018/code/process_cd4_data.R)

R CMD BATCH ./process_cd4_data.R

# evaluate our models in julia

julia cd4.jl

# evaluate our performance in R
# This is run locally. See trials/vaccine/p070/adhoc/ALuedtke_2018/code/cd4_aucs.R.

R CMD BATCH ./cd4_aucs.R
