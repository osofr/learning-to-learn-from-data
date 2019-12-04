#!/bin/bash

# Evaluates performance of NN3 and NN33 MLEs against least favorable
# data generating mechanism for our learned estimator.


# Because we're running this on the rhino cluster at Fred Hutch,
# need to load the R and Julia modules

ml R

# Run the analyses

R CMD BATCH ./standardApproachesAgainstMinimaxLeastFavorable.R
