# Notes for running Example 1 (basic point estimation)

## Estimating the parameters of a normal distribution

The script for learning estimators of a normal mean and standard deviation can be found in `pointEstimates.jl`. To run the numerical example from the command line:
```
./pointEstimates.jl > pointEstimates.out &
```
The maximal risks of the learned procedures at any given iteration can be run using commands like the following in julia:
```
using JLD2
@load "globalNormalMean_n50/risks/maxRisk.jld2" maxRisk riskmat riskEpoch
maxRisk
```
A final estimate of the maximal risk for the best performing learned procedure (across iterations) can be evaluated by first running te following from the command line:
```
./runFinalRisk.jl > runFinalRisk.out &
```
and then subsequently running commands like the following in julia
```
using JLD2
@load "globalNormalMean_n50/risks/finalMinMaxRisk.jld2" finalMaxRisk finalRiskMat minMaxRiskEpoch
finalMaxRisk
```

## Estimating a parameter in a setting where the MLE is inconsistent

The script for learning estimators of the parameter in [Radford Neal's inconsistent MLE example](https://radfordneal.wordpress.com/2008/08/09/inconsistent-maximum-likelihood-estimation-an-ordinary-example/) can be found in `neal.jl`. To run the numerical example from the command line:
```
./neal.jl > neal.out &
```
The risks of the trained procedures can be evaluated by running the following on the command line:
```
./neal_riskgrid.jl > neal_riskgrid.out &
```
The risk of the selected learned procedure can be loaded using commands like the following in julia:
```
using JLD2
@load "globalNeal_n10/risks/maxRisk_estBounded.jld2" maxRisk riskmat riskEpoch
maxRisk
```
The risk of the MLE can be evaluated by running `neal_MLE.R` in R.
