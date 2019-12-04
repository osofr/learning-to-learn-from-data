# Notes for running Example 2 (Poorly conditioned point estimation)

This repository was used to run the ill-conditioned regression examples
presented in the paper. Note that this example used `Julia v0.7.0` unlike the rest of the numerical experiments, which used `Julia v0.6.2`. Therefore, the example should be run on the provided `Julia v0.7.0` AWS image, rather than on the provided Julia v0.6.2 AWS image.
<br>

These examples were based on the `mtcars` and `mandel` data
sets in R. The CSV files that we used in our julia scripts were generated using the R script `collinear_to_csv.R`.

The script for learning estimators in the ill-conditioned regression problem can be found in `regbeta.jl`. To run the numerical example from the command line:
```
./regbeta.jl > regbeta.out &
```
The results appear in the `mtcars2` and `mandel2` folders in this directory.
<br>

Final performance metrics were evaluated using the file `final_regbeta.jl`. To compute these metrics, we ran the following from the command line:
```
./final_regbeta_interrogate.jl > final_regbeta_interrogate.out &
```
The performance of the final chosen procedure for the `mandel` data set can be evaluated by running the following commands in julia:
```
using JLD2
@load "mandel2/risks/finalMaxRisk.jld2" finalMaxRisk finalMaxRisk_outer finalMaxRisk_uniform epoch
finalMaxRisk
@load "mtcars2/risks/finalMaxRisk.jld2" finalMaxRisk finalMaxRisk_outer finalMaxRisk_uniform epoch
finalMaxRisk
```

Comparator procedures were interrogated in R using the script `interrogate_LM_glmnet.R`.
