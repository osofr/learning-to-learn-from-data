# Notes for running Example 3 (binary prediction problem)

## Learning and interrogating the procedures

The script for learning estimators of a normal mean and standard deviation can be found in `pointEstimates.jl`. To run the numerical example from the command line:
```
./pointEstimates.jl > pointEstimates.out &
```
Running this script will learn the procedures corresponding to each of the settings listed in Table 1 of the publication. The procedures are contained in the following folders:

| Setting | Folder                                           |
|---------|--------------------------------------------------|
| i       | linear_logistic_2d_n50_borders                   |
| ii      | NN3_2d_n50_mixed                                 |
| iii     | NN33_2d_n50_mixed                                |
| iv      | linear_logistic_10d_n50_nointercept_borders      |
| v       | linear_logistic_10d_n50_pm1_borders              |
| vi      | linear_logistic_10d_n50_borders                  |
| vii     | linear_logistic_10d_n50_nointercept_borders_rho3 |
| viii    | linear_logistic_10d_n50_nointercept_borders_rho6 |
| ix      | linear_logistic_10d_n50_nointercept_borders_rho9 |
| x       | cd4_linear_logistic_2d_n50_pm05_borders          |
| xi      | cd4_linear_logistic_2d_n50_pm1_borders           |
| xii     | cd4_linear_logistic_2d_n50_pm2_borders           |

The aware procedures that provide the procedures in Settings ii and iii with the output of a linear model can be found in `NN3_2d_n50_mixed_glmInit` and `NN33_2d_n50_mixed_glmInit`, respectively.

For a shallow interrogation of the learned procedures, run:
```
./interrogate.jl > interrogate.out &
```
For each setting, procedure that appeared to perform best in the shallow interrogation was then further interrogated, and the corresponding estimate of its maximum risk was reported in the paper. To perform the deeper interrogation, run:
```
./deep_interrogate.jl > deep_interrogate.out &
```
The performance of the final chosen procedure can then be evaluated as follows:
```
using JLD2
@load "./NN33_2d_n50_mixed/risks/3280_maxRisk_250starts.jld2"
maxRisk
@load "./NN33_2d_n50_mixed/risks/3280_maxRisk_50starts.jld2"
maxRisk
```
Note that, for each setting, only one procedure should have files in the `risks` folder ending on `_250starts.jld2` and `_50starts.jld2`. These files indicate that this procedure was the final chosen procedure for the given setting.

Comparator approaches were interrogated in R. The jobs were submitted using the scripts `standardApproachesJob.sh` and `standardApproachesAgainstMinimaxLeastFavorable.sh`.

## Data illustration

Note: The HIV data set `icsresp070.csv` included this repository is a mock-up of the data set used in our analysis, which has the same format but contains different values than the actual data from the HVTN 070 trial.

We ran the data analyses on a private cluster, rather than on the provided AMI. As such, the provided AMIs are not configured to run the scripts for these examples. The scripts that we executed to run our examples can be found in the files `run_titanic2d_analysis.sh`, `run_titanic_analysis.sh`, and `run_cd4_analysis.sh`.

## A bug and an easy fix

After completing some of the training for this experiment, we realized that there is a bug in one of the files. We soon realized that, surprisingly, this bug had no impact on the interrogation of our learned procedures. Moreover, there was a trivial modification that could be made to the output of the final trained procedures to correct for the error, yielding trained procedures that were numerically identical to the procedures that would have been trained had the error never occurred. Because this problem was so easy to correct posthoc, we elected not to retrain all of our procedures on new GPU instances.

We now describe the issue in greater detail. The problem occurs in `param_sim.jl`, namely on those lines on which `z1n[2]` is defined (see the comments in the file indicating these lines). The effect of this bug is that, instead of simulating the data so that we have regression function E[Y|W=w] equal to an adversarially selected function f(w), the data are simulated with regression function E[Y|W=w]=1-f(w). Though the data were evaluated with the regression function 1-f(w), we treated the target of estimation as f(w) when evaluating the loss of the trained procedure. That is, the Statistician was trying to learn f(w)=1-E[Y|W=w], rather than 1-f(w)=E[Y|W=w], in this game.

It is therefore not surprising that, when applying the final learned procedure on a real data set, an estimate of E[Y|W=w] can be obtained by simply computing the estimated Yhat and then subsequently returning 1-Yhat. This is what we did in the file `run_data_analysis.jl` on the line `preds[ind,:] = 1.-cond_mean(coefs,xval[1])`.

What is perhaps more surprising is that the adversarial minimax game that we played in this example gave the same results (once the 1-Yhat transformation is applied to the final output) as it would have in the case that the bug in `param_sim.jl` had not occurred (up to the random initialization of the Statistician's strategy, which would have differed). This interesting phenomenon occurs because the cross-entropy loss function `L(a,b)=-[a*log(b) + (1-a)*log(1-b)]` satisfies the following invariance property: `L(a,b)=L(1-a,1-b)`.
