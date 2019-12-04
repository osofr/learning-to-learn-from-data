# learning-to-learn-from-data

Code for paper "Learning to learn from data: using deep adversarial learning to construct optimal statistical procedures" by A. Luedtke, M. Carone, N. Simon, and O. Sofrygin. [[link](https://websitelinkhere)]

## Environment
All numerical experiments were performed on AWS GPU instances (`p2.xlarge` and `p3.2xlarge`), using
- [[Julia 0.6.2](github.com/JuliaLang/julia/tree/v0.6.2)],
and the following Julia packages (versions to the right):
- Knet                          0.9.0
- Flux                          0.5.1
- CuArrays                      0.5.0
- ArgParse                      0.5.0
- BSON                          0.1.3
- CLArrays                      0.1.3
- CLFFT                         0.5.2
- CUDAnative                    0.6.3
- CURAND                        0.0.4
- Clustering                    0.9.1
- Distributions                 0.15.0
- JLD                           0.8.3
- JLD2                          0.0.5
- StatsBase                     0.21.0

Since `Julia` version `0.6.2` is no longer maintained, we also provide a publicly available AWS image configured to run our numerical experiments on AWS GPU instances. This image can be accessed by searching for the following AWS Image ID in the region `us-east-2a`:
- ID: `ami-05aa8ea8a7f911839`

We note that Example 2 (poorly conditioned point estimation), and Example 2 only, was run in `Julia` version `0.7.0`. Therefore, we have provided a second AWS Image that can be used to run this example:
- ID: `ami-0e951811ae6a3a747`


## Installation

Install the appropriate versions of Julia/Knet/Flux using the following instructions:
```
git clone git://github.com/JuliaLang/julia.git
cd julia
git checkout v0.6.2
make -j Nproc
```

The repository also includes several `R` scripts. All of these scripts were run on a private cluster, and therefore cannot be run on the AWS Images that we have provided without installing the required packages from [CRAN](https://cran.r-project.org/).

## Replicating numerical experiments

See the specific folders for instructions on running the numerical experiments.

### 01_pointEstimation_basic

This folder contains instructions and code for running the first 3 point-estimation examples (mean, sd & Neal example with inconsistent MLE).

### 02_pointEstimation_illPosed

This folder contains instructions and code for running the two poorly-conditioned linear regression problems.

### 03_prediction_binary

This folder contains instructions and code for running the individual-level prediction problems for binary outcome.

### 04_prediction_cluster_GMM

This folder contains instructions and code for running the clustering example in two-component Gaussian mixture model.

### 05_confidence_region_construction

This folder contains instructions and code for running the experiments for confidence region construction.

## Citation
If you use our code, please consider citing the following:

Luedtke A, Carone M, Simon N, Sofrygin, O. (2020). Learning to learn from data: using deep adversarial learning to construct optimal procedures. <i>Journal</i>.
