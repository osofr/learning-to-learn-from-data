# Notes for running Example 5 (confidence region construction in a Gaussian mixture model)

The main executable scripts are:
- `01_main_train_script_centersCx.jl`: Julia script for training the estimator of the centers in the confidence region (CR) procedure; and
- `02_main_train_script_offsetsLU.sh`: bash shell script for training the estimator of the offsets in the CR procedure.

The two scripts need to be executed in the following sequence.

- First, run the Julia script `01_main_train_script_centersCx.jl` until completion (for 100K epochs):
```
cd 05_confidence_region_construction
./01_main_train_script_centersCx.jl > 01_main_train_script_centersCx.out
```

- Second, identify the epoch with the best estimator of the centers via interrogation of risks, using the Julia code in `interrogate_risks_01_centersCx.jl`. Alternatively, use a pre-saved neural network estimator of the centers in `/saved_models_centersCx_pretrained/072200.bson`.

- Third, run the bash shell script to train the estimator of the offsets (resulting in final CR procedure):
```
./02_main_train_script_offsetsLU.sh > 02_main_train_script_offsetsLU.out
```

One can modify various options for the estimation of the centers and offsets by modifying the Julia scripts `01_main_train_script_centersCx.jl` and `02_main_train_script_offsetsLU.jl`, respectively. In particular, the latter script can be modified to provide a path for the new estimator of the centers to the offsets estimation procedure. For additional instructions, see the comments in the above scripts. For various options for training the CR procedure, see the arguments that can be provided to the main Julia file `maximinNN1.jl`.

To evaluate the maximal risk(s) of the trained procedure(s), see the code and comments inside the script files:
- `interrogate_risks_01_centersCx.jl`,
- `interrogate_risks_02_shallow_CR_cover_size.jl`
- `interrogate_risks_03_grid_CR_cover_size.jl`
- `interrogate_risks_04_grid_CR_cover_size_lowsigma.jl`

The folders `saved_models_centersCx_pretrained` and `saved_models_offsetsLU`, contain the final LSTM networks for centers and offsets, respectively.

**NOTE**: We do not recommend running the offsets Julia script directly (`interrogate_risks_01_centersCx.jl`), since it does not automatically restart in case of an error (due to memory management bug in `Flux.jl` package). Instead, we suggest running it's  bash shell script counterpart (`02_main_train_script_offsetsLU.sh`), which will automatically restart the training process from pre-saved checkpoint in case of an error.
