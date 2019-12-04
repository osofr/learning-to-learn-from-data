# Notes for running Example 4 (clustering in a two-component Gaussian Mixture Model)

The main executable script is `01_main_train_script.jl`. To run the numerical example in batch mode:
```
cd 04_prediction_cluster_GMM
./01_main_train_script.jl > 01_main_train_script.out
```

To run in the example in Julia interactive mode, execute the Julia code inside `01_main_train_script.jl` after running:
```
cd 04_prediction_cluster_GMM
julia
```

For various options, see the arguments that can be provided to the main Julia file: `prediction_cluster_GMM.jl`.

To evaluate the maximal risk(s) of the trained procedure(s), see the comments inside the script file `evaluate_risks/interrogate_risks_01_Tk.jl`.

The folder `saved_models_n10` contains the final neural network under `models`.
