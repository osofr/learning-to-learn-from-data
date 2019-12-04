using DataFrames
using CSV
using JLD2
using Glob

# read in the processed data
processed_data = CSV.read("processed_"*ds_name*"_data/processed_data.csv")

# read in the indices of rows in processed_data to use as validation sets
# each row is a different set of validation set indices
val_inds = CSV.read("processed_"*ds_name*"_data/val_inds.csv")


include("maximinNN1.jl")
loadNN = maximinNN1.loadNN
predictTᵏ = maximinNN1.predictTᵏ
cond_mean = maximinNN1.cond_mean

atype = eval(parse("Array{Float32}"))


for fldr in fldrs
	if fldr == "cd4_linear_logistic_2d_n50_pm05_borders"
		psirange = [[-0.5, 0.5], [-0.5, 0.5], [-1.0, 1.0]]
		hiddenReg = [0]
		glmInit = 0
	elseif fldr == "cd4_linear_logistic_2d_n50_pm1_borders"
		psirange = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
		hiddenReg = [0]
		glmInit = 0
	elseif fldr == "cd4_linear_logistic_2d_n50_pm2_borders"
		psirange = [[-2.0, 2.0], [-2.0, 2.0], [-1.0, 1.0]]
		hiddenReg = [0]
		glmInit = 0
	elseif fldr == "linear_logistic_2d_n50_borders"
		psirange = [[-2.0, 2.0] for i in 1:3]
		hiddenReg = [0]
		glmInit = 0
	elseif fldr == "NN3_2d_n50_mixed"
		psirange = [[-2.0, 2.0] for i in 1:13]
		hiddenReg = [3]
		glmInit = 0
	elseif fldr == "NN33_2d_n50_mixed"
		psirange = [[-2.0, 2.0] for i in 1:25]
		hiddenReg = [3, 3]
		glmInit = 0
	elseif fldr == "NN3_2d_n50_mixed_glmInit"
		psirange = [[-2.0, 2.0] for i in 1:13]
		hiddenReg = [3]
		glmInit = 1
	elseif fldr == "NN33_2d_n50_mixed_glmInit"
		psirange = [[-2.0, 2.0] for i in 1:25]
		hiddenReg = [3, 3]
		glmInit = 1
	elseif fldr == "linear_logistic_10d_n50_nointercept_borders"
		psirange = vcat([[-0.5, 0.5] for i in 1:10],[[0.0,0.0]])
		hiddenReg = [0]
		glmInit = 0
	elseif fldr == "linear_logistic_10d_n50_pm1_borders"
		psirange = vcat([[-0.5, 0.5] for i in 1:10],[[-1.0,1.0]])
		hiddenReg = [0]
		glmInit = 0
	elseif fldr == "linear_logistic_10d_n50_borders"
		psirange = vcat([[-0.5, 0.5] for i in 1:10],[[-2.0,2.0]])
		hiddenReg = [0]
		glmInit = 0
	end
	psirange = convert.(atype,psirange)

	selectedEpoch = basename(glob(joinpath(fldr,"risks","*250starts.jld2"))[1])[1:4]
	mPiᵏ, mPiᵣ, mTᵏ = loadNN(atype,joinpath(fldr,"models",selectedEpoch*".jld2"))

	preds = ones(size(val_inds)).+1000.0

	for ind in 1:nrow(val_inds)
		curr_val_inds = convert(Array,val_inds[ind,:])[1,:]
		select_training = trues(nrow(processed_data))
		select_training[curr_val_inds] = false
		traindat = processed_data[select_training,:]
		valdat = processed_data[curr_val_inds,:]


		if ds_name=="cd4"
			# calculate centering and standard deviation for all non-binary columns
			# based on the training data
			mu_bmi = mean(traindat[:bmi])
			sd_bmi = std(traindat[:bmi])

			make_x = function(currdat)
				return [[convert(Array{Float32},[2.*currdat[:sexbin].-1 (currdat[:bmi].-mu_bmi)./sd_bmi])],convert(Array{Float32},currdat[:any_response]),nothing]
			end
		else
			pred_names = setdiff(names(traindat), [Symbol(response_var)])
			mu = colwise(mean,traindat[pred_names])
			sd = colwise(std,traindat[pred_names])
			make_x = function(currdat)
				return [[convert(Array{Float32},hcat([(currdat[pred_names[i]].-mu[i])./sd[i] for i=1:length(pred_names)]...))],convert(Array{Float32},currdat[[Symbol(response_var)]]),nothing]
			end
		end

		xtrain = make_x(traindat)
		coefs = predictTᵏ(mTᵏ,xtrain,psirange,hiddenReg...;glmInit=glmInit)

		xval = make_x(valdat)
		# See the README file for an explanation of why we have 1.-cond_mean(coefs,xval[1]),
		# rather than cond_mean(coefs,xval[1]), on the right-hand side below
		preds[ind,:] = 1.-cond_mean(coefs,xval[1])
	end
	
	if !any(readdir().==ds_name*"_preds")
		mkdir(ds_name*"_preds")
	end
	CSV.write(ds_name*"_preds/"*fldr*".csv",convert(DataFrame,preds))
end
