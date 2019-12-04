module findMinMaxEpoch
	export findMMEpoch

	################################################################################
	# Function for finding the epoch with the minimal estimated max risk

	using JLD2, Glob
	findMMEpoch = function(curr_dir;files=nothing)
		cwd = pwd()
		cd(joinpath(curr_dir,"risks"))
		if files==nothing
			files = glob(joinpath("*_maxRisk_*starts.jld2"))
		end
		files = [joinpath(pwd(),fl) for fl in files]
		cd(cwd)
		minMaxEpoch = nothing
		minMaxRisk = Inf
		for i in 1:length(files)
			@load files[i] maxRisk hardestγ unifBayesRisk borderBayesRisk
			if maxRisk<=minMaxRisk
				minMaxRisk = maxRisk
				minMaxEpoch = parse(Int,split(split(files[i],joinpath(curr_dir,"risks/"))[2],"_maxRisk_")[1])
			end
		end
		return minMaxEpoch
	end

end # module