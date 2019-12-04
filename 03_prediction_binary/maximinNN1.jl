## to run, start julia in parent dir "./funcGrad.jl/Julia"
# julia> include("maximinNN1.jl")
# julia> maximinNN1.main("--nepochs 10 --gpu 1 --fast 1")
## alternatively can be called from command line:
# $ julia maximinNN1 --nepochs 10

# __precompile__()
module maximinNN1

using Knet, ArgParse, JLD2, FileIO#, GLM, DataFrames
using Base.Iterators: repeated, partition
include("./commonNNfuns.jl")
include("./param_model.jl")
include("./param_truth.jl")
include("./param_sim.jl")
include("./SGAhardestPk.jl")
include("./riskgrid.jl")
include("./randomSearch.jl")
export main

## ------------------------------------------------
## Parsing defaults
## ------------------------------------------------
function parse_options(args)
    s = ArgParseSettings()
    s.description =
        "Neural Nets with Bayesian maximin inference for [μ,σ] in bounded Gaussian / log-normal model."

    @add_arg_table s begin
        ("--gpu"; arg_type=Int; default=0; ; help="set to 1 to use gpu")
        ("--nbatchPi"; arg_type=Int; default=1000; help="batch size for Pi")
    ("--nbatchT"; arg_type=Int; default=1000; help="batch size for T")
        ("--ntest"; arg_type=Int; default=10000; help="size of the test set")
        ("--udim"; arg_type=Int; default=2; help="dimension of noise U -> P(U)")
        ("--odim"; arg_type=Int; default=1; help="dimension of observation")
        ("--n"; arg_type=Int; default=10; help="sample size (number of observations)")
        ("--nepochs"; arg_type=Int; default=50; help="# of training iterations (note that nepochs is a misnomer, really these are just number of iterations performed)")
        ("--niter"; arg_type=Int; default=10; help="# of training training steps within each epoch (within each iteration -- epochs is a misnomer)")
    ("--nT"; arg_type=Int; default=10; help="# of steps on T for every nT steps on Pi")
    ("--nPi"; arg_type=Int; default=1; help="# of steps on Pi for every nT steps on T")
        ("--seed"; arg_type=Int; default=54321; help="random seed")
        ("--parsrange"; nargs='*'; default = ["[[-2.5f0, 2.5f0],", "[1.0f0, 5.0f0]]"]; arg_type=String; help="range of all parameters at once")
    ("--psirange"; nargs='*'; default = ["[[-2.5f0, 2.5f0],", "[1.0f0, 5.0f0]]"]; arg_type=String; help="range of all psi inds at once")
        ("--truepsi"; default="Ψμ"; help="function for evaluation of the true parameter value under P0")
        ("--name"; default="norm"; help="distribution name for P0")
        ("--optimPi"; default="Adam(lr=0.002, beta1=0.9)")
        ("--optimT"; default="Adam(lr=0.001, beta1=0.9)")
        ("--loadfile"; default=nothing; help="file to load trained models")
        ("--outdir"; default=nothing; help="output dir for models/generations")
        ("--saveEvery"; arg_type=Int; default=1; help="save network every saveEvery epochs")
        ("--hiddenPi"; nargs='*'; default = [8,16,16,8]; arg_type=Int; help="sizes of hidden layers for Piᵏ, defaults to --hidden 8 16 16 8 for a net with 4 hidden layers")
        ("--innerSGAnsteps"; default=500; arg_type=Int; help="number of steps to take on inner SGA, only used if hiddenPi equals 0")
        ("--innerSGAnstartsBorder"; default=0; arg_type=Int; help="when training T using SGA, number of random starts on the border")
        ("--innerSGAnstartsMixed"; default=0; arg_type=Int; help="when training T using SGA, number of random starts with some indices starting on the border and some drawn uniformly")
       ("--innerSGAnbatch"; default=50; arg_type=Int; help="inner SGA batch size")
       ("--innerSGAntest"; default=1000; arg_type=Int; help="inner SGA ntest (especially important if performing a random grid search)")
       ("--repsPerGamma"; default=1; arg_type=Int; help="when trainingT, number of draws to take from each Gamma")
        ("--hiddenT"; nargs='*'; default = [20,20,2];  arg_type=Int; help="sizes of hidden layers for Tᵏ, defaults to --hidden 20 20 2 a net with 3 hidden layers")
        ("--hiddenReg"; nargs='*'; default = [2,2];  arg_type=Int; help="sizes of hidden layers for regression network, defaults to --hidden 2 2 a net with 2 hidden layers")
        ("--maxRiskEvery"; arg_type=Int; default=1; help="report maximum risk of current estimator every maxRiskEvery epochs, using SGA for hardest risk")
        ("--SGAnruns"; default=300; arg_type=Int; help="SGA number of gradient updates (for each random start)")
        ("--SGAnstarts"; default=10; arg_type=Int; help="SGA number of independent random starts of parameters γ")
        ("--SGAnbatch"; default=100; arg_type=Int; help="SGA batch size")
        ("--Rgrid"; default=0; arg_type=Int; help="evaluate the risk surface at resolution `Rgridsize` each time SGA runs")
        ("--Rgridsize"; default=0.1f0; arg_type=Float32; help="the size of the step in each parameter for risk surface evaluation")
        ("--verbose"; default=1; arg_type=Int; help="print diagnostics, e.g. the range of the prior and of the estimator")
        ("--maxRiskInit"; default=1; arg_type=Int; help="compute the max risk of the initial network? 0=No, 1=Yes")
        ("--plot"; default=0; help="real slow plotting of risk surfaces at each epoch")
        ("--interrogateEpoch"; default=-1; arg_type=Int; help="only give the max risk for the specified saved NN. If interrogateEpoch is -1, then does not interrogate (instead trains network). If interrogateEpoch is -2, then interrogates the procedure with weights averaged across saved procedure networks from the models folder.")
        ("--interrogationStarts"; default=5; arg_type=Int; help="if interrogateEpoch is true, then this defines the number of starts of the Piᵏ network for the interrogation (default is 5)")
        ("--numInterDS"; default=1; arg_type=Int; help="number of data sets (divided by 5000) to use on crossentropy2 when evaluating the max risk")
        ("--interrogationngridBorder"; default=1; arg_type=Int; help="number of grid points on the border during each step of the interrogation random search")
        ("--interrogationngrid"; default=1; arg_type=Int; help="number of uniformly chosen grid points during each step of the interrogation random search")
        ("--interrogationSE"; default=0; arg_type=Int; help="output standard error for interrogation?")
        ("--glmInit"; default=0; arg_type=Int; help="Provide GLM coefficients as input to LSTM network? 0=no,1=yes. Note: This currently only works for a two-dimensional bivariate normal predictor with identity covariance matrix (is hard-coded for this case). Includes intercept, main linear terms, main quadratic terms, linear interaction.")

    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    if o[:outdir] != nothing
        o[:outdir] = abspath(o[:outdir])
    end
    return o
end

function main(args="")
    # args="";
    atypeCPU = "Array{Float32}"
    atypeGPU = "KnetArray{Float32}"
    o = parse_options(args)
    o[:seed] > 0 && srand(o[:seed]);
    if o[:gpu]==0
        atype = eval(parse(atypeCPU))
    else
        atype = eval(parse(atypeGPU))
    end

    processRange = function(theRange;psitype=false)
        ## parse the parameter range
        theRange = reduce(*, theRange)
        theRange = eval(parse(theRange))
    if psitype
        theRange = [convert(atype,psi) for psi in theRange]
    end
        # If only one range was included then duplicate this range for all parameters
        if length(theRange)==1
            numParams = 0
            currdim = o[:odim]-1
            if [o[:hiddenReg]...] == [0]
                loopover = [1]
            else
                loopover = vcat(o[:hiddenReg]...,[1])
            end
            for nextdim in loopover
                numParams += nextdim*(currdim+1)
                currdim=nextdim
            end
            theRange = map(i->theRange[1],1:numParams)
        end
    return theRange
    end

    o[:parsrange] = processRange(o[:parsrange])
    o[:psirange] = processRange(o[:psirange];psitype=true)
    
    ## define the model parameter space
    parbox = ParamBox([convert(atype, par) for par in o[:parsrange]])
    modelP = ParamModelPᵏ(parbox, Ψ=Symbol(o[:truepsi]); name = parse(o[:name]), n=o[:n], odim=o[:odim])
    # modelP = ParamModelPᵏ(ParamBox([[0.0f0, 5.0f0], [1.0f0, 3.0f0]]), Ψ=:Ψμ; name=:norm)

    if (o[:verbose]==1)
        println("Range for each parameter: ")
        println("hiddenPi, network for Piᵏ: ", o[:hiddenPi])
        println("hiddenT, network for Tᵏ: ",  o[:hiddenT])
        println("ParamBox:"); println(parbox)
        println("Using Ψ: $(Symbol(o[:truepsi]))")
    end

    if o[:interrogateEpoch]>=0
        o[:loadfile] = joinpath(o[:outdir],"models",@sprintf("%04d.jld2",o[:interrogateEpoch]))
    elseif o[:interrogateEpoch]==-2
        o[:loadfile] = "interrogate_average"
    end

    if (o[:loadfile]==nothing)
        # def 3 NNs: mPiᵏ -- hardest prior; mPiᵣ -- hardest diffuse prior; mTᵏ -- best estimator
        if (o[:verbose]==1) println("initializing NNs"); end
        if [o[:hiddenPi]...]!=[0]
            mPiᵏ = initPiᵏ(o[:hiddenPi]...; atype=atype, winit=1.5, udim = o[:udim], outdim = length(modelP));
            mPiᵣ = deepcopy(mPiᵏ)
        end
        mTᵏ = initTᵏ(o[:hiddenT]...; atype=atype, winit=-0.5, odim = o[:odim], outdim = length(modelP), glmInit = o[:glmInit]);
    else
        if (o[:loadfile]=="interrogate_average")
            println("a")
            all_models = readdir(joinpath(o[:outdir],"models"))
            all_models = all_models[Int(floor(length(all_models)*0.75)):length(all_models)]
            @load joinpath(o[:outdir],"models",all_models[1]) Pi PiDiffuse T
            mTᵏ = T
            for i = 2:length(all_models)
                @load joinpath(o[:outdir],"models",all_models[i]) Pi PiDiffuse T
                mTᵏ = (i*mTᵏ + T)/(i+1)
            end
            mPiᵏ = Pi
            mPiᵣ = PiDiffuse
            mPiᵏ = convertNN(mPiᵏ, atype)
            mPiᵣ = convertNN(mPiᵣ, atype)
            mTᵏ  = convertNN(mTᵏ, atype) 
            println("b")
        else
            # load 3 pre-saved NNs
            if (o[:verbose]==1) println("loading existing NN"); end
            mPiᵏ, mPiᵣ, mTᵏ = loadNN(atype, o[:loadfile])
        end
    end
    if [o[:hiddenPi]...]!=[0]
        nets = Dict(:mPiᵏ => mPiᵏ, :mPiᵣ => mPiᵣ, :mTᵏ => mTᵏ)
    else
        nets = Dict(:mTᵏ => mTᵏ)
    end

    ## lets make some noise!
    noise = sample_noise(atype,o[:udim],o[:nbatchPi]);
    ## sample parameters from the prior NN
    #γsamp = predictPiᵏ(nets[:mPiᵣ],noise,modelP,o[:hiddenReg]...);

    # if (o[:verbose]==1)
    #     println("init min μ: $(minimum(γsamp[1])); init max μ: $(maximum(γsamp[1]))")
    #     println("init min σ: $(minimum(γsamp[2])); init max σ: $(maximum(γsamp[2]))")
    # end

    if o[:outdir] != nothing #&& !isdir(o[:outdir])
        mkpath(o[:outdir])
        mkpath(joinpath(o[:outdir],"models"))
        mkpath(joinpath(o[:outdir],"risks"))
    end
    ## training loop over epochs

    if o[:interrogateEpoch]==-1
        maxRisk, riskmat, riskEpoch = trainepochs(o,nets,modelP,atype)
        return maxRisk, riskmat, riskEpoch
    else
        if o[:outdir] != nothing
            # getting Bayes risk under uniform prior over border
            z1ntest = allocX1n(modelP, o[:ntest]; atype=atype);
            γsamples = [bounds(modelP, rand([1,2], length(lb(modelP)))) for i=1:100]
            γsamples = convert.(atype,γsamples)
            γsamples = makeRegNN(hcat(γsamples...),o[:hiddenReg]...;atype=atype,wdim=modelP.odim-1)
            borderBayesRisk = Tᵏloss(nets[:mTᵏ],γsamples,z1ntest,modelP,o[:psirange],o[:repsPerGamma],atype,o[:hiddenReg]...;glmInit=o[:glmInit])
            println(borderBayesRisk)

            # getting Bayes risk under uniform prior
            z1ntest = allocX1n(modelP, o[:ntest]; atype=atype);
            γsamples = rand(modelP, o[:ntest])
            γsamples = convert.(atype,γsamples)
            γsamples = makeRegNN(hcat(γsamples...),o[:hiddenReg]...;atype=atype,wdim=modelP.odim-1)
            unifBayesRisk = Tᵏloss(nets[:mTᵏ],γsamples,z1ntest,modelP,o[:psirange],o[:repsPerGamma],atype,o[:hiddenReg]...;glmInit=o[:glmInit])
            println(unifBayesRisk)

            getMaxRisk = function(γ;m=2500,varOut=false)
                z1n = allocX1n(modelP, m; atype=atype)
                z1n = sim(modelP, γ, z1n; diffγ=false)
                Tᵏ = predictTᵏ(nets[:mTᵏ],z1n,o[:psirange],o[:hiddenReg]...;glmInit=o[:glmInit])
                crossentropy2(Tᵏ, γ, [], modelP; atype=atype, nMC=1000, varOut=varOut)
            end

            maxRisk = nothing
            hardestγ = nothing
            for _ in 1:o[:interrogationStarts]
                if([o[:hiddenPi]...]!=[0])
                    mPiᵏ = initPiᵏ(o[:hiddenPi]...; atype=atype, winit=1.5, udim = o[:udim], outdim = length(modelP));
                    mPiᵣ = []
                    z1nPi = allocX1n(modelP, o[:nbatchPi]; atype=atype);
                    optPiᵏ = map(wi->eval(parse(o[:optimPi])), mPiᵏ); # optPiᵏ = optimizers(:mPiᵏ, Adam; lr=0.001, beta1=0.9);
                    noise = sample_noise(atype,o[:udim],o[:nbatchPi])
                    avgLossShort = 0
                    avgLossFar = 0
                    i = 0
                    while (i<10000) & !((i>500) & ((avgLossFar-avgLossShort)<0.00001))
                        i+=1
                        noise = sample_noise(atype,o[:udim],o[:nbatchPi])
                        currLoss = trainPiᵏ!(mPiᵏ,mPiᵣ,nets[:mTᵏ],noise,z1nPi,modelP,optPiᵏ,o,atype)
                        avgLossFar = 0.995*avgLossFar + 0.005 * currLoss
                        avgLossShort = 0.98*avgLossShort + 0.02 * currLoss
                        if ((mod(i,25)==0) & (o[:verbose]==1))
                            println((:iter,i,:currLoss,round(currLoss,4),:avgLossShort,round(avgLossShort,4),:avgLossFar,round(avgLossFar,4),:term,round((avgLossFar-avgLossShort),5)))
                        end
                    end

                    noise = sample_noise(atype,o[:udim],500)
                    γdraws = predictPiᵏ(mPiᵏ,noise,modelP,o[:hiddenReg]...)
                    currMaxRisks = broadcast(i->getMaxRisk(broadcast(layer->[layer[i]],γdraws)),1:length(γdraws[1]))
                    currMaxRiskInd = indmax(currMaxRisks)
                    currHardestγ = broadcast(layer->[layer[currMaxRiskInd]],γdraws)
                else 
                    predictFun(z1n) = predictTᵏ(nets[:mTᵏ],z1n,o[:psirange],o[:hiddenReg]...;glmInit=o[:glmInit])
                    currHardestγ = randomSearch(o,predictFun,modelP,atype)
                end
                currMaxRisk = mean(broadcast(k->getMaxRisk(currHardestγ;m=5000),1:o[:numInterDS]))

                println((:currMaxRisk,currMaxRisk))
                if maxRisk == nothing
                    maxRisk = currMaxRisk
                    hardestγ = currHardestγ
                elseif currMaxRisk>maxRisk
                    maxRisk = currMaxRisk
                    hardestγ = currHardestγ
                end
            end
            # Max risk estimate will be biased up, run getMaxRisk to remove this bias
            mFinal = 5000
            if o[:interrogationSE]==1
                numRep = 2*o[:numInterDS]
                tmp = broadcast(k->getMaxRisk(hardestγ;m=mFinal,varOut=true),1:(numRep))
                maxRisk = mean(broadcast(x->x[1],tmp))
                se = sqrt((mean(broadcast(x->x[2]+x[1]^2,tmp)) - maxRisk^2)/(mFinal*numRep))
            else
                maxRisk = mean(broadcast(k->getMaxRisk(hardestγ;m=mFinal),1:(2*o[:numInterDS])))
            end
            
	    if o[:interrogateEpoch]>=0
                filename = @sprintf("%04d",o[:interrogateEpoch])*"_maxRisk_"*string(o[:interrogationStarts])*"starts.jld2"
            elseif o[:interrogateEpoch]==-2
                filename = "averagedWeights_maxRisk_"*string(o[:interrogationStarts])*"starts.jld2"
            end
            filepath = joinpath(o[:outdir],"risks",filename)
            if o[:interrogationSE]==1
                save(filepath,
                    "maxRisk", maxRisk,
                    "hardestγ", [convert.(Array{Float32,2},hardestγ[i]) for i in 1:length(hardestγ)],
                    "unifBayesRisk", unifBayesRisk,
                    "borderBayesRisk", borderBayesRisk,
                    "se", se)
            else
                save(filepath,
                    "maxRisk", maxRisk,
                    "hardestγ", [convert.(Array{Float32,2},hardestγ[i]) for i in 1:length(hardestγ)],
                    "unifBayesRisk", unifBayesRisk,
                    "borderBayesRisk", borderBayesRisk)
            end
        end
        return nothing
    end
end

function computeSGA(o,nets,modelP,atype=Array{Float32}; allOut=false)
    return computeSGA2(o[:psirange],o[:SGAnruns],0,o[:SGAnstarts],o[:SGAnbatch],nets[:mTᵏ],atype,modelP,o[:hiddenReg]...; allOut=allOut, glmInit=o[:glmInit])
end

function computeSGA2(psirange,SGAnruns,SGAnstartsBorder,SGAnstarts,SGAnbatch,mTᵏ,atype,modelP,hiddenReg...; allOut=false, ntest=1000, nMC=100, arrayOut=false, xinit=[], lr=0.001, SGAnstartsMixed=0, glmInit=0)
    modelTᵏ(z1n) = predictTᵏ(mTᵏ,z1n,psirange,hiddenReg...;glmInit=glmInit)
    return findPᵏ(modelTᵏ,modelP,hiddenReg...;atype=atype,nruns=SGAnruns,nstarts=SGAnstarts,nbatch=SGAnbatch,allOut=allOut,nstartsBorder=SGAnstartsBorder,nMC=nMC,arrayOut=arrayOut,xinit=xinit,lr=lr, ntest=ntest, nstartsMixed=SGAnstartsMixed);
end

function trainepochs(o,nets,modelP,atype=Array{Float32})
    maxRisk = Vector{Float32}(div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit])
    riskEpoch = Vector{Int}(div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit])
    grid = linspace(modelP.bounds, step=o[:Rgridsize])
    riskmat = zeros(Float32, (length(grid[1]), length(grid[2]), div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit]))
    if [o[:hiddenPi]...]!=[0]
        optPiᵏ = map(wi->eval(parse(o[:optimPi])), nets[:mPiᵏ]);
    end
    optTᵏ  = map(wi->eval(parse(o[:optimT])), nets[:mTᵏ]);
    if o[:repsPerGamma]==1
        z1nT = allocX1n(modelP, o[:nbatchT]; atype=atype);
    else
        z1nT = allocX1n(modelP, o[:repsPerGamma]; atype=atype);
    end
    z1nPi = allocX1n(modelP, o[:nbatchPi]; atype=atype);
    z1ntest = allocX1n(modelP, o[:ntest]; atype=atype);
    println("training started..."); flush(STDOUT)

    if (o[:maxRiskInit]==1)
        riskEpoch[1] = 0
        maxRisk[1], hardestγ = computeSGA(o,nets,modelP,atype)
println(broadcast(x->convert(Array{Float32,2},x[1]),hardestγ))
    println((:epoch,0,:trueMaxRisk,maxRisk[1]));
        if o[:Rgrid]==1
            println("computing risk surface with grid size $(o[:Rgridsize])")
            riskmat[:,:,1] = riskgrid(o,nets,modelP,grid[1],grid[2];glmInit=o[:glmInit])
            println("max riskmat: $(maximum(riskmat[:,:,1]))")
        end
    end
    ## save init models
    if o[:outdir] != nothing
        filename = @sprintf("%04d.jld2",0)
        filepath = joinpath(o[:outdir],"models",filename)
    if [o[:hiddenPi]...]!=[0]
            saveNN(filepath,nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ])
    else
        saveNN(filepath,[],[],nets[:mTᵏ])
    end
    end

    for epoch = 1:o[:nepochs]
        dlossval = glossval = 0
        for _ in 1:o[:niter]
            if [o[:hiddenPi]...]!=[0]
                for _ in 1:o[:nPi]
                        noise = sample_noise(atype,o[:udim],o[:nbatchPi])
                        dlossval += trainPiᵏ!(nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ],noise,z1nPi,modelP,optPiᵏ,o,atype)
                end
            end
            noise = sample_noise(atype,o[:udim],o[:nbatchT])
            if [o[:hiddenPi]...]!=[0]
                γsamples = predictPiᵏ(nets[:mPiᵏ],noise,modelP,o[:hiddenReg]...)
            else
                _, γsamples = computeSGA2(o[:psirange],o[:innerSGAnsteps],o[:innerSGAnstartsBorder],o[:nbatchT],o[:innerSGAnbatch],nets[:mTᵏ],atype,modelP,o[:hiddenReg]...; allOut=false, ntest=o[:innerSGAntest],SGAnstartsMixed=o[:innerSGAnstartsMixed], glmInit=o[:glmInit]) #computeSGA(o,nets,modelP,atype; allOut=true)
            end
            for _ in 1:o[:nT]
                glossval += trainTᵏ!(γsamples,nets[:mTᵏ],noise,z1nT,modelP,optTᵏ,o,atype)
            end
        end
        dlossval /= o[:niter]; glossval /= (o[:niter]*o[:nT]);

        if (o[:verbose]==1)
        ## evaluate both losses on new test data
            if [o[:hiddenPi]...]!=[0]
        noise = sample_noise(atype,o[:udim],o[:ntest]);
                γsamples = predictPiᵏ(nets[:mPiᵏ],noise,modelP,o[:hiddenReg]...)
                dTESTloss = Piᵏloss(nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ],noise,z1ntest,modelP,o[:psirange],atype,o[:hiddenReg]...;glmInit=o[:glmInit]);
            else
        # test against a random sample of gammas
                if length(border(modelP))<=o[:ntest]
                γsamples = [border(modelP); rand(modelP, o[:ntest]-size(border(modelP),1))]
        else
            γsamples = rand(modelP,o[:ntest])
            end
        γsamples = convert.(atype,γsamples)
        γsamples = makeRegNN(hcat(γsamples...),o[:hiddenReg]...;atype=atype,wdim=modelP.odim-1)
        dTESTloss = 0.0
            end
            gTESTloss = Tᵏloss(nets[:mTᵏ],γsamples,z1ntest,modelP,o[:psirange],o[:repsPerGamma],atype,o[:hiddenReg]...;glmInit=o[:glmInit]);
        # γsamp = predictPiᵏ(nets[:mPiᵏ],noise,modelP,o[:hiddenReg]...)
    z1ntest, _ = sim(modelP, γsamples, z1ntest)
        predT = predictTᵏ(nets[:mTᵏ],z1ntest,o[:psirange],o[:hiddenReg]...;glmInit=o[:glmInit]);
    z1ntest, _ = sim(modelP, γsamples, z1ntest)
    condMeanPi = convert(Array{Float32,2},z1ntest[3])
    condMeanT = convert(Array{Float32,2},cond_mean(predT,z1ntest[1]))
    condMeanPi = reshape(condMeanPi,length(condMeanPi))
    condMeanT = reshape(condMeanT,length(condMeanT))
    condMeanRatio = condMeanT./condMeanPi
            println((:epoch,epoch,:Piloss,round(dTESTloss,3),:Tloss,round(gTESTloss,3), :PiQ, round.(quantile(condMeanPi,[0.05,0.25,0.5,0.75,0.95]),2), :TQ, round.(quantile(condMeanT,[0.05,0.25,0.5,0.75,0.95]),2), :RatioQ, round.(quantile(condMeanRatio,[0.05,0.25,0.5,0.75,0.95]),2)))
        flush(STDOUT)
        end

        if (epoch%o[:maxRiskEvery]==0)
            riskEpoch[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]] = epoch
            maxRisk[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]], hardestγ = computeSGA(o,nets,modelP,atype)
println(broadcast(x->convert(Array{Float32,2},x[1]),hardestγ))
            println((:epoch,epoch,:trueMaxRisk,maxRisk[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]]));
            if o[:Rgrid]==1
                println("computing risk surface with grid size $(o[:Rgridsize])")
                riskmat[:,:,div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]] = riskgrid(o,nets,modelP,grid[1],grid[2])
                println("max riskmat: $(maximum(riskmat[:,:,div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]]))")
            end
        end

        ## save models and generations
        if ((o[:outdir] != nothing) & (epoch%o[:saveEvery]==0))
            filename = @sprintf("%04d.jld2",epoch)
            filepath = joinpath(o[:outdir],"models",filename)
            if [o[:hiddenPi]...]!=[0]
                saveNN(filepath,nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ])
            else
                saveNN(filepath,[],[],nets[:mTᵏ])
            end
        end
    end

    if o[:outdir] != nothing
        filename = "maxRisk.jld2"
        filepath = joinpath(o[:outdir],"risks",filename)
        save(filepath,
            "maxRisk", maxRisk,
            "riskmat", riskmat,
            "riskEpoch", riskEpoch)
    end
    return maxRisk, riskmat, riskEpoch
end

function initPiᵏ(h...; atype=Array{Float32}, winit=1.5, udim = 2, outdim = 2)
    w = Any[]
    for nextd in [h..., outdim] #
        push!(w, convert(atype, winit*randn(Float32,nextd,udim)))
        push!(w, convert(atype, zeros(Float32,nextd,1)))#0.5*2.*randn(Float32,nextd,1)-1.0))
        udim = nextd
    end
    return w
end

function initTᵏ(h...; atype=Array{Float32}, winit=1.5, odim = 2, outdim = 1, glmInit = 0)
    num_node = [h...][1]
    if glmInit==1 # include estimated glm coefficients as input to LSTM?
        weight = convert(atype, winit*randn(Float32,4*num_node,odim + 6 + num_node))
    else
        weight = convert(atype, winit*randn(Float32,4*num_node,odim + num_node))
    end
    bias = convert(atype, zeros(Float32,4*num_node,1))
    finalweight = convert(atype, winit*randn(Float32,outdim,num_node))
    finalbias = convert(atype, zeros(Float32,outdim,1))
    w = Any[]
    push!(w,weight)
    push!(w,bias)
    push!(w,finalweight)
    push!(w,finalbias)
    return w
end

function loadNN(atype,loadfile=nothing)
    @load loadfile Pi PiDiffuse T
    mPiᵏ = convertNN(Pi, atype)
    mPiᵣ = convertNN(PiDiffuse, atype)
    mTᵏ  = convertNN(T, atype)
    return mPiᵏ, mPiᵣ, mTᵏ
end

function saveNN(savefile,mPiᵏ,mPiᵣ,mTᵏ)
    save(savefile,
         "Pi",        convertNN(mPiᵏ,Array{Float32}),
         "PiDiffuse", convertNN(mPiᵣ,Array{Float32}),
         "T",         convertNN(mTᵏ,Array{Float32})
         )
end

function convertNN(w, atype=Array{Float32})
    w0 = map(wi->convert(atype, wi), w)
    w1 = convert(Array{Any}, w0)
end

function predictPiᵏ(w,x,modelP,hiddenReg...)
    x = mat(x)
    for i=1:2:length(w)-3
    x = Knet.relu.(w[i]*x .+ w[i+1])
    end
    x = Knet.sigm.(w[end-1]*x .+ w[end])

    x = vcat(map(i->lb(modelP)[i].+x[i:i,:].*(ub(modelP)[i].-lb(modelP)[i]),1:length(modelP))...)

    x = makeRegNN(x,hiddenReg...;atype=typeof(x), wdim = modelP.odim-1)

    return x
end

function lstm(w,x,hidden,cell,gates)
    hsize = size(hidden,1)
    gates = w[1]*vcat(x,hidden) .+ w[2]
    cell = cell .* Knet.sigm.(gates[1:hsize,:]) + Knet.sigm.(gates[1+hsize:2hsize,:]) .* Knet.tanh.(gates[1+3hsize:4hsize,:])
    hidden = Knet.sigm.(gates[1+2hsize:3hsize,:]) .* Knet.tanh.(cell)
    return (hidden,cell)
end

my_lm_broadcast = function(xlist)
    # population-level estimate of inv(xx' * xx) -- allowed because marginal distribution of X is known
    xxtxx_inv = convert(typeof(xlist[1]),[
        2f0 -0.5f0 -0.5f0 0f0 0f0 0f0 ;
        -0.5f0 0.5f0 0f0 0f0 0f0 0f0 ;
        -0.5f0 0f0 0.5f0 0f0 0f0 0f0 ;
        0f0 0f0 0f0 1f0 0f0 0f0 ;
        0f0 0f0 0f0 0f0 1f0 0f0 ;
        0f0 0f0 0f0 0f0 0f0 1f0])./size(xlist[1],1)
    return xxtxx_inv * hcat(broadcast(x->hcat(convert(typeof(x[:,1:1]),ones(size(x,1),1)),abs2.(x[:,1:2]),x[:,1:2],x[:,1:1].*x[:,2:2])' * x[:,size(x,2):size(x,2)],xlist)...)
end

function predictTᵏ(w,x,psirange,hiddenReg...;glmInit=0)
    hsize = div(size(w[1],1),4)
    gates = convert(typeof(x[1][1]),zeros(Float32,size(w[1],1),length(x[1])))
    hidden = convert(typeof(x[1][1]),zeros(Float32,hsize,length(x[1])))
    cell = convert(typeof(x[1][1]),zeros(Float32,hsize,length(x[1])))
    out = convert(typeof(x[1][1]),zeros(Float32,size(w[length(w)],1),length(x[1])))
    odim = size(x[1][1],2)+1
    numobs = size(x[1][1],1)
    burn_in = div(size(x[1][1],1),2)
    num_inds = numobs + burn_in
    m = length(x[1])
    # create a length n vector of odim x m arrays
    if glmInit==1
        glm_coefs = my_lm_broadcast(broadcast(i->hcat(x[1][i],x[2][:,i:i]),1:m))
    end
    x_prime = vcat(broadcast(i->hcat(x[2][:,i:i],x[1][i]),1:m)...)'
    x_prime = broadcast(i->x_prime[:,i+(((1:m)-1)*numobs)],1:numobs)
    for t in 1:num_inds
        if glmInit==1
            hidden,cell = lstm(w,vcat(x_prime[mod(t,numobs)+1],glm_coefs),hidden,cell,gates)
        else
            hidden,cell = lstm(w,x_prime[mod(t,numobs)+1],hidden,cell,gates)
        end
        if t>burn_in
            out = out .+ (w[3]*hidden .+ w[4])
        end
    end
    out = Knet.sigm.(out./(num_inds-burn_in))
    out = vcat(map(i->lintrans(out[i:i,:],psirange[i][1],psirange[i][2]),1:length(psirange))...)
    out = makeRegNN(out,hiddenReg...;atype=typeof(out), wdim = odim-1)
    return out
end

## loss for hardest distr. Piᵏ, negative Risk
function Piᵏloss(mPiᵏ,mPiᵣ,mTᵏ,noise,z1n,modelP,psirange,atype,hiddenReg...;glmInit=0)
    # 1. generate a bunch of candidates, vectors [mu, sigma],
    #    in response to a batch-size noise vector
    γsamples = predictPiᵏ(mPiᵏ,noise,modelP,hiddenReg...) # γsamples = mPiᵏ(noise)
    # 2. generate a random sample x1n (each row size n)
    x1n, weights = sim(modelP, γsamples, z1n)
    # 3. evaluate estimator Tᵏ(xn1) for each sample from γsamples
    Tᵏ = predictTᵏ(mTᵏ, x1n, psirange, hiddenReg...;glmInit=glmInit)
    # 4. evaluate the truth (separate true psi0 for each param in γsamples)
    trueΨ = Ψ(modelP, γsamples)
    # 5. evaluate the loss as -MSE(Tᵏ(x1n),trueΨ) for each param in γsamples
    RiskTᵏPiᵏ = crossentropy2(Tᵏ, trueΨ, weights, modelP; atype=atype)
    if mPiᵣ!=[]
        # 6. evaluate the penalty for mode collapse wrt diffuse prior mPiᵣ
        γsamples = predictPiᵏ(mPiᵣ,noise,modelP,hiddenReg...)
        x1n = sim(modelP, γsamples, z1n; diffγ=false)
        Tᵏ = predictTᵏ(mTᵏ, x1n, psirange, hiddenReg...;glmInit=glmInit)
        trueΨ = Ψ(modelP, γsamples)
        RiskTᵏPiᵣ = crossentropy2(Tᵏ, trueΨ, [], modelP; atype=atype)
    else
        RiskTᵏPiᵣ = 0
    end
    λpenalty = (relu(RiskTᵏPiᵣ-RiskTᵏPiᵏ))^2
    loss = -RiskTᵏPiᵏ + 75*λpenalty
    return loss
end
Piᵏlossgradient = gradloss(Piᵏloss)

function Tᵏloss(mTᵏ,γsamples,z1n,modelP,psirange,repsPerGamma,atype,hiddenReg...;glmInit=0)
    getLoss = function(γ)
        # 1. generate batch of observed data under new Piᵏ
        x1n = sim(modelP, γ, z1n; diffγ=false)
        # 2. evaluate the estimator under new data
        Tᵏ = predictTᵏ(mTᵏ, x1n,psirange, hiddenReg...;glmInit=glmInit)
        # 3. evaluate the truth and loss (MSE(Tᵏ,))
        trueΨ = Ψ(modelP, γ)
        # 4. evaluate the loss as MSE(Tᵏ(x1n),trueΨ) for each param in γsamples
        return crossentropy2(Tᵏ,trueΨ, [], modelP; atype=atype)
    end
    if repsPerGamma==1
        return getLoss(γsamples)
    else
        loss = 0
        for i in 1:length(γsamples[1])
            loss += getLoss(broadcast(curr->[curr[i]],γsamples))
        end
        return loss/length(γsamples[1])
    end
end
Tᵏlossgradient = gradloss(Tᵏloss)

## normal
function sample_noise(atype,udim,nsamples,mu=0.5f0,sigma=0.5f0)
    num_normal = div(udim,2)+mod(udim,2)
    noise = randn(Float32, udim, nsamples)
    noise[1:num_normal,:] = (noise[1:num_normal,:] .- mu) ./ sigma
    noise[(num_normal+1):end,:] = 2.*(noise[(num_normal+1):end,:].>0).-1
    noise = convert(atype,noise)
    return noise
end

## Performs a single training step for hardest Piᵏ (mu, sigma)
## maximize risk R(Piᵏ,Tᵏ) for Piᵏ wrt input noise vector
function trainPiᵏ!(mPiᵏ,mPiᵣ,mTᵏ,noise,z1n,modelP,optPiᵏ,o,atype)
    # reset diffuse prior
    mPiᵣ = initPiᵏ(o[:hiddenPi]...; atype=atype, winit=1.5, udim = o[:udim], outdim = length(modelP))

    gradients, lossval = Piᵏlossgradient(mPiᵏ,mPiᵣ,mTᵏ,noise,z1n,modelP,o[:psirange],atype,o[:hiddenReg];glmInit=o[:glmInit])
    update!(mPiᵏ, gradients, optPiᵏ)
    return lossval
end

## Performs a single training for the estimator Tᵏ(x1n)
## minimize risk R(Piᵏ,Tᵏ) for Tᵏ wrt input data x1n that derives from Piᵏ
function trainTᵏ!(γsamples,mTᵏ,noise,z1n,modelP,optTᵏ,o,atype)
    gradients, lossval = Tᵏlossgradient(mTᵏ,γsamples,z1n,modelP,o[:psirange],o[:repsPerGamma],atype,o[:hiddenReg];glmInit=o[:glmInit])
    update!(mTᵏ, gradients, optTᵏ)
    return lossval
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia maximinNN1 --nepochs 10
# julia> maximinNN1.main("--nepochs 10")
PROGRAM_FILE == "maximinNN1.jl" && main(ARGS)

end # module
