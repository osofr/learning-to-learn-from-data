## to run, start julia in parent dir "./funcGrad.jl/Julia"
# julia> include("maximinNN1.jl")
# julia> maximinNN1.main("--nepochs 10 --gpu 1 --fast 1")
## alternatively can be called from command line:
# $ julia maximinNN1 --nepochs 10

# __precompile__()
module maximinNN1
#
using Knet, ArgParse, JLD2, FileIO
using Base.Iterators: repeated, partition
using CSV
include("./commonNNfuns.jl")
include("./param_model.jl")
include("./param_truth.jl")
include("./param_sim.jl")
include("./SGAhardestPk.jl")
include("./riskgrid.jl")
export main, loadNN, predictTᵏ

## ------------------------------------------------
## Parsing defaults
## ------------------------------------------------
function parse_options(args)
    s = ArgParseSettings()
    s.description =
        "Neural Nets with Bayesian maximin inference for [μ,σ] in bounded Gaussian / log-normal model."

    @add_arg_table s begin
        ("--gpu"; arg_type=Int; default=0; ; help="set to 1 to use gpu")
        ("--nbatch"; arg_type=Int; default=1000; help="batch size")
        ("--ntest"; arg_type=Int; default=10000; help="size of the test set")
        ("--udim"; arg_type=Int; default=2; help="dimension of noise U -> P(U)")
        ("--xdim"; arg_type=Int; default=10; help="dimension of input data X -> T(X)")
        ("--nepochs"; arg_type=Int; default=50; help="# of training iterations (note that nepochs is a misnomer, really these are just number of iterations performed)")
        ("--niter"; arg_type=Int; default=10; help="# of training training steps within each epoch (within each iteration -- epochs is a misnomer)")
        ("--nT"; arg_type=Int; default=10; help="# of steps on T for every nT steps on Pi")
        ("--seed"; arg_type=Int; default=54321; help="random seed")
        ("--numPars"; arg_type=Int; default = 1; help = "number of parameters in model")
        ("--truepsi"; default="Ψμ"; help="function for evaluation of the true parameter value under P0")
        ("--name"; default="norm"; help="distribution name for P0")
        ("--optimPi"; default="Adam(lr=0.001, beta1=0.9)")
        ("--optimT"; default="Adam(lr=0.001, beta1=0.9)")
        ("--loadfile"; default=nothing; help="file to load trained models")
        ("--outdir"; default=nothing; help="output dir for models/generations")
        ("--saveEvery"; arg_type=Int; default=1; help="save network every saveEvery epochs")
        ("--hiddenPi"; nargs='*'; default = [8,16,16,8]; arg_type=Int; help="sizes of hidden layers for Piᵏ, defaults to --hidden 8 16 16 8 for a net with 4 hidden layers")
        ("--hiddenT"; nargs='*'; default = [20,20,2];  arg_type=Int; help="sizes of hidden layers for Tᵏ, defaults to --hidden 20 20 2 a net with 3 hidden layers")
        ("--maxRiskEvery"; arg_type=Int; default=1; help="report maximum risk of current estimator every maxRiskEvery epochs, using SGA for hardest risk")
        ("--SGAnruns"; default=300; arg_type=Int; help="SGA number of gradient updates (for each random start)")
        ("--SGAnstarts"; default=10; arg_type=Int; help="SGA number of independent random starts of parameters γ")
        ("--SGAnbatch"; default=100; arg_type=Int; help="SGA batch size")
        ("--Rgrid"; default=0; arg_type=Int; help="evaluate the risk surface at resolution `Rgridsize` each time SGA runs")
        ("--Rgridsize"; default=0.1f0; arg_type=Float32; help="the size of the step in each parameter for risk surface evaluation")
        ("--verbose"; default=1; arg_type=Int; help="print diagnostics, e.g. the range of the prior and of the estimator")
        ("--maxRiskInit"; default=1; arg_type=Int; help="compute the max risk of the initial network? 0=No, 1=Yes")
        ("--plot"; default=0; help="real slow plotting of risk surfaces at each epoch")
        ("--lossNormalizer"; default=nothing;  arg_type=String; help="parameter to normalize the loss function by. Parameter in the same format as for truepsi. Useful if want to use an information normalized loss function. If equal to 1 then no normalization is performed. NOTE: not implemented for Rgrid or SGA, only for training")
        ("--l2constraint"; default=1f0; arg_type=Float32; help="l2 equality constraint on model parameters. To induce an inequality constraint, add one dummy parameter to the dimension of the model.")
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

    ## parse the parameter range
    parsrange = [[-o[:l2constraint],o[:l2constraint]] for i in 1:o[:numPars]]
    # parsrange = reduce(*, parsrange)
    # parsrange = eval(parse(parsrange))

    ## define the model parameter space
    parbox = ParamBox([convert(atype, par) for par in parsrange])
    modelP = ParamModelPᵏ(parbox, Ψ=Symbol(o[:truepsi]); name = parse(o[:name]), xdim=o[:xdim], l2constraint=o[:l2constraint])
    # modelP = ParamModelPᵏ(ParamBox([[0.0f0, 5.0f0], [1.0f0, 3.0f0]]), Ψ=:Ψμ; name=:norm)

    # If o[:lossNormalizer] is nonempty, define a function lossNormalizer
    if o[:lossNormalizer]!=nothing
        lNSymbol = Symbol(o[:lossNormalizer])
        lossNormalizer = function(modelP::ParamModelPᵏ, paramγ::Any)
            getfield(maximinNN1, lNSymbol)(paramγ)
        end
    else
        lossNormalizer = nothing
    end

    if (o[:verbose]==1)
        println("Range for each parameter: ")
        for parrange in parsrange println(parrange); end
        println("hiddenPi, network for Piᵏ: ", o[:hiddenPi])
        println("hiddenT, network for Tᵏ: ",  o[:hiddenT])
        println("ParamBox:"); println(parbox)
        println("Using Ψ: $(Symbol(o[:truepsi]))")
    end

    if (o[:loadfile]==nothing)
        # def 3 NNs: mPiᵏ -- hardest prior; mPiᵣ -- hardest diffuse prior; mTᵏ -- best estimator
        if (o[:verbose]==1) println("initializing NNs"); end
        mPiᵏ = initPiᵏ(o[:hiddenPi]...; atype=atype, winit=1.5, udim = o[:udim], outdim = length(modelP));
        mTᵏ = initTᵏ(o[:hiddenT]...; atype=atype, winit=-0.5, xdim = o[:xdim], outdim = 1);
        # mPiᵏ = pretrainPiᵏ(mPiᵏ,modelP,mTᵏ,o,atype)
        mPiᵣ = deepcopy(mPiᵏ)
    else
        # load 3 pre-saved NNs
        if (o[:verbose]==1) println("loading existing NN"); end
        mPiᵏ, mPiᵣ, mTᵏ = loadNN(atype, o[:loadfile])
    end
    nets = Dict(:mPiᵏ => mPiᵏ, :mPiᵣ => mPiᵣ, :mTᵏ => mTᵏ)

    noise = sample_noise(atype,o[:udim],o[:nbatch]);

    if o[:outdir] != nothing 
        mkpath(o[:outdir])
        mkpath(joinpath(o[:outdir],"models"))
        mkpath(joinpath(o[:outdir],"risks"))
    end
    ## training loop over epochs
    maxRisk, riskmat, riskEpoch = trainepochs(o,nets,modelP,lossNormalizer,atype)
end

function computeSGA(o,nets,modelP,atype=Array{Float32})
    modelTᵏ(z1n) = predictTᵏ(nets[:mTᵏ],z1n)
    maxRisk, hardestγ = findPᵏ(modelTᵏ,modelP;atype=atype,nruns=o[:SGAnruns],nstarts=o[:SGAnstarts],nbatch=o[:SGAnbatch]);
    return maxRisk, hardestγ
end

function trainepochs(o,nets,modelP,lossNormalizer,atype=Array{Float32})
    maxRisk = Vector{Float32}(div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit])
    riskEpoch = Vector{Int}(div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit])
    grid = linspace(modelP.bounds, step=o[:Rgridsize])
    riskmat = zeros(Float32, (length(grid[1]), length(grid[2]), div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit]))
    optPiᵏ = map(wi->eval(parse(o[:optimPi])), nets[:mPiᵏ]); 
    optTᵏ  = map(wi->eval(parse(o[:optimT])), nets[:mTᵏ]); 
    z1n = allocX1n(modelP, o[:nbatch]; atype=atype);
    z1ntest = allocX1n(modelP, o[:ntest]; atype=atype);
    println("training started..."); flush(STDOUT)

    if (o[:maxRiskInit]==1)
        riskEpoch[1] = 0
        maxRisk[1], hardestγ = computeSGA(o,nets,modelP,atype)
        println((:epoch,0,:trueMaxRisk,maxRisk[1],:hardestγ,hardestγ));flush(STDOUT)

        noise = sample_noise(atype,o[:udim],o[:nbatch])
        γsamp = predictPiᵏ(nets[:mPiᵏ],noise,modelP,nets[:mTᵏ],z1ntest)
        for i = 1:length(modelP)
            println((:param, i, :quant,  round.(quantile(convert(Array{Float32},abs.(γsamp[i])),[0.0,0.1,0.25,0.5,0.75,0.9,1.0]),2)))
        end
        println((:cor,cor(convert(Array{Float32},γsamp[2]),convert(Array{Float32},γsamp[3]))))

        if o[:Rgrid]==1
            println("computing risk surface with grid size $(o[:Rgridsize])");flush(STDOUT)
            riskmat[:,:,1] = riskgrid(o,nets,modelP,grid[1],grid[2])
            println("max riskmat: $(maximum(riskmat[:,:,1]))");flush(STDOUT)
        end
    end
    ## save init models
    if o[:outdir] != nothing
        filename = @sprintf("%04d.jld2",0)
        filepath = joinpath(o[:outdir],"models",filename)
        saveNN(filepath,nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ])
    end

    for epoch = 1:o[:nepochs]
        dlossval = glossval = 0
        for _ in 1:o[:niter]
            noise = sample_noise(atype,o[:udim],o[:nbatch])
            dlossval += trainPiᵏ!(nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ],noise,z1n,modelP,optPiᵏ,lossNormalizer,o)
            for _ in 1:o[:nT]
                noise = sample_noise(atype,o[:udim],o[:nbatch])
                glossval += trainTᵏ!(nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ],noise,z1n,modelP,optTᵏ,lossNormalizer,o)
            end
        end
        dlossval /= o[:niter]; glossval /= (o[:niter]*10);

        ## evaluate both losses on new test data
        noise = sample_noise(atype,o[:udim],o[:ntest]);
        dTESTloss = Piᵏloss(nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ],noise,z1ntest,modelP,lossNormalizer);
        gTESTloss = Tᵏloss(nets[:mTᵏ],nets[:mPiᵏ],nets[:mPiᵣ],z1ntest,noise,modelP,lossNormalizer);

        
        if (o[:verbose]==1)
            γsamp = predictPiᵏ(nets[:mPiᵏ],noise,modelP,nets[:mTᵏ],z1ntest)
            predT = predictTᵏ(nets[:mTᵏ],z1ntest);
            println((:epoch,epoch,:dTESTloss,dTESTloss,:gTESTloss,gTESTloss))
            flush(STDOUT)
            println((:epoch, epoch, :minμ,  minimum(γsamp[1]), :maxμ,  maximum(γsamp[1])))
            println((:epoch, epoch, :minσ,  minimum(γsamp[2]), :maxσ,  maximum(γsamp[2])))
            println((:epoch, epoch, :minTᵏ, minimum(predT),    :maxTᵏ, maximum(predT)))
        end

        if (epoch%o[:maxRiskEvery]==0)
            riskEpoch[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]] = epoch
            maxRisk[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]], hardestγ = computeSGA(o,nets,modelP,atype)
            println((:epoch,epoch,:trueMaxRisk,maxRisk[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]],:hardestγ,hardestγ));flush(STDOUT)
            # Doing this twice to get a sense of variability
            maxRisk[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]], hardestγ = computeSGA(o,nets,modelP,atype)
            println((:epoch,epoch,:trueMaxRisk,maxRisk[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]],:hardestγ,hardestγ));flush(STDOUT)
            if o[:Rgrid]==1
                println("computing risk surface with grid size $(o[:Rgridsize])")
                riskmat[:,:,div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]] = riskgrid(o,nets,modelP,grid[1],grid[2])
                println("max riskmat: $(maximum(riskmat[:,:,div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]]))")
            end
            
            γsamp = predictPiᵏ(nets[:mPiᵏ],noise,modelP,nets[:mTᵏ],z1ntest)
            for i = 1:length(modelP)
                println((:param, i, :quant,  round.(quantile(convert(Array{Float32},abs.(γsamp[i])),[0.0,0.1,0.25,0.5,0.75,0.9,1.0]),2)))
            end
            println((:cor,cor(convert(Array{Float32},γsamp[2]),convert(Array{Float32},γsamp[3]))))
        end

        ## save models and generations
        if ((o[:outdir] != nothing) & (epoch%o[:saveEvery]==0))
            filename = @sprintf("%04d.jld2",epoch)
            filepath = joinpath(o[:outdir],"models",filename)
            saveNN(filepath,nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ])
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
        push!(w, convert(atype, randn(Float32,nextd,1)))
        udim = nextd
    end
    return w
end

function pretrainPiᵏ(mPiᵏ,modelP,mTᵏ,o,atype;numUpdate = 10000, verbose = false)
    noise = sample_noise(atype,o[:udim],o[:nbatch]);
    z1n = allocX1n(modelP, o[:nbatch]; atype=atype);

    PiᵏInitLoss = function(w)
        preds = predictPiᵏ(w,noise,modelP,mTᵏ,z1n)
        -sum((sum.(abs2,preds)./length(preds[1]) .- abs2.(sum.(preds)./length(preds[1]))))
    end
    PiᵏInitLossGradient = gradloss(PiᵏInitLoss)

    optPiᵏinit = map(wi->eval(parse(o[:optimPi])), mPiᵏ);
    for i in 1:numUpdate
        gradients, lossval = PiᵏInitLossGradient(mPiᵏ)
        update!(mPiᵏ, gradients, optPiᵏinit)

        if verbose
            println("---------------")
            println((:initIter,i,:loss,convert(Float32,lossval)))
            γsamp = predictPiᵏ(mPiᵏ,noise,modelP,mTᵏ,z1n)
            for i = 1:length(modelP)
                    println((:param, i, :quant,  round.(quantile(convert(Array{Float32},abs.(γsamp[i])),[0.0,0.25,0.5,0.75,1.0]),2)))
            end
        end
    end

    return mPiᵏ
end

function initTᵏ(h...; atype=Array{Float32}, winit=1.5, xdim = 10, outdim = 1)
    w = Any[]
    for nextd in [h..., outdim]
        push!(w, convert(atype, winit*randn(Float32,nextd,xdim)))
        push!(w, convert(atype, zeros(Float32,nextd,1)))
        xdim = nextd
    end
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

function gencornersCx(mTᵏ,noise,z1n,modelP)
  Piᵏin = [convert(typeof(z1n),hcat([mid(modelP)]...))[:,1]]

  return vcat(map(γ -> (predictTᵏ(mTᵏ, sim!(modelP, γ, z1n)).-Ψ(modelP, γ)), Piᵏin)...)
end

function predictPiᵏ(w,x,modelP,mTᵏ,z1n; pdrop=0.05)
    x = mat(x)

    for i=1:2:length(w)-2
        x = Knet.relu.(w[i]*dropout(x, pdrop) .+ w[i+1]) # max(0,x) w/ dropout
    end

    x = w[end-1]*x .+ w[end]
    std_by = sqrt.(sum(x.*x,1)[1,:])
    x = [(modelP.l2constraint).*x[i,:]./std_by for i=1:length(modelP)]

    return x
end

function predictTᵏ(w,x; pdrop=0.00)
    x = mat(x)
    for i=1:2:length(w)-2
        x = Knet.relu.(w[i]*dropout(x, pdrop) .+ w[i+1]) # max(0,x) w/ dropout
    end
    x = w[end-1]*x .+ w[end]
    return x
end

## loss for hardest distr. Piᵏ, negative Risk
function Piᵏloss(mPiᵏ,mPiᵣ,mTᵏ,noise,z1n,modelP,lossNormalizer)
    # 1. generate a bunch of candidates, vectors [mu, sigma],
    #    in response to a batch-size noise vector
    γsamples = predictPiᵏ(mPiᵏ,noise,modelP,mTᵏ,z1n) # γsamples = mPiᵏ(noise)
    # 2. generate a random sample x1n (each row size n)
    x1n = sim!(modelP, γsamples, z1n)
    # 3. evaluate estimator Tᵏ(xn1) for each sample from γsamples
    Tᵏ = predictTᵏ(mTᵏ, x1n)
    # 4. evaluate the truth (separate true psi0 for each param in γsamples)
    trueΨ = Ψ(modelP, γsamples)
    # 5. evaluate the loss as -MSE(Tᵏ(x1n),trueΨ) for each param in γsamples
    if lossNormalizer!=nothing
        weight = lossNormalizer(modelP, γsamples)
        RiskTᵏPiᵏ = weightedMse(Tᵏ, reshape(trueΨ, 1,length(trueΨ)),reshape(weight, 1,length(weight)))
    else
        RiskTᵏPiᵏ = mse(Tᵏ, reshape(trueΨ, 1,length(trueΨ)))
    end
    # 6. evaluate the penalty for mode collapse wrt diffuse prior
    γsamplesᵣ = predictPiᵏ(mPiᵣ,noise,modelP,mTᵏ,z1n)
    x1n = sim!(modelP, γsamplesᵣ, z1n)
    Tᵏ = predictTᵏ(mTᵏ, x1n)
    trueΨ = Ψ(modelP, γsamplesᵣ)
    if lossNormalizer!=nothing
        weight = lossNormalizer(modelP, γsamplesᵣ)
        RiskTᵏPiᵣ = weightedMse(Tᵏ, reshape(trueΨ, 1,length(trueΨ)),reshape(weight, 1,length(weight)))
    else
        RiskTᵏPiᵣ = mse(Tᵏ, reshape(trueΨ, 1,length(trueΨ)))
    end
    λpenalty = (relu(RiskTᵏPiᵣ-RiskTᵏPiᵏ))^2
    loss = -RiskTᵏPiᵏ + 75*λpenalty

    return loss
end
Piᵏlossgradient = gradloss(Piᵏloss)

## loss for the estimator Tᵏ
## same as Piᵏloss, but positive MSE and without the mode collapse penalty
function Tᵏloss(mTᵏ,mPiᵏ,mPiᵣ,z1n,noise,modelP,lossNormalizer)
    # 1. update the best guess for hardest Piᵏ
    γsamples = predictPiᵏ(mPiᵏ,noise,modelP,mTᵏ,z1n)
    # 2. generate batch of observed data under new Piᵏ
    x1n = sim!(modelP, γsamples, z1n)
    # 3. evaluate the estimator under new data
    Tᵏ = predictTᵏ(mTᵏ, x1n)
    # 4. evaluate the truth and loss (MSE(Tᵏ,))
    trueΨ = Ψ(modelP, γsamples)
    # 5. evaluate the loss as MSE(Tᵏ(x1n),trueΨ) for each param in γsamples
    if lossNormalizer!=nothing
        weight = lossNormalizer(modelP, γsamples)
        loss = weightedMse(Tᵏ,reshape(trueΨ, 1,length(trueΨ)),reshape(weight, 1,length(weight)))
    else
        loss = mse(Tᵏ,reshape(trueΨ, 1,length(trueΨ)))
    end
    return loss
end
Tᵏlossgradient = gradloss(Tᵏloss)

## normal
function sample_noise(atype,udim,nsamples,mu=0.5f0,sigma=0.5f0)
    noise = convert(atype, randn(Float32, udim, nsamples))
    noise = (noise .- mu) ./ sigma
    return noise
end

## Performs a single training step for hardest Piᵏ (mu, sigma)
## maximize risk R(Piᵏ,Tᵏ) for Piᵏ wrt input noise vector
function trainPiᵏ!(mPiᵏ,mPiᵣ,mTᵏ,noise,z1n,modelP,optPiᵏ,lossNormalizer,o)
    gradients, lossval = Piᵏlossgradient(mPiᵏ,mPiᵣ,mTᵏ,noise,z1n,modelP,lossNormalizer)
    update!(mPiᵏ, gradients, optPiᵏ)
    return lossval
end

## Performs a single training for the estimator Tᵏ(x1n)
## minimize risk R(Piᵏ,Tᵏ) for Tᵏ wrt input data x1n that derives from Piᵏ
function trainTᵏ!(mPiᵏ,mPiᵣ,mTᵏ,noise,z1n,modelP,optTᵏ,lossNormalizer,o)
    gradients, lossval = Tᵏlossgradient(mTᵏ,mPiᵏ,mPiᵣ,z1n,noise,modelP,lossNormalizer)
    update!(mTᵏ, gradients, optTᵏ)
    return lossval
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia maximinNN1 --nepochs 10
# julia> maximinNN1.main("--nepochs 10")
PROGRAM_FILE == "maximinNN1.jl" && main(ARGS)

end # module
