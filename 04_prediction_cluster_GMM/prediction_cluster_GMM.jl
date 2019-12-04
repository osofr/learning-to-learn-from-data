## to run, start julia in parent dir "./funcGrad.jl/Julia"
# julia> include("prediction_cluster_GMM.jl")
# julia> prediction_cluster_GMM.main("--nepochs 10 --gpu 1")
## alternatively can be called from command line:
# $ julia prediction_cluster_GMM --nepochs 10

# __precompile__()
module prediction_cluster_GMM
#
using Knet, ArgParse, JLD2, FileIO
using Base.Iterators: repeated, partition
using Clustering ## uncomment for k-means

include("./commonNNfuns.jl")
include("./param_model.jl")
include("./param_sim.jl")
include("./auc.jl")
include("./expectation_maximization.jl")

export main, loadNN, predictTᵏ

 cpu(x) = convert(Array{Float32}, x)
 gpu(x) = convert(KnetArray{Float32}, x)

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
        ("--nepochs"; arg_type=Int; default=50; help="# of training epochs")
        ("--niter"; arg_type=Int; default=10; help="# of training training steps within each epoch")
        ("--nT"; arg_type=Int; default=10; help="# of steps on T for every nT steps on Pi")
        ("--seed"; arg_type=Int; default=1054123; help="random seed")
        ("--parsrange"; nargs='*'; default = ["[[-3.0f0, 3.0f0],", "[-3.0f0, 3.0f0]]"]; arg_type=String; help="range of all parameters at once")
        ("--truepsi"; default="Ψμ"; help="function for evaluation of the true parameter value under P0")
        ("--name"; default="norm"; help="distribution name for P0")
        ("--optimPi"; default="Adam(lr=0.001, beta1=0.9)")
        ("--optimT"; default="Adam(lr=0.001, beta1=0.9)")
        ("--loadfile"; default=nothing; help="file to load trained models")
        ("--outdir"; default=nothing; help="output dir for models/generations")
        ("--saveEvery"; arg_type=Int; default=1; help="save network every saveEvery epochs")
        ("--hiddenPi"; nargs='*'; default = [8,16,16,8]; arg_type=Int; help="sizes of hidden layers for Piᵏ, defaults to --hidden 8 16 16 8 for a net with 4 hidden layers")
        ("--hiddenT"; nargs='*'; default = [20,20,10];  arg_type=Int; help="sizes of hidden layers for Tᵏ, defaults to --hidden 20 20 2 a net with 3 hidden layers")
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
    o[:parsrange] = reduce(*, o[:parsrange])
    o[:parsrange] = eval(parse(o[:parsrange]))

    ## define the model parameter space
    parbox = ParamBox([convert(atype, par) for par in o[:parsrange]])
    modelP = ParamModelPᵏ(parbox, Ψ=Symbol(o[:truepsi]); name = parse(o[:name]), xdim=o[:xdim])

    lossNormalizer = nothing

    if (o[:verbose]==1)
        println("Range for each parameter: ")
        for parrange in o[:parsrange] println(parrange); end
        println("hiddenPi, network for Piᵏ: ", o[:hiddenPi])
        println("hiddenT, network for Tᵏ: ",  o[:hiddenT])
        println("ParamBox:"); println(parbox)
    end

    if (o[:loadfile]==nothing)
        # def 3 NNs: mPiᵏ -- hardest prior; mPiᵣ -- hardest diffuse prior; mTᵏ -- best estimator
        if (o[:verbose]==1) println("initializing NNs"); end
        mPiᵏ = initPiᵏ(o[:hiddenPi]...; atype=atype, winit=0.9, udim = o[:udim], outdim = length(modelP));
        mPiᵣ = deepcopy(mPiᵏ)
        mTᵏ = initTᵏ(o[:hiddenT]...; atype=atype, winit=-0.5, xdim = o[:xdim], outdim = o[:xdim]);
    else
        # load 3 pre-saved NNs
        if (o[:verbose]==1) println("loading existing NN"); end
        mPiᵏ, mPiᵣ, mTᵏ = loadNN(atype, o[:loadfile])
    end
    nets = Dict(:mPiᵏ => mPiᵏ, :mPiᵣ => mPiᵣ, :mTᵏ => mTᵏ)

    ## lets make some noise!
    noise = sample_noise(atype,o[:udim],o[:nbatch]);
    ## sample parameters from the prior NN
    γsamp = predictPiᵏ(nets[:mPiᵣ],noise,modelP);

    println("γsamp[1]"); println(γsamp[1])
    println("γsamp[2]"); println(γsamp[2])

    if (o[:verbose]==1)
        println("init min μ1: $(minimum(γsamp[1])); init max μ1: $(maximum(γsamp[1]))")
        println("init min μ2: $(minimum(γsamp[2])); init max μ2: $(maximum(γsamp[2]))")
    end

    if o[:outdir] != nothing #&& !isdir(o[:outdir])
        mkpath(o[:outdir])
        mkpath(joinpath(o[:outdir],"models"))
        mkpath(joinpath(o[:outdir],"risks"))
    end
    ## training loop over epochs
    classerror, aucvec, Piloss, riskEpoch = trainepochs(o,nets,modelP,lossNormalizer,atype)
end

function trainepochs(o,nets,modelP,lossNormalizer,atype=Array{Float32})
    # maxRisk = Vector{Float32}(div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit])
    Piloss = Vector{Float32}(div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit])
    classerror = Vector{Vector{Float32}}(div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit])
    aucvec = Vector{Vector{Float32}}(div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit])
    classerrorU = Vector{Float32}(div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit])
    aucvecU = Vector{Float32}(div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit])
    riskEpoch = Vector{Int}(div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit])
    grid = linspace(modelP.bounds, step=o[:Rgridsize])
    riskmat = zeros(Float32, (length(grid[1]), length(grid[2]), div(o[:nepochs],o[:maxRiskEvery])+o[:maxRiskInit]))
    optPiᵏ = map(wi->eval(parse(o[:optimPi])), nets[:mPiᵏ]); # optPiᵏ = optimizers(nets[:mPiᵏ], Adam; lr=0.001, beta1=0.9);
    optTᵏ  = map(wi->eval(parse(o[:optimT])), nets[:mTᵏ]); # optTᵏ = optimizers(nets[:mTᵏ], Adam; lr=0.001, beta1=0.9);
    z1n = allocX1n(modelP, o[:nbatch]; atype=atype);
    z1ntest = allocX1n(modelP, o[:ntest]; atype=atype);
    println("training started..."); flush(STDOUT)

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
            dlossval += trainPiᵏ!(nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ],noise,z1n,atype,modelP,optPiᵏ,lossNormalizer,o)
            for _ in 1:o[:nT]
                noise = sample_noise(atype,o[:udim],o[:nbatch])
                glossval += trainTᵏ!(nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ],noise,z1n,atype,modelP,optTᵏ,lossNormalizer,o)
            end
        end
        dlossval /= o[:niter]; glossval /= (o[:niter]*10);

        ## evaluate both losses on new test data
        noise = sample_noise(atype,o[:udim],o[:ntest]);
        dTESTloss = Piᵏloss(nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ],noise,z1ntest,atype,modelP,lossNormalizer);
        gTESTloss = Tᵏloss(nets[:mTᵏ],nets[:mPiᵏ],nets[:mPiᵣ],z1ntest,noise,atype,modelP,lossNormalizer);
        ## γ wrt Piᵏ
        γsamp = predictPiᵏ(nets[:mPiᵏ],noise,modelP)
        x1nC1 = sim_norm!(γsamp[1], z1ntest)
        x1nC2 = sim_norm!(γsamp[2], z1ntest)
        ## γ wrt uniform prior
        γsampU = predictPiᵏUnifμσ(noise,atype,modelP)
        x1nC1U = sim_norm!(γsampU[1], z1ntest)
        x1nC2U = sim_norm!(γsampU[2], z1ntest)

        αsamp = drawα(atype, size(z1ntest,2))

        if ((o[:verbose]==1) & (epoch%o[:maxRiskEvery]==0))
            flush(STDOUT)
            println((:epoch,epoch,:dTESTloss,dTESTloss,:gTESTloss,gTESTloss))
            println((:epoch, epoch, :minμ1,  minimum(γsamp[1]), :maxμ1,  maximum(γsamp[1])))
            println((:epoch, epoch, :minμ2,  minimum(γsamp[2]), :maxμ2,  maximum(γsamp[2])))
            printPiᵏ(epoch, γsamp...)
            println("----------Tᵏ predictions for mixture at uniform α and γ~Piᵏ---------")
            classerrαU, aucαU, aucαU_EM = evalTᵏrisks(epoch,z1ntest,αsamp,γsamp,atype,x1nC1,x1nC2,nets)
            println("----------Tᵏ predictions for mixture at uniform α and γ~U---------")
            classerrαUγU, aucαUγU, aucαUγU_EM = evalTᵏrisks(epoch,z1ntest,αsamp,γsampU,atype,x1nC1U,x1nC2U,nets)

            println("------------------------------------------------------")
            println((:classerrαU, round(classerrαU,3)))
            println((:Tᵏauc_αU_γPiᵏ, round(aucαU,3), :EMaucαU_γPiᵏ, round(aucαU_EM,3)))
            println((:Tᵏauc_αU_γU, round(aucαUγU,3), :EMauc_αU_γU, round(aucαUγU_EM,3)))

            riskEpoch[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]] = epoch
            classerrorU[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]] = classerrαU
            aucvecU[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]] = aucαU
            Piloss[div(epoch,o[:maxRiskEvery])+o[:maxRiskInit]] = dTESTloss
            println("------------------------------------------------------")
        end

        ## save models and generations
        if ((o[:outdir] != nothing) & (epoch%o[:saveEvery]==0))
            filename = @sprintf("%04d.jld2",epoch)
            filepath = joinpath(o[:outdir],"models",filename)
            saveNN(filepath,nets[:mPiᵏ],nets[:mPiᵣ],nets[:mTᵏ])
        end
        if o[:outdir] != nothing
            filename = "risks.jld2"
            filepath = joinpath(o[:outdir],"risks",filename)
            save(filepath,
                "classerror",  classerror,
                "classerrorU", classerrorU,
                "aucvec",      aucvec,
                "aucvecU",     aucvecU,
                "Piloss",      Piloss,
                "riskEpoch",   riskEpoch)
        end
    end

    return classerror, aucvec, Piloss, riskEpoch
end

function initPiᵏ(h...; atype=Array{Float32}, winit=1.5, udim = 2, outdim = 2)
    w = Any[]
    # use udim = 28*28 for images
    for nextd in [h..., outdim] #
        push!(w, convert(atype, winit*randn(Float32,nextd,udim)))
        push!(w, convert(atype, randn(Float32,nextd,1)))
        udim = nextd
    end
    return w
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

# function predictPiᵏ(w,x,modelP; pdrop=0.05)
function predictPiᵏ(w,x,modelP; pdrop=0.00)
    x = mat(x)
    for i=1:2:length(w)-2
        x = Knet.relu.(w[i]*x .+ w[i+1]) # max(0,x) w/ dropout
    end
    x = sigm.(w[end-1]*x .+ w[end])
    x = [x[i,:] for i=1:length(modelP)]
    x = map(lintrans, x, lb(modelP), ub(modelP))
    return x
end

## uniform sample for [μ1,μ2]; uniform sample for [σ1,σ2];
function predictPiᵏUnifμσ(x,atype,modelP)
  function sim_unif(nbatch,lb,ub)
    x = (rand(eltype(x),nbatch) .* (ub.-lb).+lb)
    x = convert(atype, x)
    return x
  end
  lbb = lb(modelP)
  ubb = ub(modelP)
  nsim = [size(x)[2] for _ = 1:length(modelP)]
  x = map(sim_unif, nsim, lbb, ubb)
  return x
end

## uniform sample for [μ1,μ2]; constant [σ1,σ2];
function predictPiᵏUnifμConstσ(x,atype,modelP)
  function sim_unif(nbatch,lb,ub)
    x = (rand(eltype(x),nbatch) .* (ub.-lb).+lb)
    x = convert(atype, x)
    return x
  end
  lbb = lb(modelP)
  ubb = ub(modelP)
  nsim = size(x)[2]
  xμ = sim_unif(nsim, lbb[1], ubb[1])
  xμ = reshape(xμ, 1, nsim)
  xσ = repeat([w.σ],inner=(1,size(x)[2]))
  xσ = convert(atype, xσ)
  x = vcat(xμ,xσ)
  x = [x[i,:] for i=1:length(modelP)]
  return x
end

function predictTᵏ(w,x; pdrop=0.00)
    x = mat(x)
    for i=1:2:length(w)-2
        x = Knet.relu.(w[i]*x .+ w[i+1]) # max(0,x) w/ dropout
    end
    x = w[end-1]*x .+ w[end]
    return x
end

## loss for hardest distr. Piᵏ, negative Risk
function Piᵏloss(mPiᵏ,mPiᵣ,mTᵏ,noise,z1n,atype,modelP,lossNormalizer)
    # 1a. generate a bunch of candidates, vectors [μ1, σ1, μ2, σ2] that define 2 distributions: N(μ1,σ1) & N(μ2,1)
    γsamples = predictPiᵏ(mPiᵏ,noise,modelP)
    # 1b. generate the mixture probability vector from the uniform prior (α) (one draw of α~U(0,1) for each batch element)
    αsamples = drawα(atype, size(z1n,2))
    # 2a. Generate a sample Cᵢ ∈ {0,1} from mixture probability α, for each observation xᵢ (each row size n)
    CiU = rand(Float32,size(z1n))
    classCi = convert(atype, Float32.(CiU .> αsamples'))
    # 2b. Generate a random sample x1n~N(μ1,σ1) & x1n~N(μ2,σ2) (each row size n)
    x1n_μ1σ1 = sim_norm!(γsamples[1], z1n)
    x1n_μ2σ2 = sim_norm!(γsamples[2], z1n)
    # 2c. Pick only one sample based on the draw of mixture probability α
    x1n = (x1n_μ1σ1 .* (1 .- classCi)) .+ (x1n_μ2σ2 .* classCi)
    Tᵏ = predictTᵏ(mTᵏ, x1n)
    # 5. evaluate the classification loss as -MSE(Tᵏ(x1n),trueΨ) for each param in γsamples
    RiskTᵏPiᵏ = classloss(Tᵏ, classCi, γsamples[1], γsamples[2])
    # 6. evaluate the penalty for mode collapse wrt diffuse prior mPiᵣ
    # γsamples = predictPiᵏ(mPiᵣ,noise,modelP)
    γsamples = predictPiᵏUnifμσ(noise,atype,modelP)
    x1n_μ1σ1 = sim_norm!(γsamples[1], z1n)
    x1n_μ2σ2 = sim_norm!(γsamples[2], z1n)
    x1n = (x1n_μ1σ1 .* (1 .- classCi)) .+ (x1n_μ2σ2 .* classCi)
    Tᵏ = predictTᵏ(mTᵏ, x1n)
    RiskTᵏPiᵣ = classloss(Tᵏ, classCi, γsamples[1], γsamples[2])
    λpenalty = (relu(RiskTᵏPiᵣ-RiskTᵏPiᵏ))^2
    loss = -RiskTᵏPiᵏ + 100*λpenalty
    return loss
end
Piᵏlossgradient = gradloss(Piᵏloss)

## loss for the estimator Tᵏ
## same as Piᵏloss, but positive MSE and without the mode collapse penalty
function Tᵏloss(mTᵏ,mPiᵏ,mPiᵣ,z1n,noise,atype,modelP,lossNormalizer)
    # 1a. generate a bunch of candidates, vectors [μ1, σ1, μ2, σ2] that define 2 distributions: N(μ1,σ1) & N(μ2,1)
    γsamples = predictPiᵏ(mPiᵏ,noise,modelP)
    # 1b. generate the mixture probability vector from the uniform prior (α) (one draw of α~U(0,1) for each batch element)
    αsamples = drawα(atype, size(z1n,2))
    # 2a. Generate a sample Cᵢ ∈ {0,1} from mixture probability α, for each observation xᵢ (each row size n)
    CiU = rand(Float32,size(z1n))
    classCi = convert(atype, Float32.(CiU .> αsamples'))
    # 2b. Generate a random sample x1n~N(μ1,σ1) & x1n~N(μ2,σ2) (each row size n)
    x1n_μ1σ1 = sim_norm!(γsamples[1], z1n)
    x1n_μ2σ2 = sim_norm!(γsamples[2], z1n)
    # 2c. Pick only one sample based on the draw of mixture probability α
    x1n = (x1n_μ1σ1 .* (1 .- classCi)) .+ (x1n_μ2σ2 .* classCi)
    # 3. evaluate estimator Tᵏ(xn1) for each sample from γsamples
    Tᵏ = predictTᵏ(mTᵏ, x1n)
    lossTᵏ = classloss(Tᵏ, classCi, γsamples[1], γsamples[2])
    return lossTᵏ
end
Tᵏlossgradient = gradloss(Tᵏloss)

## draw mixture probabilities / weights for each batch
function drawα(atype,nbatch)
    αsamples = rand(Float32, nbatch)
    return αsamples
end

## normal
function sample_noise(atype,udim,nsamples,mu=0.5f0,sigma=0.5f0)
    noise = convert(atype, randn(Float32, udim, nsamples))
    noise = (noise .- mu) ./ sigma
    return noise
end

## Performs a single training step for hardest Piᵏ (mu, sigma)
## maximize risk R(Piᵏ,Tᵏ) for Piᵏ wrt input noise vector
function trainPiᵏ!(mPiᵏ,mPiᵣ,mTᵏ,noise,z1n,atype,modelP,optPiᵏ,lossNormalizer,o)
    gradients, lossval = Piᵏlossgradient(mPiᵏ,mPiᵣ,mTᵏ,noise,z1n,atype,modelP,lossNormalizer)
    update!(mPiᵏ, gradients, optPiᵏ)
    return lossval
end

## Performs a single training for the estimator Tᵏ(x1n)
## minimize risk R(Piᵏ,Tᵏ) for Tᵏ wrt input data x1n that derives from Piᵏ
function trainTᵏ!(mPiᵏ,mPiᵣ,mTᵏ,noise,z1n,atype,modelP,optTᵏ,lossNormalizer,o)
    gradients, lossval = Tᵏlossgradient(mTᵏ,mPiᵏ,mPiᵣ,z1n,noise,atype,modelP,lossNormalizer)
    update!(mTᵏ, gradients, optTᵏ)
    return lossval
end

## TX classification error
## min of 1) mean{  abs(TX - I(C=1)) }; 2) mean{ abs(TX - I(C=2) }.
function classerrorTᵏ(TX,TXclass,C₀)
    classerr1 = mean(abs.( TXclass .- (1 .- C₀)), 1)
    classerr2 = mean(abs.( TXclass .- C₀), 1)
    classerr = mean(min.(classerr1, classerr2))
    return classerr
end

function evalTᵏrisks(epoch,z1ntest,αsamp,γsamp,atype,x1nC1,x1nC2,nets)
    CiU = rand(Float32,size(z1ntest))
    classC₀ = convert(atype, Float32.(CiU .> αsamp'))
    x1n = (x1nC1 .* (1 .- classC₀)) .+ (x1nC2 .* classC₀)
    TX = getval(predictTᵏ(nets[:mTᵏ],x1n))
    lossTᵏ = classloss(TX, classC₀, γsamp[1], γsamp[2])
    TX = TX |> cpu
    TXclass = (TX .> 0.5f0)
    x1n = x1n |> cpu
    classC₀ = classC₀ |> cpu
    classerr = classerrorTᵏ(TX,TXclass,classC₀)

    predEM = map(i -> EM_Gaussians(x1n[:,i],2,1), 1:size(classC₀,2))
    predEM = hcat(map(x -> x[2][:,1], predEM)...)
    lossEM = classloss(predEM, classC₀, γsamp[1] |> cpu, γsamp[2] |> cpu)

    ## only evaluate AUC among draws that contain both classes
    aucTᵏ = 0.0f0
    aucEM = 0.0f0
    nclass = 0
    for i=1:size(classC₀,2)
        meanclass = mean(classC₀[:,i])
        if ((meanclass > 0.0) & (meanclass < 1.0))
            nclass += 1
            aucTᵏ_tmp = auc(classC₀[:,i], TX[:,i])
            aucTᵏ_tmp = max(aucTᵏ_tmp, 1-aucTᵏ_tmp)
            aucTᵏ += aucTᵏ_tmp
            aucEM_tmp = auc(classC₀[:,i], predEM[:,i])
            aucEM_tmp = max(aucEM_tmp, 1-aucEM_tmp)
            aucEM += aucEM_tmp
        end
    end
    aucTᵏ = aucTᵏ / nclass
    aucTᵏ = max(aucTᵏ, 1-aucTᵏ)

    aucEM = aucEM / nclass
    aucEM = max(aucEM, 1-aucEM)

    println((:epoch, epoch, :lossTᵏ, lossTᵏ, :lossEM, lossEM))
    println((:epoch, epoch, :aucTᵏ, round(aucTᵏ,3), :aucEM, round(aucEM,3)))

    return classerr, aucTᵏ, aucEM
end

## Evaluate Tᵏ risks (our loss and classification error) with respect to single (constant) γ=[μ1,μ2]
function γTᵏrisk(epoch,z1ntest,αsamp,γsamp,atype,x1nC1,x1nC2,nets)
    CiU = rand(Float32,size(z1ntest))
    classC₀ = convert(atype, Float32.(CiU .> αsamp'))
    x1n = (x1nC1 .* (1 .- classC₀)) .+ (x1nC2 .* classC₀)
    TX = getval(predictTᵏ(nets[:mTᵏ],x1n))
    TXclass = Float32.(TX .> 0.5f0)
    classerTᵏ = classerrorTᵏ(TX,TXclass,classC₀)
    lossTᵏ = classloss(TX, classC₀, γsamp[1], γsamp[2])

    TX = TX |> cpu
    x1n = x1n |> cpu
    classC₀ = classC₀ |> cpu

    return lossTᵏ, classerTᵏ
end

## Evaluate EM risks (our loss and classification error) with respect to single (constant) γ=[μ1,μ2]
function γEMrisk(epoch,z1ntest,αsamp,γsamp,atype,x1nC1,x1nC2)
    CiU = rand(Float32,size(z1ntest))
    classC₀ = convert(atype, Float32.(CiU .> αsamp'))
    x1n = (x1nC1 .* (1 .- classC₀)) .+ (x1nC2 .* classC₀)
    x1n = x1n |> cpu
    classC₀ = classC₀ |> cpu
    predEM = map(i -> EM_Gaussians(x1n[:,i],2,1), 1:size(classC₀,2))
    predEM = hcat(map(x -> x[2][:,1], predEM)...)
    lossEM = classloss(predEM, classC₀, γsamp[1], γsamp[2])
    EMclass = Float32.(predEM .> 0.5f0);
    classerEM = classerrorTᵏ(predEM,EMclass,classC₀)
    return lossEM, classerEM
end

## Evaluate K-means risk (our loss and classification error) with respect to single (constant) γ=[μ1,μ2]
## Using Julia package Clustering.jl K-means implementation:
#       https://clusteringjl.readthedocs.io/en/latest/kmeans.html
#       https://github.com/JuliaStats/Clustering.jl/blob/master/src/kmeans.jl
function γKmeansrisk(epoch,z1ntest,αsamp,γsamp,atype,x1nC1,x1nC2)
    CiU = rand(Float32,size(z1ntest))
    classC₀ = convert(atype, Float32.(CiU .> αsamp'))
    x1n = (x1nC1 .* (1 .- classC₀)) .+ (x1nC2 .* classC₀)
    x1n = x1n |> cpu
    classC₀ = classC₀ |> cpu
    predKM = map(i -> kmeans(reshape(x1n[:,i], 1, size(x1n,1)), 2; maxiter=200), 1:size(classC₀,2))
    predKM = hcat(map(x -> assignments(x) .- 1, predKM)...)
    lossKM = classloss(predKM, classC₀, γsamp[1], γsamp[2])
    KMclass = Float32.(predKM .> 0.5f0);
    classerKM = classerrorTᵏ(predKM,KMclass,classC₀)
    return lossKM, classerKM
end

## Evaluate random classifier risk (our loss and classification error) with respect to single (constant) γ=[μ1,μ2]
function randTrisk(z1ntest,αsamp,γsamp,x1nC1,x1nC2)
    CiU = rand(Float32,size(z1ntest))
    classC₀ = Float32.(CiU .> αsamp')
    x1n = (x1nC1 .* (1 .- classC₀)) .+ (x1nC2 .* classC₀)
    TX = rand(Float32,size(z1ntest))
    TXclass = Float32.(TX .> 0.5f0)
    lossTᵏ = classloss(TX, classC₀, γsamp[1], γsamp[2])
    classerTᵏ = classerrorTᵏ(TX,TXclass,classC₀)
    return lossTᵏ, classerTᵏ
end

function printTᵏ(epoch,TX,TXclass,C₀,classerr)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  ## mean class prediction for each dataset
  TxMean =  mean(TXclass,1)
  println((:epoch, epoch, :Tmeans_bybatch, quantile(TxMean[1,:], pquant)))
  println((:epoch, epoch, :T1, quantile(getval(TX[1,:]), pquant)))
  println((:epoch, epoch, :T2, quantile(getval(TX[2,:]), pquant)))
  println((:epoch, epoch, :T19, quantile(getval(TX[(size(TX,1)-1),:]), pquant)))
  println((:epoch, epoch, :T20, quantile(getval(TX[(size(TX,1)),:]), pquant)))
end

function printPiᵏ(epoch,μ1,μ2)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  println((:epoch, epoch, :Pμ1, quantile(getval(μ1) |> cpu, pquant)))
  println((:epoch, epoch, :Pμ2, quantile(getval(μ2) |> cpu, pquant)))
end

function printPiᵏ(epoch,μ1,σ1,μ2,σ2)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  println((:epoch, epoch, :Pμ1, quantile(getval(μ1) |> cpu, pquant)))
  println((:epoch, epoch, :Pμ2, quantile(getval(μ2) |> cpu, pquant)))
  println((:epoch, epoch, :Pσ1, quantile(getval(σ1) |> cpu, pquant)))
  println((:epoch, epoch, :Pσ2, quantile(getval(σ2) |> cpu, pquant)))
end

# This allows both non-interactive (shell command) and interactive calls like:
# $ julia prediction_cluster_GMM --nepochs 10
# julia> prediction_cluster_GMM.main("--nepochs 10")
PROGRAM_FILE == "prediction_cluster_GMM.jl" && main(ARGS)

end # module
