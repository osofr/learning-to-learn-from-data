
module finalRisk

using Knet, ArgParse, JLD2, FileIO
using Base.Iterators: repeated, partition
include("./commonNNfuns.jl")
include("./param_model.jl")
include("./param_truth.jl")
include("./param_sim.jl")
include("./SGAhardestPk.jl")
include("./riskgrid.jl")
include("./maximinNN1.jl")
export finalRiskMain

## ------------------------------------------------
## Parsing defaults
## ------------------------------------------------
function parse_options(args)
    s = ArgParseSettings()
    s.description =
        "Neural Nets with Bayesian maximin inference for [μ,σ] in bounded Gaussian / log-normal model."

    @add_arg_table s begin
        ("--gpu"; arg_type=Int; default=0; ; help="set to 1 to use gpu")
        ("--ntest"; arg_type=Int; default=10000; help="size of the test set")
        ("--xdim"; arg_type=Int; default=10; help="dimension of input data X -> T(X)")
        ("--seed"; arg_type=Int; default=54321; help="random seed")
        ("--parsrange"; nargs='*'; default = ["[[-2.5f0, 2.5f0],", "[1.0f0, 5.0f0]]"]; arg_type=String; help="range of all parameters at once")
        ("--truepsi"; default="Ψμ"; help="function for evaluation of the true parameter value under P0")
        ("--name"; default="norm"; help="distribution name for P0")
        ("--loaddir"; default=nothing; help="directory containing riskMats for a subset of trained models. Will look in loaddir/risks, identify the riskmat with the minimal maximal risk, and load the NN corresponding to this epoch.")
        ("--gridparsrange"; nargs='*'; default = ["[[-2.5f0, 2.5f0],", "[1.0f0, 5.0f0]]"]; arg_type=String; help="range of grid for all parameters at once")
        ("--Rgridsize"; default=0.1f0; arg_type=Float32; help="the size of the step in each parameter for risk surface evaluation")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    if o[:loaddir] != nothing
        o[:loaddir] = abspath(o[:loaddir])
    end
    return o
end

function finalRiskMain(args="")
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


	# Find the preliminary gridSearch with the lowest maximal risk and set o[:loadfile] equal to this file
	@load joinpath(o[:loaddir],"risks/maxRisk.jld2") maxRisk riskmat riskEpoch
	maxRiskPerEpoch = [maximum(riskmat[:,:,i]) for i in 1:size(riskmat,3)]
	minMaxRiskEpoch = riskEpoch[indmax(-maxRiskPerEpoch)]
	loadfile = joinpath(o[:loaddir],"models",@sprintf("%04d.jld2",minMaxRiskEpoch))

    # load 3 pre-saved NNs
    mPiᵏ, mPiᵣ, mTᵏ = loadNN(atype, loadfile)

    # Grid to search over
    o[:gridparsrange] = reduce(*, o[:gridparsrange])
    o[:gridparsrange] = eval(parse(o[:gridparsrange]))
    parbox_prime = ParamBox([convert(atype, par) for par in o[:gridparsrange]])
    modelP_prime = ParamModelPᵏ(parbox_prime, Ψ=Symbol(o[:truepsi]); name = parse(o[:name]), xdim=o[:xdim])
    grid = linspace(modelP_prime.bounds, step=o[:Rgridsize])

    # Search over this grid
    finalRiskMat = riskgrid(o,Dict(:mTᵏ => mTᵏ),modelP,grid[1],grid[2])
    finalMaxRisk = maximum(finalRiskMat)

    filename = "finalMinMaxRisk.jld2"
    filepath = joinpath(o[:loaddir],"risks",filename)
    save(filepath,
        "finalMaxRisk", finalMaxRisk,
        "finalRiskMat", finalRiskMat,
        "minMaxRiskEpoch", minMaxRiskEpoch)

    return finalMaxRisk, finalRiskMat, minMaxRiskEpoch
end


function convertNN(w, atype=Array{Float32})
    w0 = map(wi->convert(atype, wi), w)
    w1 = convert(Array{Any}, w0)
end

function loadNN(atype,loadfile=nothing)
    @load loadfile Pi PiDiffuse T
    mPiᵏ = convertNN(Pi, atype)
    mPiᵣ = convertNN(PiDiffuse, atype)
    mTᵏ  = convertNN(T, atype)
    return mPiᵏ, mPiᵣ, mTᵏ
end

function predictTᵏ(w,x; pdrop=0.00)
    x = mat(x)
    for i=1:2:length(w)-2
        x = Knet.relu.(w[i]*dropout(x, pdrop) .+ w[i+1])
    end
    x = w[end-1]*x .+ w[end]
    return x
end


end #module