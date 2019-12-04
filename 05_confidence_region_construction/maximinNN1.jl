
## ************************************************************************
## Main module implementing the example "Confidence Region Construction"
## ************************************************************************

"""
julia maximinNN1.jl -h # to see all script options
julia maximinNN1.jl --outdir ./globCIn50-out # to specify output dir
This example implements a global Minimax Confindence Intervals for Normal(μ,σ) model
"""

# __precompile__()
module maximinNN1

# using Printf # for julia 0.7
# using Statistics # for julia 0.7
# using BSON: @save, @load # for julia 0.7

using Flux
using ArgParse, JLD2, FileIO, BSON
using Distributions
using Base.Iterators: repeated, partition
using Flux: relu, σ, param
using Flux.Tracker: data, grad, istracked, zero_grad!
# using Knet: mat, dropout # using Knet
using Knet: dropout
# conditionally load CuArrays
try
  using CuArrays;
catch
    println("No GPU, CuArrays isn't loaded")
end
using Flux
# using Flux: param #, LSTM
import Flux.LSTM
import Flux.Dense
import Flux.conv
import Flux.params
import Flux.gpu
import Flux.cpu
import Flux.Conv

export main, trainepochs

include("./commonNNfuns.jl")
include("./param_model.jl")
include("./param_truth.jl")
include("./param_CImodel.jl")
include("./param_sim.jl")
include("./SGAhardestPk_FLUX.jl")
include("./RiskCIcoverage.jl")
include("./RiskCIsize.jl")
include("./RiskCIcent.jl")
include("./sim_norm_CIs.jl")
include("./riskgrid.jl")
include("./printTkfuns.jl")
include("./adamstown.jl")


initnf0(dims...) = randn(Float32, dims...) / 100
glorot_uniformf0(dims...) = (rand(Float32, dims...) - 0.5f0)*Float32(sqrt(24.0f0/(sum(dims))))
glorot_normalf0(dims...) = randn(Float32, dims...)*Float32(sqrt(2.0f0/sum(dims)))
function LSTMCell(in::Integer, out::Integer; init = glorot_uniformf0, initn = initnf0)
  cell = Flux.LSTMCell(param(1.5f0*init(out*4, in)),
                       param(1.5f0*init(out*4, out)),
                       param(zeros(Float32, out*4)),
                       param(initn(out)),
                       param(initn(out)))
  cell.b.data[Flux.gate(out, 2)] = 1
  return cell
end
LSTM(a...; ka...) = Flux.Recur(LSTMCell(a...; ka...))
function Dense(in::Integer, out::Integer, σ = identity;
               initW = glorot_uniformf0, initb = zeros)
  return Flux.Dense(param(2.5f0*initW(out, in)), param(initb(Float32, out)), σ)
end

Conv(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity; init = initnf0,
     stride::NTuple{N,Integer} = map(_->1,k),
     pad::NTuple{N,Integer} = map(_->0,k)) where N =
  Flux.Conv(param(init(k..., ch...)), param(zeros(Float32, ch[2])), σ,
       stride = stride, pad = pad)

struct IdentitySkip
   inner
end
(m::IdentitySkip)(x) = m.inner(x) .+ x
Flux.treelike(IdentitySkip)

params(x::Array)=x

struct UniformPi⁰
end
struct UniformInversePi⁰
end
struct ConstPi⁰
  μ
  σ
end
struct UniformμConstσPi⁰
  σ
end
struct SGAPᵏ
end
struct offsetUniform
end
struct GroupedPᵏ
  cent
  off
end

Flux.treelike(GroupedPᵏ)

struct CX_MLE
end

## Tᵏ model with a single reduction MLP or fixed reduction function
struct Tᵏ{T}
  mTᵏ
  mTᵏreduce
  nparams::Int
  lbb::Array{T,1}
  ubb::Array{T,1}
  trtype::Symbol
  NNtype::Symbol
end
Tᵏ(mTᵏ, mTᵏreduce, nparams::Int, lbb::Array{T,1}, ubb::Array{T,1}; trtype = :linear, NNtype = :LSTM) where T =
  Tᵏ(mTᵏ, mTᵏreduce, nparams, lbb, ubb, trtype, NNtype)
Tᵏ(mTᵏ, mTᵏreduce, modelP::ParamModelPᵏ, paramCI::ParamCI) =
  Tᵏ(mTᵏ, mTᵏreduce, paramCI.nparams, lb(paramCI), ub(paramCI); trtype = paramCI.trtype, NNtype = modelP.mTmodel)
Flux.treelike(Tᵏ)
function (c::Tᵏ)(x)
  predictTᵏ(c.mTᵏ,c.mTᵏreduce,x,c.nparams,c.lbb,c.ubb;trtype=c.trtype,NNtype=c.NNtype)
end
gpu(mTᵏ::Tᵏ) = Tᵏ(mTᵏ.mTᵏ |> gpu, mTᵏ.mTᵏreduce |> gpu, mTᵏ.nparams, mTᵏ.lbb, mTᵏ.ubb, mTᵏ.trtype, mTᵏ.NNtype)

## Tᵏ model with 1 LSTM for centers (C(X)); 1 LSTM for lower and upper offsets (L(X),U(X))
struct Tᵏ2x{T}
  mTᵏcent
  mTᵏoff
  mTᵏreduce1
  mTᵏreduce2
  nparams::Int
  lbb::Array{T,1}
  ubb::Array{T,1}
  δ::Array{T,1}
  trtype::Symbol
  NNtype::Symbol
end
Tᵏ2x(mTᵏcent, mTᵏoff, mTᵏreduce1, mTᵏreduce2, nparams::Int, lbb::Array{T,1}, ubb::Array{T,1}, δ::Array{T,1}; trtype = :linear, NNtype = :LSTM) where T =
  Tᵏ2x(mTᵏcent, mTᵏoff, mTᵏreduce1, mTᵏreduce2, nparams, lbb, ubb, δ, trtype, NNtype)
Tᵏ2x(mTᵏcent, mTᵏoff, mTᵏreduce1, mTᵏreduce2, modelP::ParamModelPᵏ, paramCI::ParamCI) =
  Tᵏ2x(mTᵏcent, mTᵏoff, mTᵏreduce1, mTᵏreduce2, paramCI.nparams, lb(paramCI), ub(paramCI), paramCI.δ; trtype = paramCI.trtype, NNtype = modelP.mTmodel)

Flux.treelike(Tᵏ2x)

function (c::Tᵏ2x)(x,η)
  predictTᵏ(c.mTᵏcent,c.mTᵏoff,c.mTᵏreduce1,c.mTᵏreduce2,x,η,c.nparams,c.lbb,c.ubb,c.δ;trtype=c.trtype,NNtype=c.NNtype)
end

gpu(mTᵏ::Tᵏ2x) = Tᵏ2x(mTᵏ.mTᵏcent |> gpu, mTᵏ.mTᵏoff |> gpu, mTᵏ.mTᵏreduce1 |> gpu, mTᵏ.mTᵏreduce2 |> gpu, mTᵏ.nparams, mTᵏ.lbb, mTᵏ.ubb, mTᵏ.δ, mTᵏ.trtype, mTᵏ.NNtype)

## ------------------------------------------------
## Parsing defaults
## ------------------------------------------------
function parse_options(args)
  s = ArgParseSettings()
  s.description =
      "Neural Nets with Bayesian maximin inference for [μ,σ] in bounded Gaussian / log-normal model."
  @add_arg_table s begin
    ("--nepochs1Cx"; arg_type=Int; default=25; help="# of pre-training iterations for centers against fixed uniform prior")
    ("--nepochs2Cx"; arg_type=Int; default=25; help="# of pre-training iterations for centers")
    ("--nepochsOff"; arg_type=Int; default=50; help="# of training iterations for offsets")
    ("--iepoch"; arg_type=Int; default=0; help="# of times training models have been reloaded so far (use default 0 if not reloading)")
    ("--reloadEvery"; arg_type=Int; default=0; help="how often should the training process be reloaded (every # of traiing epochs; use default 0 if not reloading)")
    ("--gpu"; arg_type=Int; default=0; ; help="set to 1 to use gpu")
    ("--nbatch"; arg_type=Int; default=500; help="batch size")
    ("--SGAnbatch"; default=500; arg_type=Int; help="SGA batch size")
    ("--SGAnruns"; default=50; arg_type=Int; help="SGA number of gradient updates (for each random start)")
    ("--SGAnstarts1"; default=10; arg_type=Int; help="SGA number of random starts for γ for epochs 1-1000")
    ("--SGAnstarts2"; default=10; arg_type=Int; help="SGA number of random starts for γ for epochs 'SGAramp_maxepoch' and after")
    ("--SGAramp_maxepoch"; default=1000; arg_type=Int; help="epoch at which SGA number of random starts for γ equals 'SGAnstarts2'")
    ("--kcent"; default=2.0f0; arg_type=Float32; help="max exponent for sigma^(kmax) scaler for CI center risk for Tᵏ, Piᵏ and Pᵏ")
    ("--losstypecent"; default=:L2; arg_type=Symbol; help="Type of loss for the centers, can be either L2 or MAD")
    ("--udim"; arg_type=Int; default=2; help="dimension of noise U -> P(U)")
    ("--xdim"; arg_type=Int; default=10; help="dimension of input data X -> T(X)")
    ("--nstepsTcent"; arg_type=Int; default=1; help="# of steps on Tᵏ for CI size, within each iteration")
    ("--nstepsPicent"; arg_type=Int; default=1; help="# of steps on Piᵏ.cent within each iteration (pre-training and main training); 0 fixes Piᵏ at initial (no training)")
    ("--fixPiTcent"; arg_type=Int; default=0; help="set to 1 to stop training C(X) for Piᵏ and Tᵏ after pre-training (no more updating of C(X) when training quantiles/offsets U(X) / L(X))")
    ("--seed"; arg_type=Int; default=238939; help="initial random seed")
    ("--seedreload"; arg_type=Int; default=0; help="set this random seed after each reload")
    ("--Pirange"; nargs='*'; default = ["[[-1.0f0, 1.0f0],", "[1.0f0, 2.0f0]]"]; arg_type=String; help="range of each parameters for Pi output")
    ("--nparamsT"; arg_type=Int; default=2; help="# of params for which T(X) will build a CI region of the form [center, range]")
    ("--Tsetrange"; arg_type=Int; default = 0; arg_type=Int; help="set to 0 (default) to use --Pirange to transform T(X) output, otherwise set to 1 and define --Trange [[...],[...]]")
    ("--Trange"; nargs='*'; default = ["[[-1.0f0, 1.0f0],", "[1.0f0, 2.0f0]]"]; arg_type=String; help="range of each parameter produced by T(X), ignored unless --Tsetrange 1")
    ("--Tlossη1"; default=0.20f0; arg_type=Float32; help="The 1st fixed η value to be used for evaluating CI size / coverage performance of Tᵏ (applies only to offsets) (the risk η .* CIsize kicks in)")
    ("--Tlossη2"; default=0.50f0; arg_type=Float32; help="The 2nd fixed η value to be used for evaluating CI size / coverage performance of Tᵏ (applies only to offsets) (the risk η .* CIsize kicks in)")
    ("--Tlossη_α"; default=2.0f0; arg_type=Float32; help="Parameter 1 (α) for the beta distribution of the tuning  parameter η")
    ("--Tlossη_β"; default=4.0f0; arg_type=Float32; help="Parameter 2 (β) for the beta distribution of the tuning  parameter η")
    ("--truepsi"; default="Ψ2dim_μ_σ"; help="function for evaluation of the true parameter value under P0")
    ("--name"; default="norm"; help="distribution name for P0")
    ("--Pibiascorrect"; default=0; arg_type=Int; help="set to 1 to adjust/correct the bias of initial Piᵏ wrt the inputs of Tᵏ corners")
    ("--lrPi"; default=0.001f0; arg_type=Float32; help="learning rate for Adam optim for Piᵏ")
    ("--β1Pi"; default=0.6f0; arg_type=Float32; help="β1 for Adam optim for Piᵏ")
    ("--lrT"; default=0.001f0; arg_type=Float32; help="learning rate for Adam optim for Tᵏ")
    ("--β1T"; default=0.6f0; arg_type=Float32; help="β1 for Adam optim for Tᵏ")
    ("--alpha"; default=0.05f0; arg_type=Float32; help="Define the α for CI coverage prob 1-α bandwitch")
    ("--lambdaMC"; default=100.0f0; arg_type=Float32; help="λ penalty against mode collapse, the inflation constant for penalty of loosing coverage wrt diffuse prior")
    ("--outdir"; default=nothing; help="output dir to save risks/models/optimizers")
    ("--loaddir"; default=nothing; help="output dir to load pre-trained models/optimizers")
    ("--loadfile"; default=nothing; help="pre-trained model/optimizer files to load")
    ("--saveEvery"; arg_type=Int; default=1; help="save network every saveEvery epochs")
    ("--hiddenPi"; nargs='*'; default = [16,32,32,16]; arg_type=Int; help="sizes of hidden layers for Piᵏ, defaults to --hidden 8 16 16 8 for a net with 4 hidden layers")
    ("--hiddenT"; nargs='*'; default = [10];  arg_type=Int; help="sizes of hidden layers for Tᵏ, defaults to --hidden 20 20 2 a net with 3 hidden layers")
    ("--numLSTMcells"; default=0; arg_type=Int; help="number of additional, repeated LSTM cells (of the same size as the initial spec'ed in --hiddenT)")
    ("--mTmodel"; arg_type=String; default = "LSTM2x"; help="NN architecture for Tᵏ, specify either 1) 'LSTM', 2) 'CXmle_LUlstm' or 3) 'LSTM2x'")
    ("--mTreduce"; arg_type=String; default = "sum"; help="reduction step over LSTM output of Tᵏ, either sum, MLP or CNN")
    ("--mPimodel"; arg_type=String; default = "MLP"; help="NN architecture for Piᵏ, specify either MLP or CNN")
    ("--maxRiskEvery"; arg_type=Int; default=1; help="report maximum risk of current estimator every maxRiskEvery epochs, using SGA for hardest risk")
    ("--verbose"; default=1; arg_type=Int; help="print diagnostics, e.g. the range of the prior and of the estimator")
  end

  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)
  if o[:outdir] != nothing
    o[:outdir] = abspath(o[:outdir])
  end
  return o
end

function main(args="";Pirange=nothing,Trange=nothing)
  atypeCPU = "Array{Float32}"
  atypeGPU = "CuArray{Float32}"
  o = parse_options(args)
  o[:seed] > 0 && srand(o[:seed]);
  atype = (o[:gpu]==0) ? eval(parse(atypeCPU)) : atype = eval(parse(atypeGPU))
  o[:atype] = atype

  ## parse the parameter range for Piᵏ
  if (Pirange!=nothing)
    o[:Pirange] = Pirange
  else
    o[:Pirange] = reduce(*, o[:Pirange])
    o[:Pirange] = eval(parse(o[:Pirange]))
  end

  o[:multivarSGA] = false
  o[:mTmodel] = Symbol(o[:mTmodel])
  o[:mTreduce] = Symbol(o[:mTreduce])
  o[:odimLSTM] =  10
  o[:odimLSTM1] = 10
  o[:odimLSTM2] = 10
  o[:mPimodel] = Symbol(o[:mPimodel])
  o[:losstypecent] = Symbol(o[:losstypecent])
  o[:epoch] = 0

  ## parse the parameter range for Tᵏ
  if (o[:Tsetrange]==0)
    o[:Trange] = o[:Pirange]
  else
    if (Trange!=nothing)
      o[:Trange] = Trange
    else
      o[:Trange] = reduce(*, o[:Trange])
      o[:Trange] = eval(parse(o[:Trange]))
    end
  end

  o[:Tδoffsets] = [0.5f0, 0.2f0]
  o[:Piδoffsets] = [0.5f0, 0.099f0]
  o[:lambdaCI] = 2.0f0
  o[:Ttransform] = "linear"
  o[:ntest] = 2000
  o[:lrSGA] = 0.01f0

  ## define the model parameter space for Piᵏ
  parboxPiᵏ = ParamBox([par for par in o[:Pirange]])
  modelP = ParamModelPᵏ(parboxPiᵏ,Ψ=Symbol(o[:truepsi]);name=parse(o[:name]),
                        xdim=o[:xdim],mTmodel=o[:mTmodel],δ=o[:Piδoffsets])
  ## define the model parameter space for Tᵏ
  parboxTᵏ = ParamBox([par for par in o[:Trange]])
  paramCI = ParamCI(parboxTᵏ,o[:nparamsT];α=o[:alpha],
                    η1=o[:Tlossη1],η2=o[:Tlossη2],η_α=o[:Tlossη_α],η_β=o[:Tlossη_β],
                    λ=Float32(o[:lambdaCI]),λMC=o[:lambdaMC],
                    δ=o[:Tδoffsets],trtype=Symbol(o[:Ttransform]))

  if (o[:verbose]==1 && o[:iepoch]==0)
    println("Range for each parameter: ")
    for parrange in o[:Pirange] println(parrange); end
    println("hiddenPi, network for Piᵏ: ", o[:hiddenPi])
    println("hiddenT, network for Tᵏ: ",  o[:hiddenT])
    println("Network architecture for Tᵏ, --mTmodel ",  o[:mTmodel])
    println("Reduce step for Tᵏ, --mTreduce ",  o[:mTreduce])
    println("Output dimension for LSTM, --odimLSTM: $(o[:odimLSTM])")
    println("Output dimension for LSTM#1, --odimLSTM1: $(o[:odimLSTM1])")
    println("Output dimension for LSTM#2, --odimLSTM2: $(o[:odimLSTM2])")
    println("Network architecture for Piᵏ, --mPimodel ",  o[:mPimodel])
    println("ParamBox Piᵏ:"); println(parboxPiᵏ)
    println("ParamBox Tᵏ:"); println(parboxTᵏ)
    println("Using Ψ: $(Symbol(o[:truepsi]))")
    println("Number of parameters for Tᵏ CI: $(paramCI.nparams)")
    println("Transformation type of output of Tᵏ: $(paramCI.trtype)")
    println("Optimizer for Piᵏ, lr = $(o[:lrPi]); β1 = $(o[:β1Pi])")
    println("Optimizer for Tᵏ, lr = $(o[:lrT]); β1 = $(o[:β1T]);")
    println("nstepsTcent = $(o[:nstepsTcent]); nstepsPicent = $(o[:nstepsPicent]); fixPiTcent = $(o[:fixPiTcent])")
    println("kcent: $(o[:kcent])")
    println("η_α: $(o[:Tlossη_α]); η_β: $(o[:Tlossη_β]);")
    println("losstypecent: $(o[:losstypecent]);")
    println("SGAnbatch: $(o[:SGAnbatch]); SGAnruns: $(o[:SGAnruns]); SGAnstarts1: $(o[:SGAnstarts1]); SGAnstarts2: $(o[:SGAnstarts2])")
  end

  if (o[:verbose]==1 && o[:iepoch]==0) println("initializing NNs"); end

  ## def init mPiᵏ -- hardest initial prior, will train Tᵏ against it
  mPi⁰init = initPiᵏ([10]...; atype=atype, winit=-1.5, udim = o[:udim]+6*length(paramCI), outdim = length(modelP));

  if (o[:mPimodel] == :MLP)
    println("_______Initializing MLP for Piᵏ_______")
    mPi⁰ = initPiᵏ(o[:hiddenPi]...; atype=atype, winit=-0.6, udim = o[:udim]+6*length(paramCI), outdim = length(modelP));
  end

  nobs = o[:xdim]
  nbatch = o[:nbatch]
  ## final CI output dimensions (number of centers + ranges to output) (NO LONGER USED)
  outdim = paramCI.nparams
  odimLSTM = (o[:mTreduce] == :sum) ? outdim : o[:odimLSTM]
  ## final CI output dims for the centers C(X) (number of parameters, one center per each)
  outdim1 = paramCI.nparams
  ## output dim for LSTM #1 (centers C(X))
  odimLSTM1 = (o[:mTreduce] == :sum) ? outdim1 : o[:odimLSTM1]
  ## final CI output dims for the upper / lower offsets (L(X),U(X)) (2 x # of parameters)
  outdim2 = 2*paramCI.nparams
  ## output dim for LSTM #2 (upper / lower offsets (L(X),U(X)))
  odimLSTM2 = (o[:mTreduce] == :sum) ? outdim2 : o[:odimLSTM2]

  if (o[:verbose]==1 && o[:iepoch]==0)
    println("_______Initializing LSTM for Tᵏ_______")
    println("outdim: $outdim; odimLSTM1: $odimLSTM1; odimLSTM2: $odimLSTM2; hiddenT: $(o[:hiddenT][1]); numLSTMcells: $(o[:numLSTMcells])")
  end

  mTᵏcent =  Chain(LSTM(1, o[:hiddenT][1]))
  mTᵏoff = Chain(LSTM(2+paramCI.nparams, o[:hiddenT][1]))
  for i=1:o[:numLSTMcells]
    push!(mTᵏcent.layers,   LSTM(o[:hiddenT][1], o[:hiddenT][1]))
    push!(mTᵏoff.layers,  LSTM(o[:hiddenT][1], o[:hiddenT][1]))
  end
  push!(mTᵏcent.layers,   Dense(o[:hiddenT][1], o[:hiddenT][1], leakyrelu))
  push!(mTᵏoff.layers,  Dense(o[:hiddenT][1], o[:hiddenT][1], leakyrelu))
  push!(mTᵏcent.layers,   Dense(o[:hiddenT][1], odimLSTM1))
  push!(mTᵏoff.layers,  Dense(o[:hiddenT][1], odimLSTM2))

  if (o[:loadfile] != nothing)
    o[:nepochs1Cx] = 0 ## no pre-training against uniform when loading existing Tᵏ model
    if (o[:loaddir] == nothing) error("must specify the path to model file using --loaddir"); end
    filepath = joinpath(o[:loaddir], o[:loadfile])
    if (o[:verbose]==1 && o[:iepoch]==0) println("loading pre-saved Tᵏ from $filepath"); end
    mTᵏload = loadNN(atype, filepath, o)
    mTᵏcent = mTᵏload.mTᵏcent
    if (o[:verbose]==1 && o[:iepoch]==0) println("successfully loaded mTᵏ"); println((mTᵏload)); end
  end

  ## Chain #1a: reduce + on out[1] & out[2] and L2 norm reduce on out[3] & out[4] (nparams=2)
  if (o[:verbose]==1 && o[:iepoch]==0) println("_______Initializing mean and L2-norm pooling for mTreduce_______"); end
  mTᵏreduce = Chain(x -> L2normReduce(x, nobs, nbatch, paramCI.nparams))
  ## Chain #1b reduce + on all output channels
  mTᵏreduce1 = Chain(x -> (reduce(+,x) ./ nobs))
  mTᵏreduce2 = Chain(x -> (reduce(+,x) ./ nobs))

  if (o[:mTmodel] == :LSTM2x)
    if (o[:verbose]==1 && o[:iepoch]==0) println("defining LSTM2x for 1) C(X) and 2) L(X)/U(X)"); end
    ## two separate LSTMs
    ## Tᵏ model with 1 LSTM for centers (C(X)); 1 LSTM for lower and upper offsets (L(X),U(X))
    mTᵏ = Tᵏ2x(mTᵏcent, mTᵏoff, mTᵏreduce1, mTᵏreduce2, modelP, paramCI)

  elseif (o[:mTmodel] == :CXmle_LUlstm)
    if (o[:verbose]==1 && o[:iepoch]==0) println("defining 1) LSTM for L(X)/U(X) and 2) MLE C(X)"); end
    mTᵏ = Tᵏ2x(CX_MLE(), mTᵏoff, mTᵏreduce1, mTᵏreduce2, modelP, paramCI)

  elseif (o[:mTmodel] == :LSTM)
    ## single LSTM outputs CI centers and ranges
    if (o[:verbose]==1 && o[:iepoch]==0) println("defining single LSTM...");  println("...with single reduction function"); end
    mTᵏ = Tᵏ(mTᵏcent, mTᵏreduce, modelP, paramCI)
  end

  mTᵏ = (o[:gpu]==0) ? mTᵏ : mTᵏ |> gpu ## convert the NN params to gpu CuArray
  if (o[:verbose]==1 && o[:iepoch]==0) println("mTᵏ:"); println((mTᵏ)); end

  dir1 = "pretrain1_models"
  dir2 = "pretrain2_models"
  if o[:outdir] != nothing #&& !isdir(o[:outdir])
    mkpath(o[:outdir])
    mkpath(joinpath(o[:outdir], dir1))
    mkpath(joinpath(o[:outdir], dir2))
    mkpath(joinpath(o[:outdir], "models"))
    mkpath(joinpath(o[:outdir], "opt"))
    mkpath(joinpath(o[:outdir], "risks"))
  end

  ## define the networks
  nets = Dict(:mPiᵏ => GroupedPᵏ(UniformPi⁰(), offsetUniform()),
              :mPiᵣ => GroupedPᵏ(UniformPi⁰(), offsetUniform()),
              :mTᵏ => mTᵏ)

  flush(STDOUT)
  o[:seed] > 0 && srand(o[:seed]); ## set the seed for 1st pre-training against fixed Piᵏ
  if (o[:nepochs1Cx] > 0)
    println("------------------------------------");
    println("pre-training Tᵏ centers for $(o[:nepochs1Cx]) epochs...")
    println("...against uniform [μ,σ]")
  end
  nets[:mPiᵏ] = GroupedPᵏ(UniformPi⁰(), offsetUniform()) ## use uniform prior instead of Piᵏ
  optPiᵏ = Dict(:optPiᵏcent => Flux.ADAM(params(nets[:mPiᵏ].cent), o[:lrPi], β1=o[:β1Pi]))
  optTᵏ = Dict(:mTᵏcent => Flux.ADAM(params(nets[:mTᵏ].mTᵏcent), o[:lrT], β1=o[:β1T]))
  presavenstepsPicent = o[:nstepsPicent]; o[:nstepsPicent] = 0
  pretrainTᵏcent(o[:nepochs1Cx],o,nets,optTᵏ[:mTᵏcent],optPiᵏ,modelP,paramCI,dir1,atype)

  flush(STDOUT)
  o[:seed] > 0 && srand(o[:seed]); ## set the seed for 2nd pre-training against NN Piᵏ
  if (o[:nepochs2Cx] > 0)
    println("------------------------------------");
    println("training Tᵏ centers for $(o[:nepochs2Cx]) epochs...")
    println("...against MLP Piᵏ.cent for [μ,σ]")
  end

  nprePiᵏ = (o[:nepochs2Cx]==0) ? 1 : 200
  mPiᵏcent = pretrainPiᵏ(nprePiᵏ,o,mPi⁰init,mPi⁰,nets[:mTᵏ],modelP,paramCI,atype)
  nets[:mPiᵏ] = GroupedPᵏ(mPiᵏcent, offsetUniform())
  optPiᵏ = Dict(:optPiᵏcent => Flux.ADAM(params(nets[:mPiᵏ].cent), o[:lrPi], β1=o[:β1Pi]))
  optTᵏ = Dict(:mTᵏcent => Flux.ADAM(params(nets[:mTᵏ].mTᵏcent), o[:lrT], β1=o[:β1T]))
  o[:nstepsPicent] = presavenstepsPicent
  pretrainTᵏcent(o[:nepochs2Cx],o,nets,optTᵏ[:mTᵏcent],optPiᵏ,modelP,paramCI,dir2,atype)

  flush(STDOUT)
  o[:seed] > 0 && srand(o[:seed]); ## set the seed for full training (includes quantiles)
  println("------------------------------------");
  println("training Tᵏ offsets for $(o[:nepochsOff]) epochs...")

  iOptmTᵏoff = initAdam(params(nets[:mTᵏ].mTᵏoff))
  optTᵏ = Dict(:mTᵏoff => ADAM(iOptmTᵏoff, o[:lrT], β1=o[:β1T]), :mTᵏoffstate => iOptmTᵏoff)
  maxRisks = trainepochs(o[:iepoch],o[:reloadEvery],o[:nepochsOff],o,nets,optTᵏ,optPiᵏ,modelP,paramCI,atype)
  return nothing
end

## -- iepoch counter always goes 0,1,2,3,…
## -- nepoch is always the final max # of epochs that we are trying to run
## -- # of iter in current train loop is based on value in reloadEvery
## -- istart = iepoch*reloadEvery + 1
## -- iend = (iepoch+1)*reloadEvery
function trainepochs(iepoch,reloadEvery,nepoch,o,nets,optTᵏ,optPiᵏ,modelP,paramCI,atype=Array{Float32})
  if (o[:fixPiTcent] == 1) println("fixing Piᵏ.cent and Tᵏ.cent (will no longer be updated)"); o[:nstepsPicent] = 0; end
  if (o[:seedreload] > 0) println("resetting re-load random seed to seedreload=$(o[:seedreload]);"); end
  o[:seedreload] > 0 && srand(o[:seedreload])

  reloadEvery = (reloadEvery == 0) ? nepoch : reloadEvery
  istart = iepoch*reloadEvery + 1
  iend = (iepoch + 1)*reloadEvery

  println((:iepoch, iepoch, :istart, istart, :iend, iend, :reloadEvery, reloadEvery, :nepoch, nepoch))

  lenrisks = (nepoch ÷ o[:maxRiskEvery])

  maxRisks = Dict(
    :rEpoch  => zeros(Int,lenrisks),
    :η1 => paramCI.η1,
    :η2 => paramCI.η2,
    :rtypeIerr_η1  => zeros(Float32,lenrisks),
    :rtypeIerr_η2  => zeros(Float32,lenrisks),
    :L2mean_typeIerr_η1 => zeros(Float32,lenrisks),
    :L2mean_typeIerr_η2 => zeros(Float32,lenrisks),
    :rCIlen_η1  => zeros(Float32,lenrisks),
    :rCIlen_η2  => zeros(Float32,lenrisks),
    :rCIlenScaled_η1 => zeros(Float32,lenrisks),
    :rCIlenScaled_η2 => zeros(Float32,lenrisks)
  )

  mPi⁰ = GroupedPᵏ(deepcopy(nets[:mPiᵏ].cent), deepcopy(nets[:mPiᵏ].off))
  z1n = allocX1n(modelP, o[:nbatch]); # z1n = allocX1n(modelP, o[:nbatch]; atype=atype);

  ## save init model and init optimizer state (only if this is the first time trainepochs is called)
  if ((o[:outdir] != nothing) && (iepoch == 0))
    mfile = @sprintf("%06d.",0) * ("bson")
    fpathTᵏ = joinpath(o[:outdir], "models", mfile)
    saveNN(fpathTᵏ, nets[:mTᵏ], o)
    fpathOpt = joinpath(o[:outdir], "opt", mfile)
    saveOpt(fpathOpt, optTᵏ[:mTᵏoffstate], o)
  end

  ## re-load prev. saved model and optimizer state to continue training
  ## re-load prev. saved maxRisks to continue filling those in
  if ((o[:outdir] != nothing) && (iepoch > 0))
    mfile = @sprintf("%06d.",istart-1) * ("bson")

    fpathTᵏ = joinpath(o[:outdir], "models", mfile)
    if (o[:verbose]==1) println("re-loading Tᵏ from $fpathTᵏ"); end
    nets[:mTᵏ] = loadNN(atype, fpathTᵏ, o)

    ## NOTE: everytime we re-load the model, we also have to re-init the optimizer (otherwise (x,Δ) in ParamAdam point to old NN)
    iOptmTᵏoff = initAdam(params(nets[:mTᵏ].mTᵏoff))
    optTᵏ = Dict(:mTᵏoff => ADAM(iOptmTᵏoff, o[:lrT], β1=o[:β1T]), :mTᵏoffstate => iOptmTᵏoff)

    fpathOpt = joinpath(o[:outdir], "opt", mfile)
    if (o[:verbose]==1) println("re-loading optTᵏ from $fpathOpt"); end

    loadOpt!(atype, fpathOpt, optTᵏ[:mTᵏoffstate], o)

    fpathR = joinpath(o[:outdir], "risks", "maxRisks.jld2")
    maxRisks = load(fpathR)["maxRisks"] # BSON.@load String(fpath) maxRisks
    if (o[:verbose]==1)
        println("....re-loaded risks maxRisks...");
        idx = (istart÷o[:maxRiskEvery])
        println("maxRisks[:rtypeIerr_η1]"); println((maxRisks[:rtypeIerr_η1][1:idx]))
        println("maxRisks[:rtypeIerr_η2]"); println((maxRisks[:rtypeIerr_η2][1:idx]))
        println("maxRisks[:L2mean_typeIerr_η1]"); println((maxRisks[:L2mean_typeIerr_η1][1:idx]))
        println("maxRisks[:L2mean_typeIerr_η2]"); println((maxRisks[:L2mean_typeIerr_η2][1:idx]))
        println("maxRisks[:rCIlen_η1]"); println((maxRisks[:rCIlen_η1][1:idx]))
        println("maxRisks[:rCIlen_η2]"); println((maxRisks[:rCIlen_η2][1:idx]))
        println("maxRisks[:rCIlenScaled_η1]"); println((maxRisks[:rCIlenScaled_η1][1:idx]))
        println("maxRisks[:rCIlenScaled_η2]"); println((maxRisks[:rCIlenScaled_η2][1:idx]))
    end
  end

  findworstPᵏ2(mTᵏ,modelP,paramCI,o,SGAtypes,nstarts,setη) = findMaxRisksPᵏ2(mTᵏ,modelP,paramCI,o,SGAtypes,nstarts,setη;
                                                                        multivarSGA=o[:multivarSGA],
                                                                        lrSGA=o[:lrSGA],
                                                                        nruns=o[:SGAnruns],
                                                                        nbatch=o[:SGAnbatch],
                                                                        ntest=o[:ntest])

  if (istart==1)
    SGAtypes = [:cent, :len, :lenScaled]
    maxRs, maxRγs, maxRγsD, SGAtypesRef, RisksALL, randγ = findworstPᵏ2(nets[:mTᵏ],modelP,paramCI,o,SGAtypes,o[:SGAnstarts2],:rand)
    for j in 1:size(randγ)[2]
      println((:iγ, randγ[:,j], :icent, round(RisksALL[SGAtypesRef[:cent],j],3), :ilenScaled,  round(RisksALL[SGAtypesRef[:lenScaled],j],3)))
    end
    println("-----------------------------------------------------------------")
  end

  for epoch = istart:iend
    o[:epoch] = epoch
    println((:epoch, o[:epoch]))

    flush(STDOUT)
    dlossval = losscovTᵏP₀ = Tllenval = 0

    maxepoch = o[:SGAramp_maxepoch]
    SGAnstarts_i = (epoch < maxepoch) ? o[:SGAnstarts1] + Int(floor((o[:SGAnstarts2] - o[:SGAnstarts1]) / (maxepoch - epoch))) : o[:SGAnstarts2]
    SGAnstarts = min(SGAnstarts_i, o[:SGAnstarts2])

    for _ in 1:o[:nstepsPicent]
      ## Train Piᵏ for worse CI length γ~Piᵏ (prior as a NN or specific fixed distribution, e.g. uniform)
      dlossval = trainPiᵏcent!(nets[:mPiᵏ].cent,nets[:mPiᵣ],nets[:mTᵏ],z1n,modelP,paramCI,optPiᵏcent,o,atype;printloss=true)
    end

    ## Sample gammas for CI length from that prior
    noise = sample_noise(atype,o[:udim],o[:nbatch])
    γCIcent = predictPiᵏ(nets[:mPiᵏ].cent,noise,nets[:mTᵏ],z1n,modelP,paramCI)
    γCIoff = predictPiᵏ(nets[:mPiᵏ].off,noise,nets[:mTᵏ],z1n,modelP,paramCI,SGAnstarts)
    Pγ = Dict(:cent => γCIcent, :off => γCIoff)
    losscovTᵏP₀ = trainTᵏ!(nets[:mTᵏ],Pγ,z1n,modelP,paramCI,optTᵏ,o,atype)

    ## find max risks with random grid search over γ (SGA)
    if (epoch%o[:maxRiskEvery]==0)
      idx = (epoch÷o[:maxRiskEvery])
      maxRisks[:rEpoch][idx] = epoch
      SGAtypes = [:typeIerr,:len,:lenScaled] # SGAtypes = [:offLμ,:offLσ,:offUμ,:offUσ]

      println("Running SGA")
      @time maxRs, maxRγs, maxRγsD, SGAtypesRef, RisksALL, randγ = findworstPᵏ2(nets[:mTᵏ],modelP,paramCI,o,SGAtypes,o[:SGAnstarts2],:η1)
      for j in 1:size(randγ)[2]
        println((:γ, randγ[:,j], :typeIerr, round(RisksALL[SGAtypesRef[:typeIerr],j],3), :len, round(RisksALL[SGAtypesRef[:len],j],3), :lenScaled, round(RisksALL[SGAtypesRef[:lenScaled],j],3)))
      end
      println("-----------------------------------------------------------------")
      Risks_typeIerr = RisksALL[SGAtypesRef[:typeIerr],:]
      L2mean_typeIerr = sqrt(mean(Risks_typeIerr .^ 2))
      println((:epoch, o[:epoch], :SGARISKS, :η1, paramCI.η1,
               :CIlen, round(maxRγsD[:len][1],3),
               :CIlenScaled, round(maxRγsD[:lenScaled][1],3),
               :typeIerr, round(maxRγsD[:typeIerr][1],3),
               :L2mean_typeIerr, round(L2mean_typeIerr,3)))
      maxRisks[:rtypeIerr_η1][idx]       = maxRγsD[:typeIerr][1]
      maxRisks[:L2mean_typeIerr_η1][idx] = L2mean_typeIerr
      maxRisks[:rCIlen_η1][idx]          = maxRγsD[:len][1]
      maxRisks[:rCIlenScaled_η1][idx]    = maxRγsD[:lenScaled][1]

      maxRs, maxRγs, maxRγsD, SGAtypesRef, RisksALL, randγ = findworstPᵏ2(nets[:mTᵏ],modelP,paramCI,o,SGAtypes,o[:SGAnstarts2],:η2)
      for j in 1:size(randγ)[2]
        println((:γ, randγ[:,j], :typeIerr, round(RisksALL[SGAtypesRef[:typeIerr],j],3), :len, round(RisksALL[SGAtypesRef[:len],j],3), :lenScaled, round(RisksALL[SGAtypesRef[:lenScaled],j],3)))
      end
      println("-----------------------------------------------------------------")
      Risks_typeIerr = RisksALL[SGAtypesRef[:typeIerr],:]
      L2mean_typeIerr = sqrt(mean(Risks_typeIerr .^ 2))
      println((:epoch, o[:epoch], :SGARISKS, :η2, paramCI.η2,
               :CIlen, round(maxRγsD[:len][1],3),
               :CIlenScaled, round(maxRγsD[:lenScaled][1],3),
               :typeIerr, round(maxRγsD[:typeIerr][1],3),
               :L2mean_typeIerr, round(L2mean_typeIerr,3)))
      maxRisks[:rtypeIerr_η2][idx]       = maxRγsD[:typeIerr][1]
      maxRisks[:L2mean_typeIerr_η2][idx] = L2mean_typeIerr
      maxRisks[:rCIlen_η2][idx]          = maxRγsD[:len][1]
      maxRisks[:rCIlenScaled_η2][idx]    = maxRγsD[:lenScaled][1]

      maxRs, maxRγs, maxRγsD, SGAtypesRef, RisksALL, randγ = findworstPᵏ2(nets[:mTᵏ],modelP,paramCI,o,SGAtypes,o[:SGAnstarts2],:rand)
      for j in 1:size(randγ)[2]
        println((:γ, randγ[:,j], :typeIerr, round(RisksALL[SGAtypesRef[:typeIerr],j],3), :len, round(RisksALL[SGAtypesRef[:len],j],3), :lenScaled, round(RisksALL[SGAtypesRef[:lenScaled],j],3)))
      end
      println("-----------------------------------------------------------------")
      Risks_typeIerr = RisksALL[SGAtypesRef[:typeIerr],:]
      L2mean_typeIerr = sqrt(mean(Risks_typeIerr .^ 2))
      println((:epoch, o[:epoch], :SGARISKS, :ηrand,
               :CIlen, round(maxRγsD[:len][1],3),
               :CIlenScaled, round(maxRγsD[:lenScaled][1],3),
               :typeIerr, round(maxRγsD[:typeIerr][1],3),
               :L2mean_typeIerr, round(L2mean_typeIerr,3)))
    end

    ## save models and generations, remove older model files
    if ((o[:outdir] != nothing) & (epoch%o[:saveEvery]==0 || epoch==iend))
      mfile = @sprintf("%06d.",epoch) * ("bson")

      fpathTᵏ = joinpath(o[:outdir], "models", mfile)
      saveNN(fpathTᵏ,nets[:mTᵏ],o)

      if (epoch==iend)
        fpathOpt = joinpath(o[:outdir], "opt", mfile)
        saveOpt(fpathOpt,optTᵏ[:mTᵏoffstate],o)
      end

      fpath = joinpath(o[:outdir], "risks", "maxRisks.jld2")
      save(fpath, "maxRisks", maxRisks)

      delepoch = 2000
      rmepoch = epoch-delepoch
      if ((epoch > delepoch) && ((rmepoch % 250) != 0))
        filepath = joinpath(o[:outdir], "models", @sprintf("%06d.",rmepoch) * ("bson"))
        rm(filepath,force=true)
      end
    end
  end
  try gc(); CuArrays.clearpool(); end
  return nothing
end

function pretrainTᵏcent(nepoch,o,nets,optTᵏ,optPiᵏ,modelP,paramCI,dir,atype=Array{Float32})
  lenrisks = nepoch÷o[:maxRiskEvery]
  CxRisks = Dict(
    :rEpoch       => Vector{Int}(lenrisks),
    :rSGA_Cx      => Vector{Float32}(lenrisks),
    :gamma_SGA_Cx => [Vector{Float32}(length(modelP)) for i=1:lenrisks],
    :rTk_Cx       => Vector{Float32}(lenrisks)
  )

  # optTᵏ = optTᵏ[:optTᵏsize]
  optPiᵏcent = optPiᵏ[:optPiᵏcent]
  z1n = allocX1n(modelP, o[:nbatch]);

  findworstPᵏ2(mTᵏ,modelP,paramCI,o,SGAtypes,nstarts) = findMaxRisksPᵏ2(mTᵏ,modelP,paramCI,o,SGAtypes,nstarts,:rand;
                                                            multivarSGA=o[:multivarSGA],
                                                            lrSGA=o[:lrSGA],
                                                            nruns=o[:SGAnruns],
                                                            nbatch=o[:SGAnbatch],
                                                            ntest=o[:ntest])

  if ((nepoch > 0) & (o[:Pibiascorrect] == 1))
    u = sample_noise(atype,o[:udim],o[:nbatch])
    TᵏCxs = gencornersCx(nets[:mTᵏ],u,z1n,modelP,paramCI)
    Δbias = vcat(0.0f0,0.0f0,mean(TᵏCxs, 2)) |> gpu
    println((:Δbias, Δbias))
    println("Bias nets[:mPiᵏ].cent pre: "); println((nets[:mPiᵏ].cent[2].data))
    nets[:mPiᵏ].cent[2].data .= nets[:mPiᵏ].cent[2].data .- (4 .* Δbias)
    println("Bias nets[:mPiᵏ].cent post:"); println((nets[:mPiᵏ].cent[2].data))
  end

  for epoch = 1:nepoch
    o[:epoch] = epoch
    println((:pretrainepoch, o[:epoch]))
    flush(STDOUT)
    Tllenval = Tᵏloss = 0f0
    for _ in 1:o[:nstepsPicent]
      ## 1. Train Piᵏ for worse CI center γ~Piᵏ (prior as a NN or specific distribution, s.a. uniform)
      @time dlossval = trainPiᵏcent!(nets[:mPiᵏ].cent,nets[:mPiᵣ],nets[:mTᵏ],z1n,modelP,paramCI,optPiᵏcent,o,atype;printloss=true)
    end
    # try gc(); CuArrays.clearpool(); end
    for _ in 1:o[:nstepsTcent]
      noise = sample_noise(atype,o[:udim],o[:nbatch])
      Pγcent = predictPiᵏ(nets[:mPiᵏ].cent,noise,nets[:mTᵏ],z1n,modelP,paramCI)
      Pγ = Dict(:cent => Pγcent)
      Tᵏloss = trainTᵏcent!(nets[:mTᵏ],Pγ,z1n,modelP,paramCI,optTᵏ,o,atype)
    end

    if (epoch%o[:maxRiskEvery]==0)
        # SGAtypes = (:len,:cent)
        SGAtypes = [:cent]
        @time maxRs, maxRγs, maxRγsD, SGAtypesRef, RisksALL, randγ = findworstPᵏ2(nets[:mTᵏ],modelP,paramCI,o,SGAtypes,o[:SGAnstarts2])
        println((:epoch, o[:epoch], :rmaxCIcent, round(maxRγsD[:cent][1],3), :γCIcent, maxRγsD[:cent][2]))
        CxRisks[:rEpoch][epoch÷o[:maxRiskEvery]]        = epoch
        CxRisks[:rSGA_Cx][epoch÷o[:maxRiskEvery]]       = maxRγsD[:cent][1]
        CxRisks[:gamma_SGA_Cx][epoch÷o[:maxRiskEvery]]  = maxRγsD[:cent][2]
        CxRisks[:rTk_Cx][epoch÷o[:maxRiskEvery]]        = Tᵏloss
    end

    # if (o[:verbose]==1 && (epoch%10 == 0))
    if (o[:verbose]==1)
      ## evaluate losses on new test data
      noise = sample_noise(atype,o[:udim],o[:nbatch])
      γsampᵏ = predictPiᵏ(nets[:mPiᵏ].cent,noise,nets[:mTᵏ],z1n,modelP,paramCI)
      γsampᵣ_invunif = predictPiᵏ(UniformInversePi⁰(),noise,nets[:mTᵏ],z1n,modelP,paramCI)
      γsampᵣ_unif = predictPiᵏ(UniformPi⁰(),noise,nets[:mTᵏ],z1n,modelP,paramCI)

      ψ₀, centTᵏPiᵏ = generateCIcent(γsampᵏ,nets[:mTᵏ],z1n,modelP,paramCI)
      ψ₀, centTᵏPiᵣ_invunif = generateCIcent(γsampᵣ_invunif,nets[:mTᵏ],z1n,modelP,paramCI)
      ψ₀, centTᵏPiᵣ_unif = generateCIcent(γsampᵣ_unif,nets[:mTᵏ],z1n,modelP,paramCI)

      println((:TᵏPiᵏ))
      printPiᵏepoch(epoch,γsampᵏ...)
      printTᵏcent(epoch,centTᵏPiᵏ...)
      println((:TᵏPiᵣ_invunif))
      printPiᵏepoch(epoch,γsampᵣ_invunif...)
      printTᵏcent(epoch,centTᵏPiᵣ_invunif...)
      println((:TᵏPiᵣ_unif))
      printPiᵏepoch(epoch,γsampᵣ_unif...)
      printTᵏcent(epoch,centTᵏPiᵣ_unif...)
    end

    ## save models and generations, remove older model files
    if ((o[:outdir] != nothing) & (epoch%o[:saveEvery]==0))
      filepath = joinpath(o[:outdir], "risks", "pretrain_Cxrisks.jld2")
      save(filepath, "CxRisks", CxRisks)
      filepathTᵏ = joinpath(o[:outdir], dir, @sprintf("%06d.",epoch) * ("bson"))
      filepathPiᵏ = joinpath(o[:outdir], dir, "Pi" * @sprintf("%06d.",epoch) * "jld2")
      saveNN(filepathTᵏ, filepathPiᵏ, nets[:mTᵏ], nets[:mPiᵏ].cent, o)
    end
  end
  return nothing
end

## Train Tᵏ centers C(X) only
function trainTᵏcent!(mTᵏ,Pγ,z1n,modelP,paramCI,optTᵏ,o,atype)
  g1 = map(x -> istracked(x) && zero_grad!(grad(x)), params(mTᵏ))
  lcent = Tᵏlosscent(mTᵏ,z1n,Pγ,modelP,paramCI,o)
  Flux.back!(lcent)
  ΔTᵏcent = map(x -> istracked(x) && grad(x), params(mTᵏ.mTᵏcent))
  maxΔTᵏcent = map(x -> maximum(abs.(x)), ΔTᵏcent)
  println("lcent: maxΔTᵏcent:"); println((maxΔTᵏcent))
  ΔTᵏoff = map(x -> istracked(x) && grad(x), params(mTᵏ.mTᵏoff))
  maxΔTᵏoff = map(x -> maximum(abs.(x)), ΔTᵏoff)
  println("lcent: maxΔTᵏoff:"); println((maxΔTᵏoff))
  optTᵏ()
  g1 = map(x -> istracked(x) && zero_grad!(grad(x)), params(mTᵏ))
  println((:lTᵏ_Cx, round(data(lcent),3)))
  return data(lcent)
end

function initPiᵏ(h...; atype=Array{Float32}, winit=1.5, udim = 2, outdim = 2)
  w = Any[]
  # use udim = 28*28 for images
  for nextd in [h..., outdim] #
    push!(w, convert(atype, winit*randn(Float32,nextd,udim)))
    push!(w, convert(atype, zeros(Float32,nextd,1)))
    udim = nextd
  end
  return w
end

function loadNN(atype,loadfile,o)
  BSON.@load String(loadfile) mTᵏ
  mTᵏ = (o[:gpu]==0) ? mTᵏ : mTᵏ |> gpu ## convert the NN params to gpu CuArray
  return mTᵏ
end

function saveNN(savefileTᵏ, savefilePiᵏ, mTᵏ, mPiᵏ, o)
    mTᵏ = mTᵏ |> cpu
    BSON.@save savefileTᵏ mTᵏ
    if isa(mPiᵏ, Array)
      ## save arrays of params for Piᵏ:
      mPiᵏtosave = convertNN(mPiᵏ,Array{Float32})
      save(savefilePiᵏ, "Pi", mPiᵏtosave)
    end
end

function saveNN(savefileTᵏ, mTᵏ, o)
  mTᵏ = mTᵏ |> cpu
  BSON.@save savefileTᵏ mTᵏ
end

function loadOpt!(atype, loadfile, optTᵏstate, o)
  newOpt = BSON.load(String(loadfile))
  newOpt = newOpt[:optTᵏstate]
  newOpt = map(p -> p |> gpu, newOpt)
  ## 2 alternatives after prev. opt state has been loaded back to GPU:
  ## a) init a new optimizer, with previous state (need to return the new opt)
  #   optTᵏ = Dict(:mTᵏoff => ADAM(mTᵏoffstate, o[:lrT], β1=o[:β1T]), :mTᵏoffstate => mTᵏoffstate)
  #   return optTᵏ
  ## b) update the current optimizer state by reference, in this case the optimizer state in optTᵏ will be also automatically updated by ref
  loadparams!.(optTᵏstate, newOpt)
  return nothing
end

function saveOpt(savefile, optTᵏstate, o)
  optTᵏstate = map(p -> p |> cpu, optTᵏstate)
  BSON.@save savefile optTᵏstate  ## to save entire Flux model struct
end

function convertNN(w, atype=Array{Float32})
  w = map(wi->data(wi) |> cpu, w)
  return w
end

function gencornersCx(mTᵏ,noise,z1n,modelP,paramCI)
  Piᵏin = corner(modelP)
  mid1 = [Piᵏin[1][1], mid(modelP)[2]]
  mid2 = [Piᵏin[2][1], mid(modelP)[2]]
  push!(Piᵏin, mid1)
  push!(Piᵏin, mid2)

  mPis  = map(x -> ConstPi⁰(x[1],x[2]), Piᵏin)
  Pγs   = map(Pi -> predictPiᵏ(Pi,noise,mTᵏ,z1n,modelP,paramCI), mPis)
  TᵏCxs = map(γ  -> generateCIcent(γ,mTᵏ,z1n,modelP,paramCI)[2], Pγs)
  TᵏCxs = map(Cx -> map(Cx -> data(Cx),Cx), TᵏCxs)

  ##substract μ (γ₀[1]) and divide by σ (γ₀[2])
  TᵏCxs = map((x1,γ₀) -> [(x1[1] .- γ₀[1]) ./ γ₀[2], (x1[2] .- γ₀[2]) ./ γ₀[2]], TᵏCxs, Piᵏin);
  TᵏCxs = map(Cx -> vcat(Cx'...), TᵏCxs)
  TᵏCxs = vcat(TᵏCxs...)
  return TᵏCxs
end

# predictPiᵏ(w,x,modelP::ParamModelPᵏ,paramCI::ParamCI) = predictPiᵏ(w,x,modelP,paramCI)
function predictPiᵏ(w,x,mTᵏ,z1n,modelP,paramCI; pdrop=0.00)
  TᵏCxs = gencornersCx(mTᵏ,x,z1n,modelP,paramCI)
  x = vcat(x,TᵏCxs)
  for i=1:2:length(w)-2
    x = leakyrelu.(w[i]*dropout(x, pdrop) .+ w[i+1])
  end
  x = σ.(w[end-1]*x .+ w[end])
  x = [x[i,:] for i=1:length(modelP)]
  ## vs. 1, one-liner
  x = map(lintrans, x, lb(modelP), ub(modelP))
  return x
end

function predictPiᵏpre(w,x,mTᵏ,z1n,modelP,paramCI)
  Piᵏin = corner(modelP)
  mid1 = [Piᵏin[1][1], mid(modelP)[2]]
  mid2 = [Piᵏin[2][1], mid(modelP)[2]]
  push!(Piᵏin, mid1)
  push!(Piᵏin, mid2)

  mPis  = map(x -> ConstPi⁰(0,0), Piᵏin)
  Pγs   = map(Pi -> vcat(predictPiᵏ(Pi,x,mTᵏ,z1n,modelP,paramCI)'...), mPis)
  TᵏCxs = vcat(Pγs...)
  x = vcat(x,TᵏCxs)
  for i=1:2:length(w)-2
    x = leakyrelu.(w[i]*x .+ w[i+1])
  end
  x = σ.(w[end-1]*x .+ w[end])
  x = [x[i,:] for i=1:length(modelP)]
  x = map(lintrans, x, lb(modelP), ub(modelP))
  return x
end

## Flux CNN prediction
function predictPiᵏ(w::Flux.Chain,x,mTᵏ,z1n,modelP,paramCI)
  nobs = size(x,1)   ## nobs in single sample to loop over
  nbatch = size(x,2) ## the batch size for training
  # if using CNN, reshape the input
  x = reshape(x, :, 1, 1, nbatch)
  x = w(x)
  x = [x[i,:] for i=1:length(modelP)]
  x = map(lintrans, x, lb(modelP), ub(modelP))
  return x
end

function predictPiᵏ(w::GroupedPᵏ,x,mTᵏ,z1n,modelP,paramCI)
  return vcat(predictPiᵏ(w.cent,x,mTᵏ,z1n,modelP,paramCI),
              predictPiᵏ(w.off,x,mTᵏ,z1n,modelP,paramCI)
              )
end

## uniform sample for σ
## uniform sample for μ
function predictPiᵏ(w::UniformPi⁰,x,mTᵏ,z1n,modelP,paramCI)
  function sim_unif(nbatch,lb,ub)
    (rand(eltype(x),nbatch) .* (ub.-lb).+lb) |> gpu
  end
  lbb = lb(modelP)
  ubb = ub(modelP)
  nsim = [size(x)[2] for _ = 1:length(modelP)]
  x = map(sim_unif, nsim, lbb, ubb)
  return x
end

## precision sample for σ (i.e., uniform sample based on 1/σ^2 weighting)
## uniform sample for μ
function predictPiᵏ(w::UniformInversePi⁰,x,mTᵏ,z1n,modelP,paramCI)
  function sim_unif(nbatch,lb,ub)
    (rand(eltype(x),nbatch) .* (ub.-lb).+lb) |> gpu
  end
  lbb = lb(modelP)
  ubb = ub(modelP)
  σbounds = [lbb[2],ubb[2]]
  lbb[2] = Float32(1/(σbounds[2])^2)
  ubb[2] = Float32(1/(σbounds[1])^2)
  nsim = [size(x)[2] for _ = 1:length(modelP)]
  x = map(sim_unif, nsim, lbb, ubb)
  x[2] = sqrt.(1 ./ x[2])
  return x
end

function predictPiᵏ(w::UniformμConstσPi⁰,x,mTᵏ,z1n,modelP,paramCI)
  function sim_unif(nbatch,lb,ub)
    (rand(eltype(x),nbatch) .* (ub.-lb).+lb) |> gpu
  end
  lbb = lb(modelP)
  ubb = ub(modelP)
  nsim = size(x)[2]
  xμ = sim_unif(nsim, lbb[1], ubb[1])
  xμ = reshape(xμ, 1, nsim)
  xσ = repeat([w.σ],inner=(1,size(x)[2])) |> gpu
  x = vcat(xμ,xσ)
  x = [x[i,:] for i=1:length(modelP)]
  return x
end

## by convention, when no weight matrix is supplied, generate uniform (diffuse prior)
function predictPiᵏ(w::ConstPi⁰,x,mTᵏ,z1n,modelP,paramCI)
  x = repeat([w.μ,w.σ],inner=(1,size(x)[2])) |> gpu
  x = [x[i,:] for i=1:length(modelP)]
  return x
end

## when running SGA for Pᵏ (minimax formulation), noise arg (x) are the hardest γ
predictPiᵏ(w::SGAPᵏ,x,mTᵏ,z1n,modelP,paramCI) = x

function predictPiᵏ(w::offsetUniform,x,mTᵏ,z1n,modelP,paramCI,nstarts)
  xcorners = corner(modelP)
  xborders = Any[]
  for x in xcorners
    push!(xborders, [x[1], rand(modelP, 1)[1][2]])
    push!(xborders, [rand(modelP, 1)[1][1], x[2]])
  end
  xinit = [xcorners; xborders; [mid(modelP)]; rand(modelP, nstarts)]
  return xinit
end

## predict w/ LSTM (w1) using 2nd chain (w2) to process LSTM output
## w2 can be either another NN or reduce operation (pooling layer)
function predictLSTM(w1::Flux.Chain,w2::Flux.Chain,x,nparams)
  xdim = size(x,1)
  nobs = size(x,2)   ## nobs in single sample to loop over
  nbatch = size(x,3) ## the batch size for training
  x = [reshape(x[:,i,:],xdim,nbatch) for i=1:nobs] # x = Flux.unstack(x,1)'
  Flux.truncate!(w1)
  Flux.reset!(w1)
  tmp = w1.(x) ## burn-in throw-away period, then real preds on warmed-up Tᵏ
  ## apply first chain which is a recurrent cell (LSTM), so applied separately to each observation x (of batch size nbatch)
  out = w1.(x)
  ## apply 2nd chain to process the output of all n observations:
  out = w2(out)

  return out
end

function makeCIs2xTᵏ(out_cen,out_off,nparams,lbb,ubb,δ,trtype)
  if (nparams==2)
    LCIμ = out_cen[1] .- (out_off[1])
    UCIμ = out_cen[1] .+ (out_off[2])
    LCIσ = out_cen[2] .- (out_off[3])
    UCIσ = out_cen[2] .+ (out_off[4])
    return [LCIμ,UCIμ,LCIσ,UCIσ]
  elseif (nparams==1)
    LCIμ = out_cen[1] .- (out_off[1])
    UCIμ = out_cen[1] .+ (out_off[2])
    return [LCIμ,UCIμ]
  end
end

function makecenter(out,nparams,lbb,ubb,δ,trtype)
  out = [out[i,:] for i=1:nparams]
  return out
end

function makeoffset(out,nparams,lbb,ubb,δ,trtype)
  out = [out[i,:] for i=1:(2*nparams)]
  return out
end

predictTᵏcent(c::Tᵏ2x,x) = predictTᵏcent(c.mTᵏcent,c.mTᵏreduce1,x,c.nparams,c.lbb,c.ubb,c.δ;trtype=c.trtype,NNtype=c.NNtype)

function predictTᵏcent(wcen::Flux.Chain,wreduce1::Flux.Chain,x,nparams,lbb,ubb,δ;trtype=:linear,NNtype=:LSTM)
  out_cen = predictLSTM(wcen,wreduce1,x,nparams) ## predict w/ LSTM
  out_cen = makecenter(out_cen,nparams,lbb,ubb,δ,trtype)
  return out_cen
end

function predictTᵏcent(wcen::CX_MLE,wreduce1,x,nparams,lbb,ubb,δ;trtype=:linear,NNtype=:LSTM)
  return predictTᵏcent(wcen,x)
end

function predictTᵏcent(wcen::CX_MLE,x)
  ## input x dims are (1,nobs,nbatch)
  x = x |> cpu
  nbatch = size(x,3)
  μ̂ = reshape(mean(x,2),1,nbatch)
  xminμ = x[1,:,:] .- μ̂
  σ̂ = sqrt.(mean(abs2.(xminμ), 1))
  out_cen = [reshape(μ̂,nbatch) |> gpu, reshape(σ̂,nbatch) |> gpu]
  return out_cen
end

function predictTᵏlen(wcen::CX_MLE,x,n;γ=0.079f0)
    α1 = α2 = 1-√(1-γ) # (1-α1)*(1-α2) ## check roughly equals γ
    dχ² = Distributions.Chisq(n-1)
    dN  = Distributions.Normal(0,1)
    a   = Distributions.cquantile(dN, α1/2) ## upper α1/2 percentile of N(0,1)
    b   = Distributions.quantile(dχ²,α2/2) ## lower α2/2 percentile of χ²(n-1)
    c   = Distributions.cquantile(dχ²,α2/2) ## upper α2/2 percentile (1-α2/2) of χ²(n-1)
    μ̂, σ̂ = predictTᵏcent(CX_MLE(),x)
    lbσ = σ̂ ./ √(c/n)
    ubσ = σ̂ ./ √(b/n)
    lenσ = ubσ .- lbσ
    lenμ = 2a .* (ubσ ./ √n)
    CIsize = lenσ .* lenμ
    return CIsize
end

function predictTᵏoff(c::Tᵏ2x,x,η)
  out_cen = predictTᵏcent(c,x)
  out_off = predictTᵏoff(c.mTᵏoff,c.mTᵏreduce2,x,η,out_cen,c.nparams,c.lbb,c.ubb,c.δ;trtype=c.trtype,NNtype=c.NNtype)
  return out_off
end

## New predict fun for Tᵏ offset (use C(X) as inputs)
function predictTᵏoff(woff::Flux.Chain,wreduce2::Flux.Chain,x,η,CX,nparams,lbb,ubb,δ;trtype=:linear,NNtype=:LSTM)
  nobs = size(x,2)  ## no. of obs in one batch (rows)
  nbatch = size(x,3) ## the batch size for training (cols)
  CX = [(data(CX[i]) |> cpu) for i=1:nparams]
  CX = hcat(CX...)'
  ## add η, add features to x (each xᵢ gets a new 3-dim feature vector (η,C1(X),C2(X)))
  addx = Array{Float32}(3,nobs,nbatch)
  addx[1,:,:] .= reshape((η |> cpu),1,nbatch)
  addx[2,:,:] .= reshape(CX[1,:],1,nbatch)
  addx[3,:,:] .= reshape(CX[2,:],1,nbatch)

  xη = cat(1, x, addx |> gpu)
  out_off = predictLSTM(woff,wreduce2,xη,nparams) ## predict w/ LSTM
  out_off = makeoffset(out_off,nparams,lbb,ubb,δ,trtype)
  return out_off
end

function predictTᵏ(wcen,woff,wreduce1,wreduce2,x,η,nparams,lbb,ubb,δ;trtype=:linear,NNtype=:LSTM)
  out_cen = predictTᵏcent(wcen,wreduce1,x,nparams,lbb,ubb,δ;trtype=:linear,NNtype=:LSTM)
  out_off = predictTᵏoff(woff,wreduce2,x,η,out_cen,nparams,lbb,ubb,δ;trtype=:linear,NNtype=:LSTM)
  CIs = makeCIs2xTᵏ(out_cen,out_off,nparams,lbb,ubb,δ,trtype)
  return [CIs, out_cen, out_off, η |> gpu]
end

function Piᵏlosscent(mPiᵏcent,mPiᵣ,mTᵏ,noise,z1n,modelP,paramCI,o;printloss::Bool=false)
  λMC = paramCI.λMC
  γ = predictPiᵏ(mPiᵏcent,noise,mTᵏ,z1n,modelP,paramCI)
  ψ₀, centTᵏ = generateCIcent(γ,mTᵏ,z1n,modelP,paramCI)
  R1_Cx = rCIcent(centTᵏ...,γ,kcent=o[:kcent],losstypecent=o[:losstypecent])

  # Regularization wrt diffuse prior (can't perform worse than diffuse / uniform prior)
  γᵣU    = predictPiᵏ(UniformPi⁰(),noise,mTᵏ,z1n,modelP,paramCI)
  γᵣInvU = predictPiᵏ(UniformInversePi⁰(),noise,mTᵏ,z1n,modelP,paramCI)
  ψ₀ᵣU, centTᵏᵣU       = generateCIcent(γᵣU,mTᵏ,z1n,modelP,paramCI)
  ψ₀ᵣInvU, centTᵏᵣInvU = generateCIcent(γᵣInvU,mTᵏ,z1n,modelP,paramCI)
  R1_CxᵣU    = rCIcent(centTᵏᵣU...,   γᵣU,   kcent=o[:kcent],losstypecent=o[:losstypecent])
  R1_CxᵣInvU = rCIcent(centTᵏᵣInvU...,γᵣInvU,kcent=o[:kcent],losstypecent=o[:losstypecent])
  λ_CxᵣU    = (λMC)*leakyrelu(R1_CxᵣU   -R1_Cx)
  λ_CxᵣInvU = (λMC)*leakyrelu(R1_CxᵣInvU-R1_Cx)

  if (printloss)
    println((:lPiᵏ_Cx,       round(data(R1_Cx),3),
             :lPiᵏ_CxᵣU,     round(data(R1_CxᵣU),3),
             :lPiᵏ_CxᵣInvU,  round(data(R1_CxᵣInvU),3),
             :λ_CxᵣU,        round(data(λ_CxᵣU),3),
             :λ_CxᵣInvU,     round(data(λ_CxᵣInvU),3)
             ))
  end
  loss = -R1_Cx + λ_CxᵣU + λ_CxᵣInvU
  return loss
end

## Tᵏ loss for CI centers C(X)
function Tᵏlosscent(mTᵏ,z1n,Pγ,modelP,paramCI,o)
  γ = Pγ[:cent] # sample from current hardest prior Piᵏ or hardest fixed γ~Pᵏ
  ψ₀, centTᵏ = generateCIcent(γ,mTᵏ,z1n,modelP,paramCI)
  R1_Cx =  rCIcent(centTᵏ...,γ,kcent=o[:kcent],losstypecent=o[:losstypecent])
  loss = R1_Cx
  return loss
end

∑cov(mPiᵏ::SGAPᵏ,ProbΨoutCI) = mean(ProbΨoutCI) ## when running SGA take mean over all cov. probs, 'cause SGA uses only a single γ
∑cov(mPiᵏ,ProbΨoutCI) = ProbΨoutCI ## when running Piᵏ maximin, keep the sep cov. prob for each γ~Piᵏ

function TᵏlossLxUxback!(mTᵏ,z1n,Pγ,modelP,paramCI,o)
  α = paramCI.α
  loss = offloss = 0.0f0 ## surrogate (hinge) sum loss, only kicks in if true coverage is subnominal on particular dataset
  sumILossP₀ = 0 ## true coverage (non-differentiable)
  γ = Pγ[:off] # sample from current hardest prior Piᵏ or hardest fixed γ~Pᵏ or just diffuse Pᵏ
  nbatch = size(z1n,2)
  nγ = length(γ)
  if (o[:verbose]==1 && (o[:epoch]%o[:maxRiskEvery]==0)) println((:nγ, nγ)); end
  accumΔoff = map(x -> istracked(x) && copy(grad(x)), params(mTᵏ.mTᵏoff))
  Δoff = map(x -> istracked(x) && copy(grad(x)), params(mTᵏ.mTᵏoff))

  for i = 1:nγ
    offloss = evalQuantLossTᵏ(mTᵏ,γ[i],nbatch,modelP,paramCI,o) ./ nγ
    Flux.back!(offloss)
    Δoff = map(x -> istracked(x) && copy(grad(x)), params(mTᵏ.mTᵏoff))
    g = map(x -> istracked(x) && zero_grad!(grad(x)), params(mTᵏ.mTᵏoff))
    ## accumulate the gradient from each dataset
    Flux.Tracker.accum!(accumΔoff, Δoff)
    loss += data(offloss)
  end
  ## over-write ΔTᵏ with the final accumΔ:
  ΔmTᵏoff = map((x,y) -> x.grad .= y, params(mTᵏ.mTᵏoff), accumΔoff)
  ΔmTᵏoff = nothing
  accumΔoff = nothing
  Δoff = nothing
  return data(loss)
end

## Tᵏ loss for CI offsets [L(X), U(X)]
## Note that separate Pᵏ are used for L(X) and U(X)
## This means that this loss is best split into two losses (L(X),U(X))
#1. ONLY 1 hardest γL (fixed) (this γ will be optimized adversarily against this Tᵏ loss)
#2. Grab the centers C(X) and lower bounds L(X) from Tᵏ(X), based on the m batch sample X~γL
  ##*** NOTE All X must be sampled from the same fixed γL
#3a. qL: evaluate (1-α/4) quantile of C(X) - γL
  # then P(C(X) - γL < qL)=1-α/4
  # then P(C(X) - qL < γL)=1-α/4
  # then P(C(X) - qL < ψ₀)=1-α/4
  # then P(C(X) - qL ≧ ψ₀)=α/4
#3b. ***in separate loss fun*** qU: evaluate (1-α/4) quantile of γU - C(X)
  # then P(γU - C(X) < qU)=1-α/4
  # then P(γU < C(X) + qU)=1-α/4
  # then P(ψ₀ < C(X) + qU)=1-α/4
  # then P(ψ₀ ≧ C(X) + qU)=α/4
##4a. Obtain L(X) an estimate of qL (if we knew qL, we could construct our CI) using loss
  #****# I(L(X) < qL)*(L(X) - qL)^2 + η .* I(L(X)>qL)*(L(X) - qL)^2 #****#
  ## NOTE: suppose L(X) is smaller than qL
  ## NOTE: then P(C(X) ≧ ψ₀ + L(X)) > P(C(X) ≧ ψ₀ + qL) = α/4
  ## Which means that we would get subnominal coverage
##4b. Obtain U(X) an estimate of qU (if we knew qU, we could construct our CI) using loss
#****# I(U(X) < qU)*(U(X) - qU)^2 + η .* I(U(X)>qU)*(U(X) - qU)^2 #****#
    ##4. Why is the resulting CI given by P(C(X)-qL < ψ₀ < C(X)+qU) ≦ 1-α/2 will be correct?
    # A: union bound, since above joint bound is an intersection of two events
    # P(C(X)-qL < ψ₀ & ψ₀ < C(X)+qU) = (1-α/2)
    # 1-P(C(X)-qL < ψ₀ & ψ₀ < C(X)+qU) = α/2
    # P(C(X)-qL ≧ ψ₀ || ψ₀ ≧ C(X)+qU) = α/2
    # P(C(X)-qL ≧ ψ₀ || ψ₀ ≧ C(X)+qU) ≦ P(C(X)-qL ≧ ψ₀) + P(ψ₀ ≧ C(X)+qU)
    ## But from training out procedure and from definition of (qL,qU) we already know
    # P(C(X)-qL ≧ ψ₀) + P(ψ₀ ≧ C(X)+qU) = α/2
function evalQuantLossTᵏ(mTᵏ,γi,nbatch,modelP,paramCI,o)
  dβ = Distributions.Beta(paramCI.η_α,paramCI.η_β)
  η = Array{Float32}(nbatch)
  η = rand!(dβ, η) |> gpu # η = rand!(dβ, η)[1]
  γi = copy(γi) |> gpu
  z1n = allocX1n(modelP, nbatch)
  ψ₀, predTᵏ = generateCIparams(γi,mTᵏ,z1n,η,modelP,paramCI)
  # if (o[:verbose]==1 && (o[:epoch]%o[:maxRiskEvery]==0))
  #   println("--------------------------------------------------");
  # end
  lossμL = Tᵏloss_offLγ(mTᵏ,:μL,η,data(predTᵏ[2][1]),predTᵏ[3][1],γi,Ψμ,z1n,modelP,paramCI,o) # L(X) for μ
  lossμU = Tᵏloss_offUγ(mTᵏ,:μU,η,data(predTᵏ[2][1]),predTᵏ[3][2],γi,Ψμ,z1n,modelP,paramCI,o) # U(X) for μ
  lossσL = Tᵏloss_offLγ(mTᵏ,:σL,η,data(predTᵏ[2][2]),predTᵏ[3][3],γi,Ψσ,z1n,modelP,paramCI,o) # L(X) for σ
  lossσU = Tᵏloss_offUγ(mTᵏ,:σU,η,data(predTᵏ[2][2]),predTᵏ[3][4],γi,Ψσ,z1n,modelP,paramCI,o) # U(X) for σ
  return lossμL + lossμU + lossσL + lossσU
end

## Tᵏ loss for CI offset L(X) for either μ or σ (evaluates loss for one L(X) offset at a time)
function Tᵏloss_offLγ(mTᵏ,γname,η,CX,LX,γL,Ψfun,z1n,modelP,paramCI,o)
  α = paramCI.α
  centLX = data(CX) .- Ψfun(γL)
  qL = quantile(centLX, 1-α/4)
  lossL = ((LX .< qL) .* (LX .- qL).^2) .+ η .* ((LX .> qL) .* (LX .- qL).^2)
  lL = mean(lossL)
  lL2CX = mean(centLX .^ 2)
  # if (o[:verbose]==1 && (o[:epoch]%o[:maxRiskEvery]==0))
  #   println((data(γL) |> cpu, γname, :qL, round(qL,3), :lL, round(data(lL),3), :lL2CX, round(lL2CX,3), :lL_byCX, round(data(lL / lL2CX),3)))
  # end
  return (lL ./ lL2CX)
end

## Tᵏ loss for CI offset U(X) for either μ or σ (evaluates loss for one U(X) offset at a time)
function Tᵏloss_offUγ(mTᵏ,γname,η,CX,UX,γU,Ψfun,z1n,modelP,paramCI,o)
  α = paramCI.α
  centUX = Ψfun(γU) .- data(CX)
  qU = quantile(centUX, 1-α/4)
  lossU = ((UX .< qU) .* (UX .- qU).^2) .+ η .* ((UX .> qU) .* (UX .- qU).^2)
  lU = mean(lossU)
  lL2CX = mean(centUX .^ 2)
  # if (o[:verbose]==1 && (o[:epoch]%o[:maxRiskEvery]==0))
  #   println((data(γU) |> cpu, γname, :qU, round(qU,3), :lU, round(data(lU),3), :lL2CX, round(lL2CX,3), :lU_byCX, round(data(lU / lL2CX),3)))
  # end
  return (lU ./ lL2CX)
end

function sample_noise(atype,udim,nsamples,mu=0.5f0,sigma=0.5f0)
  noise = randn(Float32, udim, nsamples)
  noise = convert(atype, noise)
  noise = (noise .- mu) ./ sigma
  return noise
end

function trainPiᵏcent!(mPiᵏcent,mPiᵣ,mTᵏ,z1n,modelP,paramCI,optPiᵏ,o,atype;printloss::Bool=false)
  g = map(x -> istracked(x) && zero_grad!(grad(x)), params(mPiᵏcent))
  noise = sample_noise(atype,o[:udim],o[:nbatch])
  l = Piᵏlosscent(mPiᵏcent,mPiᵣ,mTᵏ,noise,z1n,modelP,paramCI,o;printloss = printloss)
  Flux.back!(l)
  g_Piᵏcent = map(x -> istracked(x) && grad(x), params(mPiᵏcent))
  g_Piᵏcent = map(x -> maximum(abs.(x)), g_Piᵏcent)
  println("maxΔ for Piᵏcent after backprop:"); println((g_Piᵏcent))
  optPiᵏ()
  g = map(x -> istracked(x) && zero_grad!(grad(x)), params(mPiᵏcent))
  return data(l)
end

function trainPiᵏcent!(mPiᵏcent::UniformPi⁰,mPiᵣ,mTᵏ,z1n,modelP,paramCI,optPiᵏ,o,atype;printloss::Bool=false)
  return nothing
end

## Performs a single training for the estimator Tᵏ(x1n)
## minimize risk R(Piᵏ,Tᵏ) for Tᵏ wrt input data x1n that derives from Piᵏ
function trainTᵏ!(mTᵏ,Pγ,z1n,modelP,paramCI,optTᵏ,o,atype)
  lcent = 0.0f0
  g1 = map(x -> istracked(x) && zero_grad!(grad(x)), params(mTᵏ))
  loff = TᵏlossLxUxback!(mTᵏ,z1n,Pγ,modelP,paramCI,o)
  ΔTᵏoff = map(x -> istracked(x) && grad(x), params(mTᵏ.mTᵏoff))
  maxΔTᵏoff = map(x -> maximum(abs.(x)), ΔTᵏoff)
  optTᵏ[:mTᵏoff]()

  ## **************************************************************
  ## ***** NOTE: VERY IMPORTANT! Set the grads of Tᵏ and Piᵏ to 0 after update (don't want the gradients to bleed)
  ## TODO: Perhaps there is a more elegant way to do this?
  ## **************************************************************
  ## Above call Flux.back!(l) performs backprop on all tracked objects involved,
  ## meaning that the gradients will be calculated and saved for both NNs: mTᵏ and mPiᵏ.
  ## The optTᵏ call below only updates the weights of Tᵏ, which also resets the grads of Tᵏ to 0.
  ## However, the grads for Piᵏ from above loss will be preserved and saved within Piᵏ object.
  ## The next step (inside trainPiᵏ) evaluates the loss for Piᵏ and does the backprop (Flux.back!(l)) on Piᵏ.
  ## The new gradients resulting from that will be added the Piᵏ gradients carried over from this step,
  ## effectivly cancelling each other out ->> Total mess ensures.
  ## To avoid this just manually reset the gradients of Piᵏ to zero, since we don't care about them.
  ## This training step is all about updating Tᵏ not Piᵏ.
  g1 = map(x -> istracked(x) && zero_grad!(grad(x)), params(mTᵏ))
  ## **************************************************************

  lcombo = data(lcent) + data(loff)
  println((
           :lTᵏ_Cx,  round(data(lcent),3),
           :lTᵏ_off,  round(data(loff),3),
           :Tᵏlcombo, round(data(lcombo),3)
           ))

  return data(lcombo)
end

function pretrainPiᵏ(nepochsPinit,o,mPi⁰init,mPi⁰,mTᵏ,modelP,paramCI,atype)
  noise = sample_noise(atype,2,o[:nbatch]); ## lets make some noise!
  z1n = allocX1n(modelP, o[:nbatch]);

  γsampPinit = predictPiᵏpre(mPi⁰init,noise,mTᵏ,z1n,modelP,paramCI); ## sample parameters from the prior NN

  γsampPi = predictPiᵏpre(mPi⁰,noise,mTᵏ,z1n,modelP,paramCI); ## sample parameters from the prior NN

  ## initial Piᵏ for centers
  mPi⁰init = pretrainPiᵣinit(o,mPi⁰init,mTᵏ,z1n,modelP,paramCI,atype;nsteps=nepochsPinit)
  mPi⁰init = map(Flux.param, mPi⁰init)
  mPi⁰ = pretrainPiᵣinit(o,mPi⁰,mTᵏ,z1n,modelP,paramCI,atype;nsteps=nepochsPinit)
  mPi⁰ = map(Flux.param, mPi⁰)

  noise = sample_noise(atype,o[:udim],o[:nbatch])
  γsampPinit = predictPiᵏpre(mPi⁰init,noise,mTᵏ,z1n,modelP,paramCI)
  γsampPi = predictPiᵏpre(mPi⁰,noise,mTᵏ,z1n,modelP,paramCI)

  return deepcopy(mPi⁰)
end

function pretrainPiᵣinit(o,mPi,mTᵏ,z1n,modelP,paramCI,atype=Array{Float32};nsteps=25)
  mPi = map(Flux.param, mPi) ## special tag for Flux to know what to do
  optPiᵣ = Flux.ADAM(params(mPi), 0.001f0, β1=0.9)
  dlossval = 0
  for epoch = 1:nsteps
    noise = sample_noise(atype,o[:udim],o[:nbatch])
    γ = predictPiᵏpre(mPi,noise,mTᵏ,z1n,modelP,paramCI)
    l = -(std(γ[1]) + std(γ[2]))
    @time Flux.back!(l)
    optPiᵣ()
    noise = sample_noise(atype,o[:udim],o[:ntest]);
    if (o[:verbose]==1) println((:epoch, epoch)); end;
  end
  return map(data, mPi) ## for Knet
end

function pretrainPiᵣ(o,mPi,mPi0,mTᵏ,z1n,modelP,paramCI,atype=Array{Float32};nsteps=25)
  optPiᵣ = Flux.ADAM(params(mPi), 0.001f0, β1=0.9)
  nbatch = 500
  dlossval = 0
  for epoch = 1:nsteps
    noise = sample_noise(atype,o[:udim],nbatch)
    γ_MLP = predictPiᵏpre(mPi0,noise,mTᵏ,z1n,modelP,paramCI)
    γ̂ = predictPiᵏpre(mPi,noise,mTᵏ,z1n,modelP,paramCI)
    l = mean((γ̂[1] .- γ_MLP[1]).^2) + mean((γ̂[2] .- γ_MLP[2]).^2)
    println("loss:$(data(l))")
    pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
    println((:Piᵏ_μ, quantile(data(γ̂[1]), pquant)))
    println((:Piᵏ_σ, quantile(data(γ̂[2]), pquant)))
    println((:Pi0_μ, quantile(data(γ_MLP[1]), pquant)))
    println((:Pi0_σ, quantile(data(γ_MLP[2]), pquant)))
    @time Flux.back!(l)
    optPiᵣ()
    if (o[:verbose]==1) println((:epoch, epoch)); end;
  end
  return mPi
end

function pretrainPiᵣ(o,mPi::Flux.Chain,mPi0,mTᵏ,z1n,modelP,paramCI,atype=Array{Float32};nsteps=25)
  optPiᵣ = Flux.ADAM(params(mPi), 0.001f0, β1=0.6)
  dlossval = 0
  for epoch = 1:nsteps
    noise0 = sample_noise(atype,o[:udim],nbatch)
    noise = sample_noise(atype,o[:udim],o[:nbatch])
    γ_MLP = predictPiᵏpre(mPi0,noise,mTᵏ,z1n,modelP,paramCI)
    γ̂ = predictPiᵏpre(mPi,noise,mTᵏ,z1n,modelP,paramCI)
    l = mean((γ̂[1] .- γ_MLP[1]).^2) + mean((γ̂[2] .- γ_MLP[2]).^2)
    println("loss:$(data(l))")
    pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
    println((:Piᵏ_μ_len, quantile(vec(data(γ̂[1])), pquant)))
    println((:Piᵏ_σ_len, quantile(vec(data(γ̂[2])), pquant)))
    @time Flux.back!(l)
    optPiᵣ()
    if (o[:verbose]==1) println((:epoch, epoch)); end;
  end
  return mPi
end

# This allows both non-interactive (shell command) and interactive direct calls to main(...)
splitdir(PROGRAM_FILE)[end] == "maximinNN1.jl" && main(ARGS)

end # module maximinNN1
