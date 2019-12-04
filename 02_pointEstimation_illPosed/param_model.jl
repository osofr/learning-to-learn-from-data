## ---------------------------------------------------------------------------------------
## Classes that define an object for parmetric models.
## This includes the number of parameters in the model, the bounds of the parameter space,
## the definition of the true parameter (target of estimation) and functionality for
## simulating data from a given parametric model.
## ---------------------------------------------------------------------------------------
## Alternative solution is to define a separate struct for the parameter
## Alternative is to separately pass a function Ψ₀, that takes in (ParamModelPᵏ) or define a new parameter struct:
    # struct Parameter
    #   Ψ₀::Function
    # end
using MacroTools: @forward

## ---------------------------------------------------------------------------------------
## Def bounds on the parameter space
## ---------------------------------------------------------------------------------------
struct ParamBox{T}
    bounds::Vector{T}
end

@forward ParamBox.bounds Base.getindex, Base.first, Base.last, Base.endof, Base.push!, Base.length
@forward ParamBox.bounds Base.start, Base.next, Base.done

bounds(γ::ParamBox, idx::Vector) = map((vec, idx) -> vec[idx], γ, idx)
bounds(γ::ParamBox, idx::Int) = bounds(γ, fill(idx, length(γ)))
lb(γ::ParamBox) = bounds(γ,1)
ub(γ::ParamBox) = bounds(γ,2)
mid(γ::ParamBox) = (lb(γ) + ub(γ)) / 2
border(γ::ParamBox) = [lb(γ),
                       ub(γ),
                       mid(γ),
                       bounds(γ, rand([1,2], 3)),
                       bounds(γ, rand([1,2], 3)),
                       bounds(γ, rand([1,2], 3)),
                       bounds(γ, rand([1,2], 3))
                       ]

## parameter space transformations
import Base.linspace
linspace(γ::ParamBox;step=0.01) = map(x -> x[1]:eltype(γ[1])(step):x[2], γ.bounds)
lintrans(x, lb, ub) = x*(ub-lb)+lb

import Base.rand
function rand(γ::ParamBox, nvals; step=0.01)
    range = linspace(γ,step=step)
    x = map(x -> rand(x,nvals), range)
    x = hcat(x...)
    return [x[i,:] for i=1:nvals]
end

## ---------------------------------------------------------------------------------------
## Def model / parameter
## ---------------------------------------------------------------------------------------
abstract type ModelPᵏ end
abstract type NonParamModelPᵏ <: ModelPᵏ end

struct ParamModelPᵏ{T} <: ModelPᵏ
    bounds::ParamBox{T}
    Ψ::Symbol           # symbol for function name that evalutes the truth (under same name)
    name::Symbol        # name of the data-gen fun for simulating 𝐗
    xdim::Int           # dimension of input data 𝐗 ↦ T(𝐗), for sample 𝐗=(Xᵢ:i=1,…,n)
    l2constraint::Float32        # l2 equality constraint on parameters
end
@forward ParamModelPᵏ.bounds bounds, lb, ub, mid, border, linspace
@forward ParamModelPᵏ.bounds Base.getindex, Base.first, Base.last, Base.endof, Base.push!, Base.length
@forward ParamModelPᵏ.bounds Base.start, Base.next, Base.done

import Base.rand
function rand(modelP::ParamModelPᵏ, nvals; step=0.01)
    # l2 ball
    randtype = Array{Float32}
    x = convert(randtype,randn(Float32,nvals,length(modelP))) # generate gaussian random variable
    x = [(modelP.l2constraint).*x[i,:]./sqrt(sum(x[i,:].*x[i,:])) for i=1:nvals] # standardize to get a draw from l2 ball
    return x
end

# NOTE: this function assumes that the first coordinate is being used to relax an equality constraint
#    on the l2 norm of the parameter to an inequality constraint. Therefore, this first coordinate is omitted
#    when simulating from the sphere, and then is added back at the end so that the l2 constraint is satisfied
function unifRand(modelP::ParamModelPᵏ, nvals; step=0.01)
    # l2 ball
    randtype = Array{Float32}
    x = convert(randtype,randn(Float32,nvals,length(modelP)-1)) # generate gaussian random variable
    r = convert(randtype,sqrt.(rand(Float32,nvals))).*modelP.l2constraint # radius
    x = [vcat(sqrt(modelP.l2constraint^2-r[i]^2),r[i].*x[i,:]./sqrt(sum(x[i,:].*x[i,:]))) for i=1:nvals] # standardize to get a draw from l2 ball
    return x
end

## define constructor that ensures type consistency (convert each bound to same types as params)
function ParamModelPᵏ(bounds::ParamBox{T}; Ψ::Symbol=Ψμ,name::Symbol=:norm,xdim::Int=10,l2constraint::Float32=1f0) where T
    ParamModelPᵏ(bounds,Ψ,name,xdim,l2constraint)
end

## evaluate the truth for any new parameter paramγ
function Ψ(modelP::ParamModelPᵏ, paramγ::Any)
    getfield(maximinNN1, modelP.Ψ)(paramγ)
end

## clip parameter vector to stay within the bounds, by ref
## only works when γ is a single parameter value (not a vector of params)
function clip!(γ, modelP::ParamModelPᵏ)
   γ .= min.(ub(modelP), max.(lb(modelP), γ))
   return γ
end
## re-allocate γ (use for autodiff or GPU)
function clip(γ, modelP::ParamModelPᵏ)
   γ = min.(ub(modelP), max.(lb(modelP), γ))
   return γ
end

function allocX1n(modelP::ParamModelPᵏ, nbatch::Int; atype=Array{Float32})
    convert(atype, randn(Float32, modelP.xdim, nbatch))
end
