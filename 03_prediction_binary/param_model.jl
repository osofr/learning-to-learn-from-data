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
                       bounds(γ, rand([1,2], length(lb(γ)))),
                       bounds(γ, rand([1,2], length(lb(γ)))),
                       bounds(γ, rand([1,2], length(lb(γ)))),
                       bounds(γ, rand([1,2], length(lb(γ))))
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
# abstract type ParamModelPᵏ <: NonParamModelPᵏ end

struct ParamModelPᵏ{T} <: ModelPᵏ
    bounds::ParamBox{T}
    Ψ::Symbol           # symbol for function name that evalutes the truth (under same name)
    name::Symbol        # name of the data-gen fun for simulating 𝐗
    n::Int              # number of observations
    odim::Int           # dimension of observation 𝐗=(Oᵢ:i=1,…,n)
end
@forward ParamModelPᵏ.bounds bounds, lb, ub, mid, border, linspace, rand
@forward ParamModelPᵏ.bounds Base.getindex, Base.first, Base.last, Base.endof, Base.push!, Base.length
@forward ParamModelPᵏ.bounds Base.start, Base.next, Base.done

## define constructor that ensures type consistency (convert each bound to same types as params)
function ParamModelPᵏ(bounds::ParamBox{T}; Ψ::Symbol=Ψμ,name::Symbol=:norm,n::Int=10,odim::Int=2) where T
    ParamModelPᵏ(bounds,Ψ,name,n,odim)
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

# γ.*max.(ell2(γ)./sqrt.(sum.(γ.^2)),1)

# this version of the function is called in the maximin file
function allocX1n(modelP::ParamModelPᵏ, nbatch::Int; atype=Array{Float32})
    allocX1n(modelP, nbatch, modelP.n; atype=atype)
end

# this one is called in the crossentropy2 function
function allocX1n(modelP::ParamModelPᵏ, nbatch::Int, n::Int; atype=Array{Float32})
    [[convert(atype, randn(Float32, n, modelP.odim-1)) for _=1:nbatch], convert(atype,randn(Float32, n, nbatch)), convert(atype,randn(Float32, n, nbatch))] # [covariate vector of matrices, matrix of outcomes, matrix of expected outcomes]
end

# ##TODO: Define method for modelP(x,...)
# ## This can do a variety of things a) simulating data?; b) evaluating the true param?
# ## For simulating data, this could return an anonymous function, depending on the model type
# ## this function would already be specialized to the distribution type (:name)
# function (p::ParamModelPᵏ)(x)
#  ... do something useful...
# end
