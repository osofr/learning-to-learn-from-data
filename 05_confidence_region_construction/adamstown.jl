## Hacking Flux Adam optimizer to be able to save the optimizer state when using GPUs

using Flux.Tracker: TrackedArray

mutable struct ParamAdam{T}
  x::T
  Δ::T
  mt::T
  vt::T
  β1p::Real
  β2p::Real
end

# Flux.treelike(ParamAdam)

ParamAdam(x::AbstractArray) = ParamAdam(x, zero(x), zero(x), zero(x), 0.0, 0.0)
ParamAdam(x::TrackedArray) = ParamAdam(x.data, x.grad, zero(x.data), zero(x.data), 0.0, 0.0)

gpu(ps::ParamAdam) = ParamAdam(ps.x |> gpu, ps.Δ |> gpu, ps.mt |> gpu, ps.vt |> gpu, ps.β1p, ps.β2p)
cpu(ps::ParamAdam) = ParamAdam(ps.x |> cpu, ps.Δ |> cpu, ps.mt |> cpu, ps.vt |> cpu, ps.β1p, ps.β2p)

call(f, xs...) = f(xs...)

## init optimizer state
initAdam(ps) = [ParamAdam(p) for p in ps]

## overwrite optimizer state with new adam params
# loadparams!(oldps,newps) = map((oldps,newps) -> (oldps.mt .= newps.mt; oldps.vt .= newps.vt), oldps, newps)
loadparams!(oldps::ParamAdam,newps::ParamAdam) = (oldps.mt .= newps.mt;
                                                  oldps.vt .= newps.vt;
                                                  oldps.β1p = newps.β1p;
                                                  oldps.β2p = newps.β2p)

# note for optimisers: set to zero; p.Δ at the end of the weights update
function optimiserAdam(ps, fs...)
  fs = map(ps) do p
    os = map(f -> f(p), fs)
    () -> foreach(call, os)
  end
  () -> foreach(call, fs)
end

ADAM(ps, η = 0.001; β1 = 0.9, β2 = 0.999, ϵ = 1e-08, decay = 0) =
  optimiserAdam(ps, p->adam(p; η=η, β1=β1, β2=β2, ϵ=ϵ), p->descent(p, 1))
  # optimiser(ps, p->adam(p; η=η, β1=β1, β2=β2, ϵ=ϵ), p->invdecay(p,decay), p->descent(p,1))

function adam(p::ParamAdam; η::Real = 0.001, β1::Real = 0.9, β2::Real = 0.999, ϵ::Real = 1e-8)
  mt = p.mt # mt = zero(p.x)
  vt = p.vt # vt = zero(p.x)
  p.β1p, p.β2p = β1, β2
  # println("init mt adam: "); println(mt)
  # println("init vt adam: "); println(vt)

  function ()
    # println("mt inside pre: "); println(mt)
    # println("vt inside pre: "); println(vt)
    @. mt = β1 * mt + (1 - β1) * p.Δ
    @. vt = β2 * vt + (1 - β2) * p.Δ^2
    @. p.Δ =  mt / (1 - p.β1p) / √(vt / (1 - p.β2p) + ϵ) * η
    p.β1p *= β1
    p.β2p *= β2
    # println("mt inside aft: "); println(mt)
    # println("vt inside aft: "); println(vt)
  end
end

function descent(p::ParamAdam, η::Real)
  function ()
    @. p.x -= η * p.Δ
    @. p.Δ = 0
  end
end
