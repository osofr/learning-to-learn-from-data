## simulating new data (an alternative strategy: create a closure for sim() and Ψ())
const simdct = Dict(:norm=>:sim_norm!, :lognorm=>:sim_lognorm!)

function sim!(modelP::ParamModelPᵏ, γ, x1n)
    funcall = getfield(maximinNN1, simdct[modelP.name])
    funcall(γ, x1n)
end

## simulate a vector/matrix of normals, each distributed as μ=γ[1][i], σ=γ[2][i] from given vector of parameters
function sim_norm!(γ, z1n)
    randn!(z1n)
    z1n = (z1n' .* γ[2] .+ γ[1])'
end

## simulate a vector/matrix of lognormals
function sim_lognorm!(γ, x1n)
    exp.(sim_norm!(γ, x1n))
end

## simulate vectror/matrix of uniforms of spec'ed range
function sim_unif!(u1n,lb,ub)
    rand!(u1n) .* (ub.-lb).+lb
end
function sim_unif(nbatch,lb,ub)
    rand(nbatch) .* (ub.-lb).+lb
end
