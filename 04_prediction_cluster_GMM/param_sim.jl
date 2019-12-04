## simulating new data (an alternative strategy: create a closure for sim() and Ψ())
const simdct = Dict(:norm=>:sim_norm!, :lognorm=>:sim_lognorm!, :neal=>:sim_neal!)

## simulate a vector/matrix of normals, each distributed as μ=γ[1][i], σ=γ[2][i] from given vector of parameters
function sim_norm!(γ, z1n)
    randn!(z1n)
    z1n = (z1n' .+ γ)'
    return z1n
end
## simulate a vector/matrix of lognormals
function sim_lognorm!(γ, x1n)
    exp.(sim_norm!(γ, x1n))
end
