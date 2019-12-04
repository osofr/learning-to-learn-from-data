
function riskgrid(o,nets,modelP,μgrid::AbstractVector{T},σgrid::AbstractVector{T};atype = Array{Float32}) where T
    nmc = o[:ntest]
    mTᵏ = deepcopy(nets[:mTᵏ])
    mTᵏ = convertNN(mTᵏ)
    modelTᵏ(z1n) = predictTᵏ(mTᵏ,z1n)
    riskmat = SharedArray{T}(length(μgrid), length(σgrid));
    @time @sync @parallel for μidx = 1:length(μgrid)
        for σidx = 1:length(σgrid)
            γ = [μgrid[μidx], σgrid[σidx]]
            z1n = allocX1n(modelP,nmc;atype=atype)
            z1n = sim!(modelP,γ,z1n)
            Tᵏ = modelTᵏ(z1n)
            ψ = Ψ(modelP,γ)
            riskmat[μidx, σidx] = mse(Tᵏ,ψ)
        end
    end
    return riskmat
end
