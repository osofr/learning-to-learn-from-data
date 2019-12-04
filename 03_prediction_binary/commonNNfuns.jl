## ------------------------------------------------
## Cost functions
## ------------------------------------------------
function mse(ŷ,y,weights)
	(sum(abs2, weights .* (y-ŷ)) / length(ŷ))
end

function crossentropy(ŷ, y)
	return -sum((y .* log.(ŷ)) .+ ((1.-y) .* log.(1.-ŷ))) / length(y)
end

function logitcrossentropy(logŷ::AbstractVecOrMat, y::AbstractVecOrMat)
	logŷ = logŷ .- maximum(logŷ, 1)
	ypred = logŷ .- log_fast.(sum(exp.(logŷ), 1))
	-sum(y .* ypred) / size(y, 2)
end

## Normalise each column of `x` to mean 0 and standard deviation 1.
function normalise(x::AbstractVecOrMat)
	μ′ = mean(x, 1)
	σ′ = std(x, 1, mean = μ′)
	return (x .- μ′) ./ σ′
end



function crossentropy2(ŷ,y,weights,modelP; atype=atype,nMC=100,varOut=false)
	z1n = allocX1n(modelP, length(y[1]), nMC; atype=atype)
	x1n = sim(modelP, y, z1n; diffγ=false)
	est = cond_mean(ŷ,x1n[1])
	if weights==[]
		if varOut
			losses = (x1n[3]) .* log.(est./x1n[3]) .+ (1.-x1n[3]) .* log.((1.-est)./(1.-x1n[3]))
			risk = -mean(losses)
			secondMoment = mean(losses.*losses)
			riskVar = secondMoment - risk^2
			return risk, riskVar
		else
			return -mean((x1n[3]) .* log.(est./x1n[3]) .+ (1.-x1n[3]) .* log.((1.-est)./(1.-x1n[3])))
		end
	else
		return -mean(mean((x1n[3]) .* log.(est./x1n[3]) .+ (1.-x1n[3]) .* log.((1.-est)./(1.-x1n[3])),1)[1,:].*weights)
	end
end



function makeRegNN(wgt_mat,h...; atype=Array{Float32}, wdim=1)
    w = Any[]
    ind = 1
    if ([h...] == [0]) | ((h...)[1] ==[0])
    	loopover = [1]
    else
    	loopover = vcat(h...,[1])
    end
    for nextd in loopover
        next_ind = ind+nextd*wdim
        push!(w, [reshape(wgt_mat[ind:(next_ind-1),i],nextd,wdim) for i in 1:size(wgt_mat,2)])
        ind = next_ind
        next_ind = ind+nextd
        push!(w, [reshape(wgt_mat[ind:(next_ind-1),i],nextd,1) for i in 1:size(wgt_mat,2)])
        ind = next_ind
        wdim = nextd
    end
    return w
end

