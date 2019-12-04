## ------------------------------------------------
## Cost functions
## ------------------------------------------------
# mse(ŷ, y) = sum((ŷ .- y).^2) / length(ŷ)
mse(ŷ,y) = (sum(abs2, y-ŷ) / length(ŷ))

weightedMse(ŷ,y,weight) = sum((ŷ .- y) .* (ŷ .- y) .* weight) / length(ŷ)

function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
	return -sum(y .* log_fast.(ŷ) .* weight) / size(y, 2)
end

function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
	return -sum(y .* log_fast.(ŷ) .* weight) / size(y, 2)
end

## classification loss for Tᵏ based on predicted class for each observation Xᵢ (TX) and latent true class C₀
function classloss(TX, C₀, μ1, μ2)
	## dim of TX and C₀ is [xdim, nbatch]
	## dim of μ1, μ2 is nbatch
	## (matrix x vector) op requires transposing TX
	TXt = TX'
	C₀t = C₀'

	## errors on individual class predictions (for each TXᵢ)
	lossᵢ1 = ( TXt .* μ1 .- (1 .- C₀t) .* μ1 ) .+ ( (1 .- TXt) .* μ2 .- C₀t .* μ2 )
	lossᵢ2 = ( (1 .- TXt) .* μ1 .- (1 .- C₀t) .* μ1 ) .+ ( TXt .* μ2 .- C₀t .* μ2 )

	## sum each row to obtain errors for each batch (loss is transposed, should result in nbatch vectors)
	loss1 =  sum(lossᵢ1 .* lossᵢ1, 2) ./ size(C₀,1)
	loss2 =  sum(lossᵢ2 .* lossᵢ2, 2) ./ size(C₀,1)
	loss = sum(min.(loss1, loss2)) / size(C₀,2)
	return loss
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
