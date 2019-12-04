## ------------------------------------------------
## Cost functions
## ------------------------------------------------
mse(ŷ,y) = (sum(abs2, y-ŷ) / length(ŷ))

weightedMse(ŷ,y,weight) = sum((ŷ .- y) .* (ŷ .- y) .* weight) / length(ŷ)

function crossentropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat; weight = 1)
	return -sum(y .* log_fast.(ŷ) .* weight) / size(y, 2)
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
