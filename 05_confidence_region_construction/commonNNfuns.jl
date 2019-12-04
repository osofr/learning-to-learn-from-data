## ------------------------------------------------
## Cost functions
## ------------------------------------------------
# mse(ŷ, y) = sum((ŷ .- y).^2) / length(ŷ)
mse(ŷ,y) = (sum(abs2, y-ŷ) / length(ŷ))
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

## ------------------------------------------------
## kernels for approximating indicator functions
## ------------------------------------------------
## fourth-order Epanechnikov kernel
# function epkern4(t::T,h = oftype(t/1, 0.01)) where {T}
function epkern4(t,h = 0.01f0)
    t = t/T(h)
    if t < -1
        zero(t)
    elseif -1 <= t <= 1
        one(t)/32 * (21t^5 - 50t^3 + 45t + 16)
    else
        one(t)
    end
end
## sigmoid kernel
sigmkern(t,h = 0.01) = Knet.sigm(t/h)
# sigmkern(t::T,h = oftype(t/1, 0.01)) where {T} = Knet.sigm(t/T(h))
# sigmkern(t::T,h = oftype(t/1, 0.01)) where {T<:Union{Float32,Float64}} = Knet.sigm(t/T(h))

## sixth-order Epanechnikov kernel (for less conservative approximation of coverage)
# function epkern6B(t,h=0.01f0)
#     t = t/h
#     γ = min(1.0f0, max(0.0f0, 1.0f0 / 256 * (-495t^7 + 1323t^5 - 1225t^3 + 525t + 128)))
# end

function epkern6B(t,h=0.1f0)
    t = t/h
    # (t < -1 ? 0 : (t<1) ? (1 / 256 * (-495t^7 + 1323t^5 - 1225t^3 + 525t + 128)) : 1)
    (t > 0 ? 1 : t < 0 ? -1:0)
end

# function epkern6(t,h=0.01f0)
function epkern6(t::T,h = oftype(t/1, 0.01))::T where {T}
    t = t/T(h)
    # t = t/h
    if t < -1
        zero(t)
        # 0.0f0
    elseif -1 <= t <= 1
        1/256 * (-495t^7 + 1323t^5 - 1225t^3 + 525t + 128)
        # one(t)/256 * (-495t^7 + 1323t^5 - 1225t^3 + 525t + 128)
        # 1.0f0 / 256 * (-495t^7 + 1323t^5 - 1225t^3 + 525t + 128)
        # 1.0f0 / 256.0f0
    else
        one(t)
        # 1.0f0
    end
end

function epkern6cu(t::T,h = oftype(t/1, 0.01))::T where {T}
    # println("h:$h")
    t = t/T(h)
    if t < -1
        zero(t)
    elseif -1 <= t <= 1
        # 1/256 * (-495(t^3 * t^3 * t) + 1323(t^3 * t^2) - 1225t^3 + 525t + 128)
        1/256 * (-495(t^2 * t^2 * t^2 * t) + 1323(t^2 * t^2 * t) - 1225(t^2 * t) + 525t + 128)
    else
        one(t)
    end
end

# using UnicodePlots, Knet, Flux
# myPlot = lineplot(x -> epkern(x,0.1), -0.1, 0.1)
# myPlot = lineplot(x -> sigmkern(x,0.1), -0.1, 0.1)
# myPlot = lineplot(x -> epkern(x,0.05), -0.1, 0.1)
# myPlot = lineplot(x -> sigmkern(x,0.05), -0.1, 0.1)
# myPlot = lineplot(x -> epkern(x,0.01), -0.1, 0.1)
# myPlot = lineplot(x -> sigmkern(x,0.01), -0.1, 0.1)
