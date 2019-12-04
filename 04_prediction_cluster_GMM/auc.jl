# This set of functions is based almost entirely on the
# excellent scikit-learn package's method of calculating
# an ROC's area under the curve (AUC). The functions below
# produce the same output as scikit-learn.



function trapsum(y, x)
    d = diff(x)
    res = sum((d .* (y[2:end] + y[1:(end-1)]) ./ 2.0))
    res
end

# trapsum([1, 2, 3], [4, 6, 8])     # 8.0


function _auc(x, y; reorder = false)
    direction = 1
    if reorder
        order = sortperm2(x, y)
        x, y = x[order], y[order]
    else
        dx = diff(x)
        if any(dx .<= 0)
            if all(dx .<= 0)
                direction = -1
            else
                error("Reordering is not turned on, and the x array is not increasing: $x")
            end
        end
    end
    area = direction * trapsum(y, x)
    area
end




"""
    auc(y_true, y_score)
This function returns the area under the curve (AUC) for the receiver operating characteristic
curve (ROC). This function takes two vectors, `y_true` and `y_score`. The vector `y_true` is the
observed `y` in a binary classification problem. And the vector `y_score` is the real-valued
prediction for each observation.
"""
function auc(y_true::T, y_score::S) where {T <: AbstractArray{<:Real, 1}, S <: AbstractArray{<:Real, 1}}
    if length(Set(y_true)) == 1
        warn("Only one class present in y_true.\n
              The AUC is not defined in that case; returning -Inf.")
        res = -Inf
    elseif length(Set(y_true)) ≠ 2
        warn("More than two classes present in y_true.\n
              The AUC is not defined in that case; returning -Inf.")
        res = -Inf
    else
        xroc = ROC(y_true, y_score)
        res = _auc(xroc.fpr, xroc.tpr, reorder = true)
    end
    res
end

# auc(y, y_score)


struct ROC
    fpr::Array{Float64, 1}
    tpr::Array{Float64, 1}
    thresholds::Array{Float64, 1}

    function ROC(y_true, y_score)
        fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
        fpr = fps/last(fps)
        tpr = tps/last(tps)
        res = new(fpr, tpr, thresholds)
        return res
    end
end


# function roc_curve(y_true, y_score)
#     fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
#     fpr = fps/last(fps)
#     tpr = tps/last(tps)
#     return (fpr, tpr, thresholds)
# end

# roc_curve(y, y_score)


function _binary_clf_curve(y_true, y_score)
    y_true = y_true .== 1       # make y_true a boolean vector
    desc_score_indices = sortperm(y_score, rev = true)

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = find(diff(y_score))
    threshold_idxs = push!(distinct_value_indices, length(y_score))

    tps = cumsum(y_true)[threshold_idxs]
    fps = threshold_idxs - tps
    return (fps, tps, y_score[threshold_idxs])
end

# y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
# y_score = [0.3, 0.2, 0.3, 0.23, 0.5, 0.34, 0.45, 0.54, 0.6, 0.7, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2, 0.6, 0.7, 0.5, 0.2, 0.1, 0.7, 0.2, 0.7, 0.4]

# _binary_clf_curve(y, y_score)

"""
Given 2 vectors, `x` and `y`, this function returns the indices that
sort the elements by `x`, with `y` breaking ties. See the example below.
julia> a = [2, 1, 3, 2]
julia> b = [3, 4, 1, 0]
julia> order = sortperm2(a, b)
4-element Array{Int64,1}:
 2
 4
 1
 3
julia> hcat(a[order], b[order]
4×2 Array{Int64,2}:
 1  4
 2  0
 2  3
 3  1
 """
function sortperm2(x, y; rev = false)
    n = length(x)
    no_ties = n == length(Set(x))
    if no_ties
        res = sortperm(x, rev = rev)
    else
        ord1 = sortperm(x, rev = rev)
        x_sorted = x[ord1]
        i = 1
        while i < n

            # println("x_i is $(x_sorted[i]) and x_(i+1) is $(x_sorted[i+1])")
            if x_sorted[i] == x_sorted[i+1]
                if rev && y[ord1][i] < y[ord1][i+1]
                    #println("(1.) Switching $(y[ord1][i]) with $(y[ord1][i+1])")
                    ord1[i], ord1[i+1] = ord1[i+1], ord1[i]
                    i = i > 1 ? i - 1 : i
                    continue
                elseif !rev && y[ord1][i] > y[ord1][i+1]
                    #println("(2.) Switching $(y[ord1][i]) with $(y[ord1][i+1])")
                    ord1[i], ord1[i+1] = ord1[i+1], ord1[i]
                    i = i > 1 ? i - 1 : i
                    continue
                end
            end
            i += 1
        end
        res = ord1
    end
    res
end

# a = [1, 5, 1, 4, 3, 4, 4, 3, 1, 4, 5, 3, 5]
# b = [9, 4, 0, 4, 0, 2, 1, 2, 1, 3, 2, 1, 1]
#
# ord = sortperm2(a, b, rev = true)
# hcat(a[ord], b[ord])
#
# ord = sortperm2(a, b)
# hcat(a[ord], b[ord])
