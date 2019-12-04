
randomSearch = function(o,predictFun,modelP,atype;verbose=false)
    interrogationngrid_total = o[:interrogationngrid] + o[:interrogationngridBorder]

    mRNN(γArray) = makeRegNN(γArray,o[:hiddenReg]...;atype=atype, wdim=modelP.odim-1)

    getMaxRisk = function(γ;m=5000)
        z1n = allocX1n(modelP, m; atype=atype)
        z1n = sim(modelP, γ, z1n; diffγ=false)
        Tᵏ = predictFun(z1n)
        crossentropy2(Tᵏ, γ, [], modelP; atype=atype, nMC=1000)
    end

    modelP_lb = vcat(convert(atype,lb(modelP)))
    modelP_ub = vcat(convert(atype,ub(modelP)))

    # initialize in the middle of the parameter space
    currHardestγArray = modelP_lb .+ (modelP_ub-modelP_lb).*convert(atype,vcat(rand(Float32,size(modelP_lb,1),1)))

    # Random search procedure
    for j=0:149
        rectMaxWidths = (modelP_ub .- modelP_lb).*(0.95^j)./2
        rectLBs = max.(modelP_lb,currHardestγArray.-rectMaxWidths)
        rectUBs = min.(modelP_ub,currHardestγArray.+rectMaxWidths)

        newInits = hcat([rectLBs for _=1:interrogationngrid_total]...) + hcat([rectUBs.-rectLBs for _=1:interrogationngrid_total]...).*convert(atype,hcat(rand(Float32,size(currHardestγArray,1),o[:interrogationngrid]),(rand(Float32,size(currHardestγArray,1),o[:interrogationngridBorder]).>0.5)))
        newInits = hcat(newInits,currHardestγArray)

        maxRisks = broadcast(i->getMaxRisk(mRNN(newInits[:,i:i]),m=2500 + 50*(j+1)),1:size(newInits,2))
        currHardestγArray = newInits[:,indmax(maxRisks):indmax(maxRisks)]
        if verbose
            biasedUpMR = maximum(maxRisks)
            println((:biasedUpMR,biasedUpMR))
        end
    end
    return mRNN(currHardestγArray)
end