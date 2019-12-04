# module SGAhardestPk
using Knet
# mse(ŷ,y) = (sum(abs2, y-ŷ) / length(ŷ))
# const d = 10::Int ## number of observations in a sample

## find the hardest Pᵏ for given estimator mTᵏ by running SGA with 10 random starts for γ
function findPᵏ(mTᵏ::Function, modelP::ParamModelPᵏ, hiddenReg...; atype=Array{Float32}, opt=Sgd, lr=0.001, nbatch=100, nruns=200, nstarts=10, ntest=1000, allOut=false, nstartsBorder=0, nMC=100, arrayOut=false, xinit = [],nstartsMixed=0)
    # T = Float32
    #lr=T(lr)
#    modelP_lb = convert(atype,lb(modelP))
#    modelP_ub = convert(atype,ub(modelP))
    #if length(border(modelP))<=nstarts
    #	xinit = [border(modelP); rand(modelP, nstarts-size(border(modelP),1))]
    #else
    uniformStarts = rand(modelP,nstarts)
    borderStarts = [bounds(modelP, rand([1,2], length(lb(modelP)))) for i=1:nstartsBorder]
    if xinit==[]
        if nstartsMixed==0
            if nstartsBorder==0
                xinit = uniformStarts
            else
                xinit = [borderStarts; uniformStarts]
            end
        else
            mixedBoundaryStarts = [bounds(modelP, rand([1,2], length(lb(modelP)))) for i=1:nstartsMixed]
            mixedUniformStarts = rand(modelP,nstartsMixed)
            mixedStarts = [mixedBoundaryStarts[i] .+ (rand(Float32,length(mixedUniformStarts[i])).*(mixedUniformStarts[i]-mixedBoundaryStarts[i])) for i=1:nstartsMixed]
            if nstartsBorder==0
                xinit = [mixedStarts; uniformStarts]
            else
                xinit = [mixedStarts; borderStarts; uniformStarts]
            end
        end
        # end
        xinit = convert.(atype,xinit)
    end
    #xinit = convert.(atype,rand(modelP,nstarts))
    #f = function(xx)
#	xx = (xx.-modelP_lb)./(modelP_ub.-modelP_lb)
#	xx = log.(xx./(1.-xx))
#    end
#    xinit = broadcast(f,xinit)
    # maxRisks = Vector{T}(length(xinit))
    getMaxRisk = function(candOptVal)
        num_param = length(xinit[1])
        z1n = allocX1n(modelP, ntest; atype=atype)
        γ_curr = makeRegNN(reshape(candOptVal,num_param,1),hiddenReg...;atype=atype, wdim=modelP.odim-1)
        #psi0 = Ψ(modelP, γ_curr)
        z1n = sim(modelP, γ_curr, z1n; diffγ=false)
        Tᵏ = mTᵏ(z1n)
        crossentropy2(Tᵏ, γ_curr, [], modelP; atype=atype, nMC=nMC)
    end

    #initRisks = broadcast(i->getMaxRisk(xinit[i]),1:length(xinit))
    optvals = optimizeSGA(mTᵏ, xinit, modelP, hiddenReg...; atype=atype, opt=opt, lr=lr, nbatch=nbatch, nruns=nruns)
    #maxRisks = broadcast(i->getMaxRisk(optvals[:,i:i]),1:length(xinit))


    if allOut
        return makeRegNN(optvals,hiddenReg...;atype=atype, wdim=modelP.odim-1)
    else        
        if size(optvals,2)>1
            #initRisks = broadcast(i->getMaxRisk(xinit[i]),1:length(xinit))
            maxRisks = broadcast(i->getMaxRisk(optvals[:,i:i]),1:length(xinit))
    #println(maxRisks)
            # maxRisk = maximum(maxRisks)
            indmaxRisk = indmax(maxRisks)
        else
            indmaxRisk = 1
        end
        maxγ = optvals[:,indmaxRisk:indmaxRisk]
        maxγArray = maxγ

        # get an unbiased estimate of the maximum risk via MC at the hardest γ:
        z1n = allocX1n(modelP, 2*ntest; atype=atype)
        maxγ = makeRegNN(maxγ,hiddenReg...;atype=atype, wdim=modelP.odim-1)
        #println(typeof(maxγ))
        #println(typeof(z1n))
        #println(typeof(maxγ[1]))
        z1n, weights = sim(modelP,maxγ,z1n)
        Tᵏ = mTᵏ(z1n)
        ψ = Ψ(modelP, maxγ)
        # maxRisk = mse(Tᵏ, ψ, weights)

        # println("=========")
        # println(length(maxRisks))
        # println(maxRisk)

        maxRisk = crossentropy2(Tᵏ, ψ, weights, modelP; atype=atype, nMC=nMC)
        # println(maxRisk)
        if arrayOut
            return maxRisk, maxγ, maxγArray
        else
            return maxRisk, maxγ
        end
    end
end

## SGA-based inner optimization routine to find the hardest Pᵏ (using Knet w/ adam optimizer)
function optimizeSGA(mTᵏ, xinit, modelP::ParamModelPᵏ, hiddenReg...; atype=Array{Float32}, opt=Adam, lr=0.001, nbatch=500, nruns=200)# where T
    #lossval = T(0.0)
    modelP_lb = convert(atype,lb(modelP))
    modelP_ub = convert(atype,ub(modelP))
    num_param = length(xinit[1])
    num_init = length(xinit)
    γ = convert(atype, hcat(xinit...))
    optPᵏ = optimizers(γ, opt; lr=lr)
    z1n = allocX1n(modelP, nbatch; atype=atype)
    function SGAlossPᵏ(γ,z1n)
        out = 0
        for i=1:size(γ,2)
	    γ_curr = makeRegNN(γ[:,i:i],hiddenReg...;atype=atype, wdim=modelP.odim-1)
#            γ_curr = makeRegNN(Knet.sigm.(γ[:,i:i]).*(modelP_ub.-modelP_lb) .+ modelP_lb,hiddenReg...;atype=atype, wdim=modelP.odim-1)
            z1n, weights = sim(modelP, γ_curr, z1n)
            Tᵏ = mTᵏ(z1n)
            out += crossentropy2(Tᵏ, γ_curr, weights, modelP; atype=atype)
        end
        return out/size(γ,2)
    end

    @inbounds for j in 1:nruns
        γ = convert(atype, γ)
       Pᵏlossgrad = gradloss(SGAlossPᵏ)
       gradients, lossval = Pᵏlossgrad(γ,z1n)
    #gradients = gradients.+convert(atype,0.1.*randn(length(gradients)))
#gradients = gradients./sqrt.(sum(gradients.*gradients))
#println(max.(convert(Array{Float32,2},gradients)))
if(mod(j,50)==0)
   println("===========")
   println(j)
   println(lossval)
end
#optPᵏ = optimizers(γ, opt; lr=lr/j^(3/4))
        update!(γ, gradients, optPᵏ)
    #γ = reshape(γ,num_param,num_init)
    for i in 1:size(γ,2)
        γ[:,i] = min.(max.(γ[:,i], modelP_lb),modelP_ub)
    end
    #γ = reshape(γ,num_param*num_init)
    end
    #γ = reshape(γ,num_param,num_init)

#    for i in 1:size(γ,2)
#        γ[:,i] = Knet.sigm.(γ[:,i]).*(modelP_ub.-modelP_lb) .+ modelP_lb
#    end
    return γ
end

