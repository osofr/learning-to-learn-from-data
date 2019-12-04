## simulating new data (an alternative strategy: create a closure for sim() and Ψ())
const simdct = Dict(:logit=>:sim_logit,:logit_cd4=>:sim_logit_cd4,:logit_rho3=>:sim_logit_rho3,:logit_rho6=>:sim_logit_rho6,:logit_rho9=>:sim_logit_rho9)

function sim(modelP::ParamModelPᵏ, γ, x1n; diffγ=true)
    funcall = getfield(maximinNN1, simdct[modelP.name])
    funcall(γ, x1n; diffγ=diffγ)
end

# ## simulate a vector/matrix of normals, each distributed as μ=γ[1][i], σ=γ[2][i] from given vector of parameters
# function sim_norm!(γ, z1n)
#     randn!(z1n)
#     z1n = (z1n' .* γ[2] .+ γ[1])'
# end
# ## simulate a vector/matrix of lognormals
# function sim_lognorm!(γ, x1n)
#     exp.(sim_norm!(γ, x1n))
# end

## simulate a simple linear-logistic regression
# function sim_logit!(γ, z1n)
# 	z1n = [(convert(typeof(z1n),randn(size(z1n[1:div(size(z1n,1),2),:])))'.*(1.+0.*γ[1]))';sigm.(-(z1n[1:div(size(z1n,1),2),:]' .* γ[2] .+ γ[1])')]
# return z1n
# end

# cond_mean(γ,Wz1n) = convert(eltype(γ[2]),0.8).*sigm.(-(Wz1n' .* γ[2] .+ γ[1])').+convert(eltype(γ[2]),0.1)

#updateNetRelu(w1,w2,currNodes) = broadcast((a,b,c)->Knet.relu.((a*b).+c),w1,currNodes,w2)
updateNetRelu = function(w1,w2,currNodes)
	f = function(a,b,c)
		tmp = (a*b).+c
		return Knet.tanh.(tmp)
	end
	return broadcast(f,w1,currNodes,w2)
end
#updateNetSigm(w1,w2,currNodes) = broadcast((a,b,c)->convert(eltype(w1[1]),0.1).+convert(eltype(w1[1]),0.8).*Knet.sigm.((a*b).+c),w1,currNodes,w2)
updateNet(w1,w2,currNodes) = broadcast((a,b,c)->(a*b).+c,w1,currNodes,w2)

function cond_mean(w,Wz1n)
	if length(w)>2
		curr = updateNetRelu(w[1],w[2],broadcast(x->x',Wz1n))
		if length(w)>4
		    for j=3:2:length(w)-2
		            curr = updateNetRelu(w[j],w[j+1],curr)
		    end
		end
		#curr = updateNetSigm(w[end-1],w[end],curr)
		curr = convert(eltype(w[1][1]),0.1).+convert(eltype(w[1][1]),0.8).*Knet.sigm.(vcat(updateNet(w[end-1],w[end],curr)...)')
	else
		#curr = updateNetSigm(w[1],w[2],broadcast(x->x',Wz1n))
		curr = convert(eltype(w[1][1]),0.1).+convert(eltype(w[1][1]),0.8).*Knet.sigm.(vcat(updateNet(w[1],w[2],broadcast(x->x',Wz1n))...)')
	end
    return curr
end



function sim_logit(γ, z1n; diffγ=true)
    # covariates
    for i=1:length(z1n[1])
    	z1n[1][i] = convert(typeof(z1n[1][i]),randn!(z1n[1][i]))
    end
    ylogx(y,x) = y.*log.(max.(x,0.0001))
    if(length(γ[1])==1)
	z1n[3] = reshape(cond_mean(γ,[vcat(z1n[1]...)]),size(z1n[1][1],1),length(z1n[1]))
    else
	z1n[3] = cond_mean(γ,z1n[1])
    end
    # NOTE: This is a known bug. By simulating outcomes in this way, we actually are simulating outcomes with probability
    # 	    1-z1n[2], rather than with probability z1n[2]. Due to the symmetry of the cross-entropy loss, the procedure is
    # 	    therefore learning to estimate one minus the conditional expectation, and so the final predictions can be corrected
    # 	    by simply subtracting them from 1.
    z1n[2] = convert(typeof(z1n[2]),convert(Array{Float32,2},z1n[3]).<rand(Float32,size(z1n[3],1),size(z1n[3],2)))
    if diffγ
		weights = sum(ylogx(z1n[2],z1n[3]) + ylogx(1.-z1n[2],1.-z1n[3]),1)
		weights = exp.(weights .- getval(weights))
		return z1n, weights
    else
    	return z1n
    end
end

function sim_logit_cd4(γ, z1n; diffγ=true)
    # covariates
    for i=1:length(z1n[1])
    	z1n[1][i] = convert(typeof(z1n[1][i]),randn!(z1n[1][i]))
    	z1n[1][i][:,1:1] = convert(typeof(z1n[1][i][:,1:1]),2.*(convert(Array{Float32,2},z1n[1][i][:,1:1]).<zeros(Float32,size(z1n[1][i][:,1:1],1),1)).-1)
    end

    ylogx(y,x) = y.*log.(max.(x,0.0001))
    if(length(γ[1])==1)
		z1n[3] = reshape(cond_mean(γ,[vcat(z1n[1]...)]),size(z1n[1][1],1),length(z1n[1]))
    else
		z1n[3] = cond_mean(γ,z1n[1])
    end
    # NOTE: This is a known bug. By simulating outcomes in this way, we actually are simulating outcomes with probability
    # 	    1-z1n[2], rather than with probability z1n[2]. Due to the symmetry of the cross-entropy loss, the procedure is
    # 	    therefore learning to estimate one minus the conditional expectation, and so the final predictions can be corrected
    # 	    by simply subtracting them from 1.
    z1n[2] = convert(typeof(z1n[2]),convert(Array{Float32,2},z1n[3]).<rand(Float32,size(z1n[3],1),size(z1n[3],2)))

    if diffγ
		weights = sum(ylogx(z1n[2],z1n[3]) + ylogx(1.-z1n[2],1.-z1n[3]),1)
		weights = exp.(weights .- getval(weights))
		return z1n, weights
    else
    	return z1n
    end
end

function sim_logit_rho3(γ, z1n; diffγ=true)
    sim_logit_rho(γ, z1n, 0.3f0; diffγ=diffγ)
end

function sim_logit_rho6(γ, z1n; diffγ=true)
    sim_logit_rho(γ, z1n, 0.6f0; diffγ=diffγ)
end

function sim_logit_rho9(γ, z1n; diffγ=true)
    sim_logit_rho(γ, z1n, 0.9f0; diffγ=diffγ)
end

function sim_logit_rho(γ, z1n, rho; diffγ=true)
    # covariates
    CovMat = ones(Float32,size(z1n[1][1],2),size(z1n[1][1],2)).*rho
    CovMat[diagind(CovMat)] = 1.0
    rootCovMat = convert(typeof(z1n[1][1]),sqrtm(CovMat))
    for i=1:length(z1n[1])
        z1n[1][i] = convert(typeof(z1n[1][i]),randn!(z1n[1][i])*rootCovMat)
    end
    ylogx(y,x) = y.*log.(max.(x,0.0001))
    if(length(γ[1])==1)
    z1n[3] = reshape(cond_mean(γ,[vcat(z1n[1]...)]),size(z1n[1][1],1),length(z1n[1]))
    else
    z1n[3] = cond_mean(γ,z1n[1])
    end
    # NOTE: This is a known bug. By simulating outcomes in this way, we actually are simulating outcomes with probability
    # 	    1-z1n[2], rather than with probability z1n[2]. Due to the symmetry of the cross-entropy loss, the procedure is
    # 	    therefore learning to estimate one minus the conditional expectation, and so the final predictions can be corrected
    # 	    by simply subtracting them from 1.
    z1n[2] = convert(typeof(z1n[2]),convert(Array{Float32,2},z1n[3]).<rand(Float32,size(z1n[3],1),size(z1n[3],2)))
    if diffγ
        weights = sum(ylogx(z1n[2],z1n[3]) + ylogx(1.-z1n[2],1.-z1n[3]),1)
        weights = exp.(weights .- getval(weights))
        return z1n, weights
    else
        return z1n
    end
end

