
function printTᵏ(μ,μlen,σ,σlen)
  println("std(μ): $(std(μ))"); println("std(σ): $(std(σ))")
  println("std(μlen): $(std(μlen))"); println("std(σlen): $(std(σlen))")
  println("μ: $(μ[1:100])"); println("μlen: $(μlen[1:100])")
  println("σ: $(σ[1:100])"); println("σlen: $(σlen[1:100])")
end

function printTᵏ(μ,μlen)
  println("std(μ): $(std(μ))"); println("std(μlen): $(std(μlen))")
  println("μ: $(μ[1:100])"); println("μlen: $(μlen[1:100])")
end

function printTᵏCIs(epoch,μ1,μ2,σ1,σ2)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  println((:epoch, epoch, :T_LCIμ,   quantile(data(μ1), pquant)))
  println((:epoch, epoch, :T_UCIμ,   quantile(data(μ2), pquant)))
  println((:epoch, epoch, :T_LCIσ,   quantile(data(σ1), pquant)))
  println((:epoch, epoch, :T_UCIσ,   quantile(data(σ2), pquant)))
end

function printTᵏCIsize(epoch,μ1,μ2,σ1,σ2)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  CIsize = abs.(data(μ2) .- data(μ1)) .* abs.(data(σ2) .- data(σ1))
  println((:epoch, epoch, :T_CIsize,     quantile(CIsize, pquant)))
  println((:epoch, epoch, :T_meanCIsize, mean(CIsize)))
end

function printTᵏCIs(epoch,μ1,μ2)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  println((:epoch, epoch, :T_LCIμ,   quantile(data(μ1), pquant)))
  println((:epoch, epoch, :T_UCIμ,   quantile(data(μ2), pquant)))
end

function printTᵏCIsize(epoch,μ1,μ2)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  CIsize = abs.(data(μ2) .- data(μ1))
  println((:epoch, epoch, :T_CIsize, quantile(CIsize, pquant)))
  println((:epoch, epoch, :T_meanCIsize, mean(CIsize)))
end

function printTᵏcent(epoch,μcent,σcent)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  println((:epoch, epoch, :T_μcent,   quantile(data(μcent), pquant)))
  println((:epoch, epoch, :T_σcent,   quantile(data(σcent), pquant)))
end

function printTᵏoff(epoch,μLoff,μUoff,σLoff,σUoff)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  println((:epoch, epoch, :T_μLoff,   quantile(data(μLoff), pquant)))
  println((:epoch, epoch, :T_μUoff,   quantile(data(μUoff), pquant)))
  println((:epoch, epoch, :T_σLoff,   quantile(data(σLoff), pquant)))
  println((:epoch, epoch, :T_σUoff,   quantile(data(σUoff), pquant)))
end

function printPiᵏepoch(epoch,μ1,σ1,μ2,σ2)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  println((:epoch, epoch, :Pμ_len, quantile(data(μ1), pquant)))
  println((:epoch, epoch, :Pσ_len, quantile(data(σ1), pquant)))
  println((:epoch, epoch, :Pμ_cov, quantile(data(μ2), pquant)))
  println((:epoch, epoch, :Pσ_cov, quantile(data(σ2), pquant)))
end

function printPiᵏepoch(epoch,μ1,μ2)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  println((:epoch, epoch, :Pμ_len, quantile(data(μ1), pquant)))
  println((:epoch, epoch, :Pμ_cov, quantile(data(μ2), pquant)))
end

function printMLECIsize(epoch,x1n,μ1,μ2,σ1,σ2)
  pquant = [0.0, 0.25, 0.5, 0.75, 1.0]
  MLECIsize = predictTᵏlen(CX_MLE(),x1n,size(x1n,2);γ=0.079f0)
  TᵏCIsize = abs.(data(μ2) .- data(μ1)) .* abs.(data(σ2) .- data(σ1))
  Tᵏ_MLE_CIsize = TᵏCIsize ./ MLECIsize
  println((:epoch, epoch, :Tᵏ,      :q, quantile(TᵏCIsize, pquant),      :mean, mean(TᵏCIsize)))
  println((:epoch, epoch, :MLE,     :q, quantile(MLECIsize, pquant),     :mean, mean(MLECIsize)))
  println((:epoch, epoch, "Tᵏ/MLE", :q, quantile(Tᵏ_MLE_CIsize, pquant), :mean, mean(Tᵏ_MLE_CIsize)))
  return quantile(Tᵏ_MLE_CIsize, pquant), mean(Tᵏ_MLE_CIsize)
end

function printTᵏPiᵏstats(name,epoch,setη,mPiᵏ,noise,mTᵏ,z1n,modelP,paramCI,atype,udim,nbatch)
  dβ = Distributions.Beta(paramCI.η_α,paramCI.η_β)
  η = Array{Float32}(nbatch)
  η = rand!(dβ, η)
  if (setη == :η1)
    η .= paramCI.η1
  elseif (setη == :η2)
    η .= paramCI.η2
  end

  noise = sample_noise(atype,udim,nbatch)
  γᵏ = predictPiᵏ(mPiᵏ,noise,mTᵏ,z1n,modelP,paramCI)
  γᵏ = map(x -> data(x), γᵏ)

  x1n = sim_norm_gpu!(γᵏ,z1n)
  x1n = reshape(x1n, 1, size(x1n,1), size(x1n,2))
  TᵏPiᵏ = mTᵏ(x1n,η)
  ψ₀ = Ψ(modelP,γᵏ)

  println("-----------------------------------------------------------------")
  println((name, :η, setη, η[1]))
  println("-----------------------------------------------------------------")

  printPiᵏepoch(epoch,γᵏ...)
  printTᵏCIs(epoch,   TᵏPiᵏ[1]...)
  printTᵏcent(epoch,  TᵏPiᵏ[2]...)
  println("-----------------------------------------------------------------")
  qTᵏ_MLE_CIsize, meanTᵏ_MLE_CIsize = printMLECIsize(epoch,x1n,TᵏPiᵏ[1]...)
  println("-----------------------------------------------------------------")

  return qTᵏ_MLE_CIsize, meanTᵏ_MLE_CIsize
end
