## CI center risk C(X) for 2-dim parameter for Tᵏ.cent(X)
## losstypecent can be either :L2 or :MAD
function rCIcent(μcent,σcent,γ;kcent=1.0f0,losstypecent=:L2)

  if (losstypecent == :L2)
    ## vector of risks for CI center (L2-norm distance of each gamma from CI estimated center)
    CIcentRisk = (μcent .- γ[1]).^2 .+ (σcent .- γ[2]).^2
  elseif (losstypecent == :MAD)
    ## vector of risks for CI center (abs distance of each gamma from CI estimated center)
    CIcentRisk = abs.(μcent .- γ[1]) .+ abs.(σcent .- γ[2])
  else
    error("losstypecent can be only :L2 or :MAD")
  end

  ## scaling factors for the center risk and taking the means
  if (kcent==0.0f0)
    CIcentRisk_norm = mean(CIcentRisk)
  elseif (kcent==1.0f0)
    CIcentRisk_norm = mean(CIcentRisk ./ (Ψσ(γ)))
  elseif (kcent==2.0f0)
    CIcentRisk_norm = mean(CIcentRisk ./ (Ψσ(γ).^2.0f0))
  elseif (kcent==0.5f0)
    CIcentRisk_norm = mean(CIcentRisk ./ (sqrt.(Ψσ(γ))))
  else
    error("only kmax=0.5, 1.0 or 2.0 are supported on GPU")
  end

  loss = CIcentRisk_norm
  return loss
end
