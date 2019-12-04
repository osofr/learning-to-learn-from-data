## simulating new data (an alternative strategy: create a closure for sim() and Ψ())
const simdct = Dict(:norm=>:sim_norm!, :lognorm=>:sim_lognorm!, :neal=>:sim_neal!, :mtcars_design=>:sim_mtcars_design!, :mandel_design=>:sim_mandel_design!)

function sim!(modelP::ParamModelPᵏ, γ, x1n)
    funcall = getfield(maximinNN1, simdct[modelP.name])
    funcall(γ, x1n)
end

## simulate a vector/matrix of normals, each distributed as μ=γ[1][i], σ=γ[2][i] from given vector of parameters
function sim_norm!(γ, z1n)
    randn!(z1n)
    z1n = (z1n' .* γ[2] .+ γ[1])'
end
## simulate a vector/matrix of lognormals
function sim_lognorm!(γ, x1n)
    exp.(sim_norm!(γ, x1n))
end
## simulate from the distribution described in 
# https://radfordneal.wordpress.com/2008/08/09/inconsistent-maximum-likelihood-estimation-an-ordinary-example/
function sim_neal!(γ, x1n)
	bb = convert(typeof(x1n),(rand(Float32,size(x1n,1),size(x1n,2)).<convert(Float32,0.5)))
    (bb'.*randn!(x1n)' .+ (1.-bb').*(γ[1].+(randn!(x1n)'.*exp.(-γ[1].*γ[1]))))'
end


# simulate from a regression for the mtcars/mandel data sets with unknown beta and fixed design matrix
function sim_X_design!(γ, z1n, X)
    randn!(z1n)
    if(length(γ[1])>1)
    	z1n = (convert(typeof(z1n),X) * hcat(γ...)') .+ z1n
    else
    	tmp = convert(typeof(z1n),X) * reshape(γ,length(γ),1)
    	z1n = hcat([tmp for i in 1:size(z1n,2)]...) .+ z1n
    end
    return z1n
end

# mtcars
mtcars = convert(Array{Float32},CSV.read("mtcars_disp_cyl.csv")[:,2:3])
mtcars = hcat(zeros(Float32,size(mtcars,1)),mtcars)

function sim_mtcars_design!(γ, z1n)
	sim_X_design!(γ, z1n, mtcars)
end

# mandel
mandel = convert(Array{Float32},CSV.read("mandel_x1_x2.csv")[:,2:3])
mandel = hcat(zeros(Float32,size(mandel,1)),mandel)

function sim_mandel_design!(γ, z1n)
	sim_X_design!(γ, z1n, mandel)
end