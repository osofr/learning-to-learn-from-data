#     Taken from: https://gist.github.com/erantone/981d4f2d0620c5ef9f5e#file-expectation_maximization-jl
#     - Expectation-Maximization Implementation based on the book "Machine Learning" by Tom M. Mitchell
#     - Find the mean of One Gaussian; and of a mixture of Two Gaussians
#     Copyright (C) 2015  Eric Aislan Antonelo

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

# using Gadfly
# using Distributions
#
# ################## Uma Gaussiana
# mu = 5
# sigma = 10
# n = 1000
#
# srand(123) # Setting the seed
#
# d = Normal(mu,sigma)
#
# points = rand(d, n)
#
# plot(x=1:length(points), y=points)
# #### a Estimativa é a média dos pontos
# # np.mean(points)
# # plt.show()
#
# size(points)
#
# ################## Duas Gaussianas
# mus = [-1, 1]
# sigma = 1
# n = 10
# distr = [Normal(mu,sigma) for mu in mus]
# x = []
# n_each = div(n, length(distr))
# for d in distr
#     x = [x; rand(d, n_each)]
# end
# ClassC = zeros(n)
# ClassC[6:10] = 1
# # x = x[randperm(n)]
# #x = np.array([sample(sel_mu) for i in range(n)])
# plot(x=1:n, y=x)

##################
#### Usaremos EM (Expectation Maximiation) para estimar as médias (hipóteses) de cada Gaussiana a partir dos dados

function EM_Gaussians(x, n_h, sigma)
    ## initial guess for hypotheses (mean of Gaussians)
    n = length(x)
    # h = [sample(x) for i in 1:n_h]
    h = rand(x, n_h)
    h_ = deepcopy(h)
    # println(h)
    # println(h_)
    # println("Estimativa inicial, medias: $h")
    # initialize vector for non-observable variables z_i1, z_i2 (which indicate which Gaussian generated the i_th point)
    z = zeros(n, length(h))
    diff = 10
    _inc = 0.00001   # avoid division by zero
    iterations = 0
    ## anonymous functions:
    p = ((x,u) -> exp((-1/(2*sigma^2))*(x-u)^2))
    expected_z = ((i,j) -> p(x[i],h[j]) / (sum([ p(x[i],h[n]) for n in 1:n_h]) + _inc))
    while diff > 0.001
        ## E-step : estimation step
        #- z will hold the expected estimation for the non-observable variables
        for i in 1:n
            for j in 1:n_h
                z[i,j] = expected_z(i, j)
                #print i,j,x[i], h[j], z[i][j], sum([ p(x[i],h[n]) for n in range(h.size) ])
            end
        end
        ## M-step: find new hypotheses (means of Gaussians) which maximize the likelihood of observing the full data y = [x,z], given the current h
        ## Soma ponderada dos pontos, peso = probabilidade do ponto pertencer a tal Gaussiana (do passo anterior)
        for j in 1:n_h
            h_[j] = sum([ z[i,j] * x[i] for i in 1:n ]) / (sum([ z[i,j] for i in 1:n ]) + _inc)
        end
        # println(h_)
        # store in diff value to check for loop termination
        diff = mean(map(abs, [ h[j] - h_[j] for j in 1:n_h ] ))
        # diff = mean(abs([ h[j] - h_[j]) for j in 1:n_h ]))
        iterations += 1
        #print h, h_
        #print diff, iterations
        # update current hypothesis
        h = deepcopy(h_)
    end
    (h,z)
end

# ##################
# medias, z = EM_Gaussians(x,length(mus),sigma)
# println("Medias das Gaussianas, resultado: $medias")
# println("Medias Reais: $mus")
#
# predsEM = z[:,1]
# predsEM = z[:,2]
# auc(ClassC, predsEM)
#
# ##################
# ## Plota estimativa final para probabilidades de cada ponto pertencer a qual Gaussiana
# j = 1
# #plot(z[:,j],'*-')
# x_1 = zeros(n)
# x_2 = zeros(n)
# x_1[:] = NaN
# x_2[:] = NaN
# for i in 1:n
#     if z[i,1] > z[i,2]
#         x_1[i] = x[i]
#     else
#         x_2[i] = x[i]
#     end
# end
# # preto: medias reais, laranja: medias estimadas por EM
# plot(layer(x=1:(2n), y=[x_1 x_2], Geom.point),
#      layer(yintercept=medias,Geom.hline(color="orange", size=0.8mm)),
# layer(yintercept=mus,Geom.hline(color="black", size=1.5mm)))
