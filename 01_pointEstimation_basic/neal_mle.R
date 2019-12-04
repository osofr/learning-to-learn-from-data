# MODIFICATION OF RADFORD NEAL'S CODE FROM
# https://radfordneal.wordpress.com/2008/08/09/inconsistent-maximum-likelihood-estimation-an-ordinary-example/
#
# Data is i.i.d. real, with distribution (1/2)N(0,1) + (1/2)N(t,exp(-1/t^2)^2),
# where t is a positive real parameter.
# Risk of the MLE is evaluated via Monte Carlo simulations

setwd("./01_pointEstimation_basic")

# GENERATE DATA FROM THE MODEL WITH A GIVEN PARAMETER VALUE.  Arguments are
# the parameter value, t, and the number of data points to generate, m.

gen <- function (t,n)
{
  m <- rep(0,n)
  s <- rep(1,n)
  
  w <- runif(n) < 1/2
  m[w] <- t
  s[w] <- exp(-1/t^2)

  rnorm(n,m,s)
}


# COMPUTE LOG DENSITIES OF DATA VALUES FOR A GIVEN PARAMETER VALUE.  Arguments
# are a vector of data values, x, and the parameter value, t.  
#
# Special care is taken to compute the log of extreme density values without 
# overflow.  Data points exactly equal to the parameter value are treated
# specially, since for them the log of the exponential part of the density 
# vanishes, producing results that don't overflow even when the standard
# deviation, exp(-1/t^2), overflows.

log.density <- function (x,t)
{
  ll1 <- dnorm(x,0,1,log=TRUE) + log(0.5)

  ll2 <- dnorm(x,t,exp(-1/t^2),log=TRUE) + log(0.5)
  ll2[x==t] <- -0.5*log(2*pi) + 1/t^2 + log(0.5)

  ll <- rep(NA,length(x))
  for (i in 1:length(x))
  { ll[i] <- add.logs(ll1[i],ll2[i])
  }

  ll
}


# COMPUTE THE LOG LIKELIHOOD GIVEN A DATA VECTOR AND PARAMETER VALUE.  Arguments
# are the vector of data values, x, and the parameter value, t.

log.lik <- function (x,t)
{
  sum(log.density(x,t))
}

# FIND the MLE. 
# This is a simplification of Radford Neal's plot.lik founction
# (since we don't need to produce a plot),
# with an added "bds" argument to enforce bounds on the parameter
# The argument is a vector of data values, x.
# An approximation to the MLE is found by taking the maximum likelihood values over a grid of
# parameter values from 0.01 to 3, in steps of 0.01, plus all positive
# data values less than 3.

get.mle <- function (x,bds=c(0,2))
{
  if(bds[2]<0.01) stop("Upper bound on the parameter too small.")
  grid <- sort(c(x[x>0 & x<3],seq(max(bds[1],0.01),bds[2],by=0.01)))

  ll <- rep(NA,length(grid))
  for (i in 1:length(grid))
  { ll[i] <- log.lik(x,grid[i])
  }

  mlv <- max(ll)
  mle <- grid[ll==mlv][1]
  return(mle)
}


# ADD NUMBERS REPRESENTED BY LOGS.  Computes log(exp(a)+exp(b)) in a way
# that avoids overflow/underflow problems.

add.logs <- function (a,b)
{ if (a>b)
  { a + log(1+exp(b-a))
  }
  else
  { b + log(1+exp(a-b))
  }
}

# Run num_mc repetitions of simulation

num_mc = 1000
param_grid = seq(0.2,2,by=0.2)
n_vals = 10*2^(0:4)

param = param_grid[1]

out = do.call(rbind,lapply(param_grid,function(param){
  mat = sapply(1:num_mc,function(i){
    x <- gen(param,100)
    sapply(seq(10,100,by=10),function(n){
      (get.mle(gen(param,n))-param)^2
      })
    })
  cbind(n_vals,rowMeans(mat),apply(mat,1,sd)/sqrt(ncol(mat)),param)
  }))

out = as.data.frame(out)

colnames(out) = c("n","MSE","se","param")
write.csv(out,file="neal_mle_out.csv",quote=FALSE,row.names=FALSE)
