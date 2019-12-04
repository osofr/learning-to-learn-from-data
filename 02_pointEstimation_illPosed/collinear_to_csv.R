# Write design matrices from data sets with collinear
# predictors (mtcars and Mandel) to csv files so that
# they can be read into Julia.
# Columns of design matrices are scaled to have empirical
# mean zero and variance one

setwd("./02_pointEstimation_illPosed/")

############################################################
# mtcars
data(mtcars)

out = subset(mtcars,select=c(disp,cyl))
out = scale(out)

write.csv(out,file="mtcars_disp_cyl.csv")


############################################################
# Mandel
library(carData)
data(Mandel)

out = subset(Mandel,select=c(x1,x2))
out = scale(out)

write.csv(out,file="mandel_x1_x2.csv")
