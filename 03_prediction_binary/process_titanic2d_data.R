
# process data in titanic3.csv
# source: http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv
# about: http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/Ctitanic3.html


####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

wd = "./03_prediction_binary"
ds_name = "titanic2d"
response_var = "survived"

####################################################################################

dir.create(paste0(wd,"/processed_",ds_name,"_data"))

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

library(data.table)
x = fread(file.path(wd,"titanic3.csv"))

processed_data = x[,c("age","fare","survived")]

processed_data = as.data.frame(processed_data)

# only keeping complete cases
processed_data = na.omit(processed_data)

write.csv(processed_data,file=file.path(wd,"processed_titanic2d_data","processed_data.csv"),quote=FALSE,row.names=FALSE)

####################################################################################

source(file.path(wd,"data_analysis_funs.R"))

####################################################################################

make_val_inds(wd,ds_name,response_var,num_vals=2e4)
