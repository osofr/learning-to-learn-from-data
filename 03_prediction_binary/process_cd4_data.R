
# processes data in adata and returns a table of 1000 validation indices (in rows)

set.seed(1)

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

wd = "./03_prediction_binary"
ds_name = "cd4"
response_var = "any_response"

####################################################################################

dir.create(paste0(wd,"/processed_",ds_name,"_data"))

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

library(data.table)
x = fread(file.path(wd,"icsresp070.csv"))

x[,sexbin:=as.numeric(x$sex=="F")]
x = x[tcellsub=="cd3+/cd4+" & (rx_code=="T1" | rx_code=="T2" | rx_code=="T3"),]

processed_data = dcast(x[visitno==9,],ptid + rx_code + bmi + sexbin ~ gene,value.var="response")
processed_data[,any_response:=pmax(ENV,GAG,POL,na.rm=TRUE)]

processed_data = processed_data[,c("sexbin","bmi","any_response")]

write.csv(processed_data,file=file.path(wd,"processed_cd4_data","processed_data.csv"),quote=FALSE,row.names=FALSE)

processed_data = as.data.frame(processed_data)

####################################################################################

source(file.path(wd,"data_analysis_funs.R"))

####################################################################################

make_val_inds(wd,ds_name,response_var,num_vals=1e3)
