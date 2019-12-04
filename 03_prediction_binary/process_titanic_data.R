
# process data in titanic3.csv
# source: http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv
# about: http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/Ctitanic3.html


####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

wd = "./03_prediction_binary"
ds_name = "titanic"
response_var = "survived"

####################################################################################

dir.create(paste0(wd,"/processed_",ds_name,"_data"))

####################################################################################
# THIS SECTION NEEDS TO CHANGE FOR EACH DATA SET

library(data.table)
x = fread(file.path(wd,"titanic3.csv"))

x[,class1:=as.numeric(pclass==1)]
x[,class2:=as.numeric(pclass==2)]
# survived
# age
x[,sexbin:=as.numeric(sex=="female")]
# sibsp
# parch
# fare
x[,Cherbourg:=as.numeric(embarked=="C")]
x[,Southampton:=as.numeric(embarked=="S")]

# only age/fare are missing, note this in variable anyMissing
x[,anyMissing:=as.numeric(is.na(age))]
# if age is missing, impute with median
x$age[is.na(x$age)] = median(x$age,na.rm=TRUE)
# if fare is missing, impute with median in class
for(cls in 1:3){
	x$fare[is.na(x$fare) & x$pclass==cls] = median(x$fare[x$pclass==cls],na.rm=TRUE)
}

processed_data = x[,c("class1","class2","age","sexbin","sibsp","parch","fare","Cherbourg","Southampton","anyMissing","survived")]

processed_data = as.data.frame(processed_data)

write.csv(processed_data,file=file.path(wd,"processed_titanic_data","processed_data.csv"),quote=FALSE,row.names=FALSE)

####################################################################################

source(file.path(wd,"data_analysis_funs.R"))

####################################################################################

make_val_inds(wd,ds_name,response_var,num_vals=2e4)
