rm(list = ls(all = T))
getwd()

rm(list=ls())

#Set the directory
setwd("C:/Users/Asus/Documents/R programming")
getwd()

#Load libraries
#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')


#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

##Read the data
data_absent = read.csv("Absenteeism_at_work_Project.csv", header = T, na.strings = c(" ", "", "NA"))

#########Converting data types ##############
str(data_absent) # Checking the required data types

data_absent$Reason.for.absence = as.factor(data_absent$Reason.for.absence)
data_absent$Month.of.absence = as.factor(data_absent$Month.of.absence)
data_absent$Day.of.the.week = as.factor(data_absent$Day.of.the.week)
data_absent$Seasons = as.factor(data_absent$Seasons)
data_absent$Disciplinary.failure = as.factor(data_absent$Disciplinary.failure)
data_absent$Education = as.factor(data_absent$Education)
data_absent$Son = as.factor(data_absent$Son)
data_absent$Social.drinker = as.factor(data_absent$Social.drinker)
data_absent$Social.smoker = as.factor(data_absent$Social.smoker)
data_absent$Pet = as.factor(data_absent$Pet)

str(data_absent) # checking the required data types again

##################Missing values analysis##############################
#Create a dataframe with missing percentage
missing_val = data.frame(apply(data_absent,2,function(x){sum(is.na(x))}))

#Convert row names into columns
missing_val$Columns = row.names(missing_val)
row.names(missing_val) = NULL

#Rename the variable name
names(missing_val)[1] =  "Missing_percentage"

#Calculate the percentage
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(data_absent)) * 100

#Arrange in descending order
missing_val = missing_val[order(-missing_val$Missing_percentage),]

#Rearrange the column names
missing_val = missing_val[,c(2,1)]

#plot bar grapghfor missing values
ggplot(data = missing_val[1:3,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
  geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
  ggtitle("Missing data percentage (Absent)") + theme_bw()


#Actual value = 0
#predicted value = 31


##create missing value 
data_absent$Disciplinary.failure[169]

data_absent$Disciplinary.failure[169] = NaN

################KNN method
data_absent = knnImputation(data_absent, k = 7)

data_absent$Disciplinary.failure[169]


#######################################Outlier Analysis############################
# ## BoxPlots - Distribution and Outlier Check
numeric_index_absent = sapply(data_absent,is.numeric) #selecting only numeric

numeric_data_absent = data_absent[,numeric_index_absent]

cnames_absent = colnames(numeric_data_absent)

##cnames_direct = c("Transportation.expense","Distance.from.Residence.to.Work","Service.time","Age","Work.load.Average.day","Weight","Height","Body.mass.index")

############## Outlier analysis ################
for (i in 1:length(cnames_absent))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames_absent[i])), data = subset(data_absent))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames_absent[i])+
           ggtitle(paste("Box plot of Absenteeism for",cnames_absent[i])))
}


## Plotting plots together
gridExtra::grid.arrange(gn1,ncol=1)
gridExtra::grid.arrange(gn2,ncol=1)
gridExtra::grid.arrange(gn3,ncol=1)
gridExtra::grid.arrange(gn4,ncol=1)
gridExtra::grid.arrange(gn5,ncol=1)
gridExtra::grid.arrange(gn6,ncol=1)
gridExtra::grid.arrange(gn7,ncol=1)
gridExtra::grid.arrange(gn8,ncol=1)
gridExtra::grid.arrange(gn9,ncol=1)
gridExtra::grid.arrange(gn10,ncol=1)



###Creating boxplot of each variable ############
boxplot(data_absent$ID)
boxplot(data_absent$Transportation.expense)
boxplot(data_absent$Distance.from.Residence.to.Work)
boxplot(data_absent$Service.time)
boxplot(data_absent$Age)
boxplot(data_absent$Work.load.Average.day)
boxplot(data_absent$Hit.target)
boxplot(data_absent$Disciplinary.failure)
boxplot(data_absent$Education)
boxplot(data_absent$Son)
boxplot(data_absent$Social.drinker)
boxplot(data_absent$Social.smoker)
boxplot(data_absent$Pet)
boxplot(data_absent$Weight)
boxplot(data_absent$Height)
boxplot(data_absent$Body.mass.index)
boxplot(data_absent$Absenteeism.time.in.hours)



############## loop to remove outlier and impute using Knn############
for(i in cnames_absent)
  {val = data_absent[,i][data_absent[,i] %in% boxplot.stats(data_absent[,i])$out]
  #print(length(val))
  data_absent[,i][data_absent[,i] %in% val] = NA
  }

data_absent = knnImputation(data_absent, k = 7)

##################################Feature Selection################################################
## Correlation Plot 
corrgram(data_absent[,numeric_index_absent], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

####Drop Body mass index ######

## Chi-squared Test of Independence
factor_index = sapply(data_absent,is.factor)
factor_data = data_absent[,factor_index]

for (i in 1:11)
{
  print(names(factor_data)[i])
  print(chisq.test(table(data_absent$Absenteeism.time.in.hours ,factor_data[,i])))
}

## Dimension Reduction
data_absent = subset(data_absent, 
                         select = -c(ID,Weight,Day.of.the.week,Education,Social.smoker,Social.drinker,Pet,Work.load.Average.day))
  
str(data_absent)

##################################Feature Scaling################################################
#Normality check
qqnorm(data_absent$Transportation.expense)
hist(data_absent$Transportation.expense) #data is right skewed

numeric_index_norm = sapply(data_absent,is.numeric) #selecting only numeric

numeric_data_norm = data_absent[,numeric_index_norm]

cnames_absent_norm = colnames(numeric_data_norm)

cnames_norm = c("Transportation.expense","Distance.from.Residence.to.Work","Service.time","Age","Hit.target","Height","Body.mass.index","Absenteeism.time.in.hours")

for(i in cnames_norm){
  print(i)
  data_absent[,i] = (data_absent[,i] - min(data_absent[,i]))/
    (max(data_absent[,i] - min(data_absent[,i])))
}

df = data_absent
#Write output results back into disk
write.csv(data_absent, "data_absent_new.csv", row.names = F)
 

####################################Train and test data ##################
###############Clean the environment
library(DataCombine)
rmExcept("df")

#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index = createDataPartition(df$Absenteeism.time.in.hours, p = .80, list = FALSE)
train = df[ train.index,]
test  = df[-train.index,]

#Load Libraries
library(rpart)
library(MASS)

#Linear regression
#check Multicollinearity

library(usdm)
vif(df[, -13])
numeric_index = sapply(df,is.numeric) #selecting only numeric

numeric_data = df[,numeric_index]

cnames_df = colnames(numeric_data)
cnames_df = data.frame(cnames_df)

vifcor(cnames_df_numeric[, -8], th = 0.9)

#Build regression model on train data
lm_model = lm(Absenteeism.time.in.hours ~., data = train)

#Summary of the model
summary(lm_model)

#Predict the values of test data by applying the model on test data
predictions_LR = predict(lm_model , test[,1:9])

#Calculate MAPE ( mean absolute percentage error)
mape(test[,10], predictions_LR)

####### ##rpart for regression
fit = rpart(Absenteeism.time.in.hours ~ ., data = train, method = "anova")

#Predict for new test cases
predictions_DT = predict(fit, test[,-13])

#Alternate method
regr.eval(test[,'Absenteeism.time.in.hours'] , predictions_DT ,stats = c('mae','rmse','mape','mse'))

# Model Performance
fit
plot(fit)
varImp(fit)*100
pred <- predict(fit, newdata = test)
RMSE(predictions_DT, test$Absenteeism.time.in.hours)
plot(predictions_DT ~ test$Absenteeism.time.in.hours)

################################### Decision Tree Regression ##########################

#########rpart for regression #####################
fit_dt = rpart(Absenteeism.time.in.hours~ ., data = df, method = "anova")

#Predict for new test cases
#Predict for new test cases
predictions_DT = predict(fit_dt, test[,-13])

RMSE(predictions_DT, test$Absenteeism.time.in.hours)
