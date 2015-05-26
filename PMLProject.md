# =======================================================================================
# LOAD LIBRARIES
# =======================================================================================
# Following packages are required for analysis and presentation
# install.packages("caret")
library(caTools)
library(caret)
library(randomForest)
library(gbm)
library(e1071)
library(ggplot2)
library(grid)
library(gridExtra)
library(lattice)
library(rpart)
library(rpart.plot)
# library(rattle)
library(doParallel)

# =======================================================================================
# PROJECT PROBLEM
# =======================================================================================
# The goal of your project is to predict the manner in which the human subject wearing the accelerometers
# did the exercise. Accelerometers measure bodily movements.
#
# This is the "classe" variable in the training set. You may use any of the other variables 
# to predict with. You should create a report describing how you built your model, 
# how you used cross validation, what you think the expected out of sample error is, 
# and why you made the choices you did. 
#
# You will also use your prediction model to predict 20 different test cases. 

# =======================================================================================
# THE DATA
# =======================================================================================
# The data and more information about it - source http://groupware.les.inf.puc-rio.br/har
# Weight Lifting Exercises (WLE) Dataset 
# Six young health participants were asked to perform one set of 10 repetitions of the 
# Unilateral Dumbbell Biceps Curlin five different fashions:
# exactly according to the specification (Class A), 
# throwing the elbows to the front (Class B), 
# lifting the dumbbell only halfway (Class C), 
# lowering the dumbbell only halfway (Class D) 
# and throwing the hips to the front (Class E).

train = read.csv("pml-training.csv", header=T, stringsAsFactors=F, na.strings = c("NA", "#DIV/0!"))
test = read.csv("pml-testing.csv",header=T, stringsAsFactors=F, na.strings = c("NA", "#DIV/0!"))


# =======================================================================================
# FEATURE SELECTION 
# Lead to Data compression
# Retain relevant information
# Created based on expert application knowledge
# =======================================================================================

# combine test and train for refactoring & feature engineering
train$isTest = rep(0,nrow(train))
test$isTest = rep(1,nrow(test))
test$classe = rep('A',nrow(test))
train$problem_id = rep(9999,nrow(train))
all = rbind(train,test)

# set dependent variable (classe) and independent variable (user_name) to a factor 
all$classe = as.factor(all$classe)
all$user_name = as.factor(all$user_name)

# data cleansing to recognize and understand / eliminate quirks of data

# Week 1 -What data should you use - Slide 12/12
# Data properties - Know how the data connects to thing you are trying to pick 
# It was determined that statistical and summary measures all contained missing values
# and contained no explanatory value.
# new_window & num_window have no explanatory value
# Raw timestamps contained no explanatory value.
# cvtd_timestamp also contained no explanatory value since other variables measured acceleration 
# and thus also contained time element
# There all were dropped.
dropcols <- c(colnames(all)[apply(is.na(all), 2, any)], 
              "new_window",
              "num_window",
              "raw_timestamp_part_1",
              "raw_timestamp_part_2",
              "cvtd_timestamp"
)

all = all[,!colnames(all) %in% dropcols]

# split back into test and train
train = all[all$isTest==0,]
test = all[all$isTest==1,]
train$isTest=NULL
train$problem_id=NULL
test$isTest=NULL
test$classe = NULL

# workspace housecleaning
rm(all)
rm(all1)
rm(dropcols)

# Split data into Training and Probe Testing to get an idea of out of sample error rate
# I decided to use training (60%), probe test (20%) & validation (20%) from data set from train
# Week 1 Prediction study design Slide 7/8

# set seed
set.seed(123)

# create randomly sampled training and test sets
inTrain = createDataPartition(y = train$classe, p = 0.6, list = FALSE)

# 60% to training
train.training = train[inTrain,]
train.test = train[-inTrain,]

# 20% to test and 20% to validation fromt train.test
set.seed(123)
inTest = createDataPartition(y = train.test$classe, p = 0.5, list = FALSE)

# 20% to test
train.testing = train.test[inTest,]
# 20% to validate
train.validate = train.test[-inTest,]

# workspace housecleaning
rm(inTrain)
rm(inTest)
rm(train.test)

# Want to err on overcreation of features but first consider removing zero covariates - All appear to be
# good candidates
nsv = nearZeroVar(train.training[,4:54], saveMetrics = TRUE)

# Exploratory Analysis - Understand summary stats to guage likely best features amd select best pre-processing method
train.training.mean = sapply(train.training[,4:54],mean,na.rm=1)
train.training.min = sapply(train.training[,4:54],min,na.rm=1)
train.training.max = sapply(train.training[,4:54],max,na.rm=1)
train.training.sd = sapply(train.training[,4:54],sd,na.rm=1)
train.training.summarystats = data.frame(train.training.min, 
                                         train.training.max, 
                                         train.training.mean,
                                         train.training.sd)
# sort by sd
train.training.summarystats = train.training.summarystats[order(-train.training.sd, train.training.mean ) , ]

# Use rf to dertmine input variable importance
inVarImp = createDataPartition(y = train.training$classe, p = 0.2, list = F)
varImpSub = train.training[inVarImp, ]
varImpSub = subset(varImpSub[3:55])
varImpRF = train(classe ~ ., data = varImpSub, method = "rf")

varImpObj <- varImp(varImpRF)
plot(varImpObj, main = "Variable Importance of All Vars")
plot(varImpObj, main = "Variable Importance of Top 25 Vars", top = 25)

# Remove X from train & test data frames
train.training$X=NULL
train.testing$X=NULL
train.validate$X=NULL

test$X=NULL


# =======================================================================================
# CROSS VALIDATION 
# =======================================================================================


# Build model on training set to pick predictors & estimate test set accuracy from training set
# Larger k - less bias, more variance
# Smaller k - more bias, less variance

# if cross validate to pick predictors estimate, you MUST estimate errors on independent data Wk 1 Cross Validation 8/8

# Pre-process
# train.training.summarystats exploratory analysis shows a significant amount of vaiables 
# exhibit exceptionally high variability - Center & Scale pre-process was selected

#-----------------------------------------

# SET UP THE PARAMETERS
pargs = classe ~ .
preProc = c("center","scale")
ctrl <- trainControl(method="repeatedcv",          # use repeated 10fold cross validation
                     repeats=10,                          # do 3 repititions of 10-fold cv
                     number=10)

# =======================================================================================
# ALGORITHMS 
# =======================================================================================

# BOOSTED TREE MODEL										
set.seed(123)
registerDoParallel(4)		                        # Registrer a parallel backend for train
getDoParWorkers()

system.time(gbm.tune1010 <- train(pargs,
                              data = train.training,
                              method = "gbm",
                              trControl = ctrl,
                              verbose=T)
)


#---------------------------------
#  Learning Vector Quantization MODEL
#
set.seed(123)
registerDoParallel(4,cores=4)
getDoParWorkers()
system.time(
  lvq.tune1010 <- train(pargs,
                   data = train.training,
                   method = "lvq",
                   preProc = preProc,
                   trControl=ctrl)	        # same as for gbm above
)	

#---------------------------------
#  Support Vector Machine SVM MODEL
#
set.seed(123)
registerDoParallel(4,cores=4)
getDoParWorkers()
system.time(
  svm.tune1010 <- train(pargs,
                        data = train.training,
                        method = "svmRadial",
                        preProc = preProc,
                        trControl=ctrl)          # same as for gbm above
)	

#---------------------------------
# RANDOM FOREST MODEL
#
set.seed(123)
registerDoParallel(4,cores=4)
getDoParWorkers()
system.time(
  rf.tune1010 <- train(pargs,
                   data = train.training,
                   method = "rf",
                   nodesize = 5,  	
                   ntree=500,
                   preProc = preProc,
                   trControl=ctrl)	        # same as for gbm above
)	


#-----------------------------------
# COMPARE MODELS USING RESAPMLING
# Having set the seed to 123 before running gbm.tune and svm.tune we have generated paired samplesfor comparing models using resampling.
#
# The resamples function in caret collates the resampling results from the two models
rValues1010 <- resamples(list(gbm=gbm.tune1010, rf=rf.tune1010,  svm=svm.tune1010))
rValues1010$values
#---------------------------------------------
# BOXPLOTS COMPARING RESULTS
bwplot(rValues1010,metric="Accuracy")		# boxplot


# =======================================================================================
# EVALUATION
# =======================================================================================

# PREDICITONS
gbm.tune.pred = predict(gbm.tune1010, newdata=train.testing)
rf.tune.pred = predict(rf.tune1010, newdata=train.testing, type="raw")

# EXPECTED OUT OF SAMPLE ERROR
# rf
rf.tune.pred.oos = predict(rf.tune1010, newdata=train.validate, type="raw")
missClass = function(values, rf.tune.pred.oos) {
  sum(rf.tune.pred.oos != values)/length(values)
}  
rf.oos.error = missClass(train.validate$classe, rf.tune.pred.oos)

# gbm
gbm.tune.pred.oos = predict(gbm.tune1010, newdata=train.validate, type="raw")
missClass = function(values, gbm.tune.pred.oos) {
  sum(gbm.tune.pred.oos != values)/length(values)
}  
gbm.oos.error = missClass(train.validate$classe, gbm.tune.pred.oos)

