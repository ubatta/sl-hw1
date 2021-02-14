# Load the necessary libraries
library('party')
library('rpart')
library('rpart.plot')
library('caret')
library('ROCR')
library("neuralnet")
library("NeuralNetTools")
#library(crypto)
library(msa)
library(corrplot)
library(dplyr)
library(gbm)
library(h2o)
library(xgboost)
library(ggplot2)
library(psych)
data= read.csv('winequality-red.csv', sep = ';')
head(data)
anyNA(data)
names(data)=c('fixed_acidity','volatile_acidity','citric_acidity','residual_sugar',
              'chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH',
              'sulphates','alcohol','quality')
head(data)
df=data
# Reprocessing
# checking for missing values
anyNA(data)
#Correlation Heatmap of Variables
pdf("rplot_corr.pdf") 
corrplot(cor(data))
dev.off()
# dealing with the target variable i.e.quality
data$quality=ifelse(data$quality>=6,'high','low')
table(data$quality)
#checking th data types
str(data)
# changing th quality variable to the right data type
data$quality=as.factor(data$quality)
#************* Exploratory Data Analysis ******************#
# Summary Statistics
describe(data)
pdf("rplot_data.pdf") 
ggplot(data,aes(x=quality,fill=factor(quality)))+geom_bar(stat = "count",position = "dodge")+
  ggtitle("Distribution of Good/Bad White Wines")+
  theme_classic()
dev.off()
# Distribution  of the variables in terms of the quality (boxplot)
pdf("boxplot.pdf")
featurePlot(x = data[, 1:11], 
            y = data$quality, plot = "box", 
            scales = list(x = list(relation="free"), y = list(relation="free")), 
            adjust = 1.5, pch = ".", 
            layout = c(4, 3), auto.key = list(columns = 3))
dev.off()
# Distribution  of the variables in terms of the quality (density)
pdf("density.pdf")
featurePlot(x = data[, 1:11], 
            y = data$quality, plot = "density", 
            scales = list(x = list(relation="free"), y = list(relation="free")), 
            adjust = 1.5, pch = ".", 
            layout = c(4, 3), auto.key = list(columns = 3))
dev.off()
#1. Decision Tree
# partion data into training and testing
set.seed(123)
n= sample(2, nrow(data), replace = T, prob = c(0.8,0.2))
train= data[n==1,]
test= data[n==2,]
# creating the model without the pruning 
starttime <- proc.time()
model= ctree(quality~., data= train)
model
pdf("rplot_dtree_model.pdf")
plot(model)
dev.off()
pred= predict(model, test)
print(pred)
#data.frame( R2 = R2(pred, test$quality), 
#            RMSE = RMSE(pred, test$quality), 
#            MAE = MAE(pred, test$quality)) 
#print("Test results : ", RMSE, MAE, R2 )
# model validation by the test data
confusionMatrix(pred, reference = as.factor(test$quality))
# model validation of the train data
predTrain= predict(model, train)
confusionMatrix(predTrain, reference = as.factor(train$quality))
print(train$quality)

print(model)
print(train)
print(predTrain)

# pruning
model= ctree(quality~., data= train,controls = ctree_control(mincriterion = 0.39,minsplit = 550))
model
pdf("prn_dt_wn.pdf")
plot(model)
dev.off()
pred= predict(model, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$quality))
# model validation of the train data
predTrain= predict(model, train)
confusionMatrix(predTrain, reference = as.factor(train$quality))

# Function to produce ROC curve
pdf("plotdt_roc.pdf")
plotROC <- function(truth, pred, ...){
  pred <- prediction(abs(pred), truth)    
  perf <- performance(pred,"tpr","fpr") 
  plot(perf, ...)
}
dev.off()
# Calculating the Misclassification Error rate
misclassificationError <- mean(pred != test$quality)
misclassificationError

stoptime <- proc.time()
dtree_exe_time <- stoptime - starttime
print("dtree wineset: ")
print(dtree_exe_time)
#Converting the quality factor into numeric
#actual_num <- as.numeric(test$quality)
# Converting Predicted r part into numeric
#predict_rpart_num <- as.numeric(pred)
# Plotting the roc curve
#plotROC(actual_num, predict_rpart_num)
# 2. Neural Network
# scaling data
starttime <- proc.time()
max = apply(df, 2 , max)
min = apply(df, 2 , min)
scaled = as.data.frame(scale(df, center = min, scale = max - min))
index= sample(2, nrow(df), replace = T, prob = c(0.8,02))
trainNN = scaled[index , ]
testNN = scaled[-index , ]
nn = as.formula("quality~.")
m = neuralnet(nn,
              data=trainNN, 
              hidden=2, 
              
              linear.output = F)
train_params <- trainControl(method = "repeatedcv", 
                             number = 10, repeats=5)
m$model.list
pdf("nn_tt.pdf")
plot(m)
dev.off()
plotnet(m, 
        alpha.val = 0.8, 
        circle_col = list('purple', 'white', 'white'), 
        bord_col = 'black')      


nnet_model <- train(train[,-12], train$quality,
                    method = "nnet",
                    trControl= train_params,
                    preProcess=c("scale","center")
)
plot(varImp(nnet_model))
pdf("nnet.pdf")
plot(nnet_model)
dev.off()
prop.table(table(train$quality))   #Baseline Accuracy

# Predictions on the training set
nnet_predictions_train <-predict(nnet_model, train)

nnet_predictions_train
str(test)
dim(test)
# Confusion matrix on training data
confusionMatrix(nnet_predictions_train, reference = as.factor(train$quality))

#table(train$quality, nnet_predictions_train)
#(565+444)/nrow(train)                    

#Predictions on the test set
nnet_predictions_test <-predict(nnet_model, test)

# Confusion matrix on test set
confusionMatrix(nnet_predictions_test, reference = as.factor(test$quality))
#print("grid - 1")
#grid <- expand.grid(size=seq(3,10,15),decay = seq(0.1,0.2,0.3))
#print("grid-2")
#model <- train(train[,-12], train$quality, method="nnet", trControl=train_params, tuneGrid=grid)
# summarize the model
print(model)
# plot the effect of parameters on accuracy
pdf("nn1.pdf")
plot(model)
dev.off()
# Predictions on the training set
nnet_predictions_train <-predict(model, train)

nnet_predictions_train
str(test)
dim(test)
# Confusion matrix on training data
confusionMatrix(nnet_predictions_train, reference = as.factor(train$quality))

stoptime <- proc.time()
dtree_exe_time <- stoptime - starttime
print("Neural network Wineset: ")
print(dtree_exe_time)


# boosting

require(gbm)
starttime <- proc.time()

boosting_model <- gbm(quality~., data= train, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 4)
pdf("boosting_wn.pdf")
summary(boosting_model)
dev.off()

#plot(boosting_model,i="lstat")
#plot(boosting_model,i="rm")
set.seed(123)

# train GBM model
gbm.fit <- gbm(
  formula = quality ~ .,
  distribution = "gaussian",
  data = train,
  n.trees = 10000,
  interaction.depth = 1,
  shrinkage = 0.001,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
print(gbm.fit)
print("gbm_wn.pdf")
summary(gbm.fit)
dev.off()
# get MSE and compute RMSE
sqrt(min(gbm.fit$cv.error))

# plot loss function as a result of n trees added to the ensemble
pdf("gbm_perf.pdf")
gbm.perf(gbm.fit, method = "cv")
dev.off()
# Tuning 
set.seed(123)

# train GBM model
gbm.fit2 <- gbm(
  formula = quality ~ .,
  distribution = "gaussian",
  data = train,
  n.trees = 5000,
  interaction.depth = 3,
  shrinkage = 0.1,
  cv.folds = 5,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  

# find index for n trees with minimum CV error
min_MSE <- which.min(gbm.fit2$cv.error)

# get MSE and compute RMSE
sqrt(gbm.fit2$cv.error[min_MSE])
# plot loss function as a result of n trees added to the ensemble
pdf("gbm_1.pdf")
gbm.perf(gbm.fit2, method = "cv")
dev.off()
# HYper
# create hyperparameter grid
hyper_grid <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  bag.fraction = c(.65, .8, 1), 
  optimal_trees = 0,               # a place to dump results
  min_RMSE = 0                     # a place to dump results
)

# for reproducibility
set.seed(123)

# train GBM model
gbm.fit.final <- gbm(
  formula = quality ~ .,
  distribution = "gaussian",
  data = train,
  n.trees = 483,
  interaction.depth = 5,
  shrinkage = 0.1,
  n.minobsinnode = 5,
  bag.fraction = .65, 
  train.fraction = 1,
  n.cores = NULL, # will use all cores by default
  verbose = FALSE
)  
par(mar = c(5, 8, 1, 1))
summary(
  gbm.fit.final, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)
# predict values for test data
pred <- predict(gbm.fit.final, n.trees = gbm.fit.final$n.trees, test)
#
model <- train(
  quality ~., data = train, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)
#Predictions on the test set
pred <-predict(model, test)

# Confusion matrix on test set
confusionMatrix(pred, reference = as.factor(test$quality))


# results
#caret::RMSE(pred, test)

stoptime <- proc.time()
dtree_exe_time <- stoptime - starttime
print("Boosting Wineset: ")
print(dtree_exe_time)


# Support Vector Machine
library(e1071)
# linear model
set.seed(222)

starttime <- proc.time()
svm.model= svm(quality~., data = train,
               kernel = "linear", cost = 10, scale = FALSE)
pdf("svm_1.pdf")
svm.model
dev.off()
# Model accuracy
pred= predict(svm.model, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$quality))
# model validation of the train data
predTrain= predict(svm.model, train)
confusionMatrix(predTrain, reference = as.factor(train$quality))
# plot
pdf("svm2.pdf")
plot(svm.model, data=train,
     alcohol~sulphates,
     slice=list(volatile_acidity=3,density=4))
dev.off()
# radial
svm.model= svm(quality~., data = train,
               kernel = "radial", cost = 10, scale = FALSE)
svm.model
# Model accuracy
pred= predict(svm.model, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$quality))
# model validation of the train data
predTrain= predict(svm.model, train)
confusionMatrix(predTrain, reference = as.factor(train$quality))
pdf("svm_r_p.pdf")
plot(svm.model, data=train,
     alcohol~sulphates,
     slice=list(volatile_acidity=3,density=4))
dev.off()
# Polynomial kernel
svm.model= svm(quality~., data = train,
               kernel = "polynomial", cost = 10, scale = FALSE)
svm.model
# Model accuracy
pred= predict(svm.model, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$quality))
# model validation of the train data
predTrain= predict(svm.model, train)
confusionMatrix(predTrain, reference = as.factor(train$quality))
pdf("svm_po_wn.pdf")
plot(svm.model, data=train,
     alcohol~sulphates,
     slice=list(volatile_acidity=3,density=4))
dev.off()
# Sigmoid kernel
svm.model= svm(quality~., data = train,
               kernel = "sigmoid", cost = 10, scale = FALSE)
svm.model
# Model accuracy
pred= predict(svm.model, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$quality))
# model validation of the train data
predTrain= predict(svm.model, train)
confusionMatrix(predTrain, reference = as.factor(train$quality))
pdf
plot(svm.model, data=train,
     alcohol~sulphates,
     slice=list(volatile_acidity=3,density=4))
# Fine Tuning
set.seed(123)
finetune= tune(svm, quality~., data=train,
               ranges = list(epsilon= seq(0,1,0.1), cost=2^(2:7)))
plot(finetune)
bestModel= finetune$best.model
summary(bestModel)
# Model accuracy
pred= predict(bestModel, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$quality))
# model validation of the train data
predTrain= predict(bestModel, train)
confusionMatrix(predTrain, reference = as.factor(train$quality))

stoptime <- proc.time()
dtree_exe_time <- stoptime - starttime
print("SVM Wineset: ")
print(dtree_exe_time)

# K-Neighbors
# fine tuning and cross validation

starttime <- proc.time()
train_params <- trainControl(method = "repeatedcv", 
                             number = 10, repeats=5)
set.seed(123)
fit= train(quality~., data=train,
           method='knn',
           tuneLength=20,
           trControl= train_params,
           preProc=c('center','scale'))
fit

fit
pred= predict(fit, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$quality))
# model validation of the train data
predTrain= predict(fit, train)
confusionMatrix(predTrain, reference = as.factor(train$quality))
pdf("KNN_wn.pdf")
plot(fit)
dev.off()
varImp(fit)
#using different values of K
stoptime <- proc.time()
dtree_exe_time <- stoptime - starttime
print("KNN winset: ")
print(dtree_exe_time)

########################## Classification problem 2 *******
# Cancer dataset
cancer= read.csv('data.csv')
str(cancer)
# remove id and x
cancer$id=NULL
cancer$x=NULL
#  convert diagnosis to categorical
cancer$diagnosis=as.factor(cancer$diagnosis)
# check any missing values
anyNA(cancer)
pdf("ggplot_dt_can.pdf")
ggplot(cancer,aes(x=diagnosis,fill=factor(diagnosis)))+geom_bar(stat = "count",position = "dodge")+
  ggtitle("Distribution of Factor")+
  theme_classic()
dev.off()
# Distribution  of the variables in terms of the quality (boxplot)
#featurePlot(x = cancer[, 1:11], 
#            y = cancer$diagnosis, plot = "box", 
#            scales = list(x = list(relation="free"), y = list(relation="free")), 
#            adjust = 1.5, pch = ".", 
#            layout = c(4, 3), auto.key = list(columns = 3))
# Distribution  of the variables in terms of the quality (density)
#featurePlot(x = cancer[, 1:11], 
#            y = cancer$diagnosis, plot = "density", 
#            scales = list(x = list(relation="free"), y = list(relation="free")), 
#            adjust = 1.5, pch = ".", 
#            layout = c(4, 3), auto.key = list(columns = 3))

# 1. Decision tree
# partion data into training and testing
set.seed(123)
n= sample(2, nrow(cancer), replace = T, prob = c(0.8,0.2))
starttime <- proc.time()
train= cancer[n==1,]
test= cancer[n==2,]
# creating the model without the pruning 
anyNA(cancer)
head(train)
model= ctree(diagnosis~., data= train)
model
pdf("dt_i_can.pdf")
plot(model)
dev.off()
pred= predict(model, test)
# model validation by the test data
confusionMatrix(pred, reference = as.factor(test$diagnosis))
# model validation of the train data
predTrain= predict(model, train)
confusionMatrix(predTrain, reference = as.factor(train$diagnosis))

# pruning
model= ctree(diagnosis~., data= train,controls = ctree_control(mincriterion = 0.39,minsplit = 270))
model
pdf("dt_p_cab.pdf")
plot(model)
dev.off()
pred= predict(model, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$diagnosis))
# model validation of the train data
predTrain= predict(model, train)
confusionMatrix(predTrain, reference = as.factor(train$diagnosis))

# Function to produce ROC curve
pdf("plotroc_dt_can.pdf")
plotROC <- function(truth, pred, ...){
  pred <- prediction(abs(pred), truth)    
  perf <- performance(pred,"tpr","fpr") 
  plot(perf, ...)
}
dev.off()
# Calculating the Misclassification Error rate
misclassificationError <- mean(pred != test$diagnosis)
misclassificationError

stoptime <- proc.time()
dtree_exe_time <- stoptime - starttime
print("dtree cancer : ")
print(dtree_exe_time)
#testit(10)

#Neural network
nn = as.formula("diagnosis~.")
m = neuralnet(nn,
              data=train, 
              hidden=2, 
              
              linear.output = F)
pdf("knn_m_c.pdf")
plot(m)
dev.off()
plotnet(m, 
        alpha.val = 0.8, 
        circle_col = list('purple', 'white', 'white'), 
        bord_col = 'black')      

starttime <- proc.time()
nnet_model <- train(train[,-1], train$diagnosis,
                    method = "nnet",
                    trControl= train_params,
                    preProcess=c("scale","center")
)
plot(varImp(nnet_model))
pdf("nn_c.pdf")
plot(nnet_model)
dev.off()
prop.table(table(train$diagnosis))   #Baseline Accuracy

# Predictions on the training set
nnet_predictions_train <-predict(nnet_model, train)

# Confusion matrix on training data
confusionMatrix(nnet_predictions_train, reference = as.factor(train$diagnosis))

#Predictions on the test set
nnet_predictions_test <-predict(nnet_model, test)

# Confusion matrix on test set
confusionMatrix(nnet_predictions_test, reference = as.factor(test$diagnosis))

stoptime <- proc.time()
dtree_exe_time <- stoptime - starttime
print("Boosting Cancer: ")
print(dtree_exe_time)



# Boosting
starttime <- proc.time()
model <- train(
  diagnosis ~., data = train, method = "xgbTree",
  trControl = trainControl("cv", number = 10)
)
#Predictions on the test set
pred <-predict(model, test)

# Confusion matrix on test set
confusionMatrix(pred, reference = as.factor(test$diagnosis))
#Predictions on the train set
pred <-predict(model, train)

# Confusion matrix on test set
confusionMatrix(pred, reference = as.factor(train$diagnosis))

stoptime <- proc.time()
dtree_exe_time <- stoptime - starttime
print("Boosting Cancer: ")
print(dtree_exe_time)

# Support Vector Machine
library(e1071)
# linear model
set.seed(222)
starttime <- proc.time()
svm.model= svm(diagnosis~., data = train,
               kernel = "linear", cost = 10, scale = FALSE)
svm.model
# Model accuracy
pred= predict(svm.model, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$diagnosis))
# model validation of the train data
predTrain= predict(svm.model, train)
confusionMatrix(predTrain, reference = as.factor(train$diagnosis))

# radial
svm.model= svm(diagnosis~., data = train,
               kernel = "radial", cost = 10, scale = FALSE)
svm.model
# Model accuracy
pred= predict(svm.model, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$diagnosis))
# model validation of the train data
predTrain= predict(svm.model, train)
confusionMatrix(predTrain, reference = as.factor(train$diagnosis))

# Polynomial kernel
svm.model= svm(diagnosis~., data = train,
               kernel = "polynomial", cost = 10, scale = FALSE)
svm.model
# Model accuracy
pred= predict(svm.model, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$diagnosis))
# model validation of the train data
predTrain= predict(svm.model, train)
confusionMatrix(predTrain, reference = as.factor(train$diagnosis))

# Sigmoid kernel
svm.model= svm(diagnosis~., data = train,
               kernel = "sigmoid", cost = 10, scale = FALSE)
svm.model
# Model accuracy
pred= predict(svm.model, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$diagnosis))
# model validation of the train data
predTrain= predict(svm.model, train)
confusionMatrix(predTrain, reference = as.factor(train$diagnosis))


# Fine Tuning
set.seed(123)
finetune= tune(svm, diagnosis~., data=train,
               ranges = list(epsilon= seq(0,1,0.1), cost=2^(2:7)))
pdf("svm_tune_c.pdf")
plot(finetune)
dev.off()
bestModel= finetune$best.model
summary(bestModel)
# Model accuracy
pred= predict(bestModel, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$diagnosis))
# model validation of the train data
predTrain= predict(bestModel, train)
confusionMatrix(predTrain, reference = as.factor(train$diagnosis))

stoptime <- proc.time()
dtree_exe_time <- stoptime - starttime
print("SVM Cancer: ")
print(dtree_exe_time)

# K-Neighbors
# fine tuning and cross validation
starttime <- proc.time()
train_params <- trainControl(method = "repeatedcv", 
                             number = 10, repeats=5)
set.seed(123)
fit= train(diagnosis~., data=train,
           method='knn',
           tuneLength=20,
           trControl= train_params,
           preProc=c('center','scale'))

fit
pred= predict(fit, test)
# model validation
confusionMatrix(pred, reference = as.factor(test$diagnosis))
# model validation of the train data
predTrain= predict(fit, train)
confusionMatrix(predTrain, reference = as.factor(train$diagnosis))
pdf("knn_c.pdf")
plot(fit)
dev.off()
varImp(fit)

stoptime <- proc.time()
dtree_exe_time <- stoptime - starttime
print("KNN cancer: ")
print(dtree_exe_time)
#using different values of K

