# Set the working Directory
setwd("F:/0_1_Trainings/Institutes/LearnBay/MachineLearning/Session3_02_June_2018_LogisticRegression/Codes")


# Import the Data Set
diabetesdata=read.csv("diabetes.csv", header=TRUE)

# Split the Data set into Training and Test in 70:30 proportion
# First run the model on Traininf Data and then validate it with Test data.
library(caret)
library(ROCR)

set.seed(123)
index <- createDataPartition(diabetesdata$Outcome, p=0.70, list=FALSE)
trainingdata <- diabetesdata[ index,]
testdata <- diabetesdata[-index,]

# List the Dimensions
dim(trainingdata)
dim(testdata)

### Predict and check on the Training Data
attach(trainingdata)
#Model Fitting
#logittrainingdata=glm(Outcome~Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction+Age, data=trainingdata, family=binomial)
logittrainingdata=glm(Outcome~., data=trainingdata, family=binomial)
summary(logittrainingdata)
logittrainingdata=glm(Outcome~.-(SkinThickness+Age), data=trainingdata, family=binomial)
summary(logittrainingdata)
logittrainingdata=glm(Outcome~., data=trainingdata, family=binomial)

#****************************************Step 1 Log Likelihood Ratio Test ****************************************#
library(lmtest)
lrtest(logittrainingdata)

#****************************************Step 2 Pseudo R Square *************************************************#
library(pscl)
pR2(logittrainingdata)

#****************************************Step 3 Individual Coefficients*****************************************#
summary(logittrainingdata)
#****************************************Step 4 ODDS RATIO - EXP **********************************************#
#****************************************Step 5 Predictor Variable Importance**************************************#
#confint(logittrainingdata)
#exp(coef(logittrainingdata))
#exp(confint(logittrainingdata))
#Newdata1=data.frame(Age>30)
#Probability1=predict(logittrainingdata, Newdata1, type="response")
#Probability1
#Odds1=Probability1/(1-Probability1)
#Odds1

#Newdata2=data.frame(Age<=30)
#Probability1=predict(logittrainingdata, Newdata2, type="response")
#Probability1
#Odds1=Probability1/(1-Probability1)
#Odds1

#****************************************Step 6 Confusion Matrix/Classification Table******************************#
predict_train=fitted(logittrainingdata)
predict_gg=floor(predict_train+0.5)
train_CM = table(Actual=trainingdata$Outcome, Prediction=predict_gg)
train_CM

# (True Positive + True Negative) / (Total)
True_Positive     =   train_CM[2,2]
True_Positive
True_Negative     =   train_CM[1,1]
True_Negative
False_Postitive   =   train_CM[1,2]
False_Postitive
False_Negative    =   train_CM[2,1]
False_Negative
Train_Total       =   (True_Positive + True_Negative + False_Postitive + False_Negative)
Train_Total
Train_Actual_No   =   (True_Negative + False_Postitive)
Train_Actual_No
Train_Actual_Yes  =   (True_Positive + False_Negative)
Train_Actual_Yes
Train_Pred_Yes    =   (True_Positive + False_Postitive)
Train_Pred_Yes
Train_Accuracy    =   (True_Positive + True_Negative)/Train_Total 
Train_Accuracy
Train_MisClas_Err =   (1 - Train_Accuracy)
Train_MisClas_Err
Train_Sensitivity =   True_Positive/Train_Actual_Yes
Train_Sensitivity
Train_FP_Rate     =   False_Postitive/Train_Actual_No
Train_FP_Rate
Train_Specificity =   True_Negative/Train_Actual_No
Train_Specificity
Train_Precision   =   True_Positive/Train_Pred_Yes
Train_Precision

#****************************************Step 7 ROC Plot **********************************************************#
#****************************************Step 8 Adjusting the Cutoff/threshold values******************************#

ROCRpred = prediction(predict_train, trainingdata$Outcome)
ROCRperf = performance(ROCRpred, "tpr", "fpr")
# Adding threshold labels
plot(ROCRperf, colorize=TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))
abline(a=0, b=1)
auc_train <- round(as.numeric(performance(ROCRpred, "auc")@y.values),2)
legend(.8, .2, auc_train, title = "AUC", cex=1)

#library(Deducer) 
#rocplot(logittrainingdata, AUC = TRUE, prob.label.digits=3)
####################################################################################################################


### Validate the above Model with the testdata Data
attach(testdata)

#**************************************** Predict for the Test Data ******************************#
predict_test <- predict(logittrainingdata, type = "response", newdata = testdata)
predict_test
testdata
predict_gg=floor(predict_test+0.5)
Test_CM = table(Actual=testdata$Outcome, Prediction=predict_gg)
Test_CM

# (True Positive + True Negative) / (Total)
True_Positive     =   Test_CM[2,2]
True_Positive
True_Negative     =   Test_CM[1,1]
True_Negative
False_Postitive   =   Test_CM[1,2]
False_Postitive
False_Negative    =   Test_CM[2,1]
False_Negative
Test_Total       =   (True_Positive + True_Negative + False_Postitive + False_Negative)
Test_Total
Test_Actual_No   =   (True_Negative + False_Postitive)
Test_Actual_No
Test_Actual_Yes  =   (True_Positive + False_Negative)
Test_Actual_Yes
Test_Pred_Yes    =   (True_Positive + False_Postitive)
Test_Pred_Yes
Test_Accuracy    =   (True_Positive + True_Negative)/Test_Total 
Test_Accuracy
Test_MisClas_Err =   (1 - Test_Accuracy)
Test_MisClas_Err
Test_Sensitivity =   True_Positive/Test_Actual_Yes
Test_Sensitivity
Test_FP_Rate     =   False_Postitive/Test_Actual_No
Test_FP_Rate
Test_Specificity =   True_Negative/Test_Actual_No
Test_Specificity
Test_Precision   =   True_Positive/Test_Pred_Yes
Test_Precision
#****************************************Step 7 ROC Plot **********************************************************#
#****************************************Step 8 Adjusting the Cutoff/threshold values******************************#
ROCRpred = prediction(predict_test, testdata$Outcome)
ROCRperf = performance(ROCRpred, "tpr", "fpr")
# Adding threshold labels
plot(ROCRperf, colorize=TRUE, print.cutoffs.at = seq(0,1,0.1), text.adj = c(-0.2, 1.7))
abline(a=0, b=1)
auc_test <- round(as.numeric(performance(ROCRpred, "auc")@y.values),2)
legend(.8, .2, auc_test, title = "AUC", cex=1)

#library(Deducer) 
#rocplot(logittrainingdata, AUC = TRUE, prob.label.digits=3)


