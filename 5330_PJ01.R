######################################################################################
# Project 1 Test on Marijuana Gateway Drug Hytpothesis

#Zhuo Li zl5sn
#Jianhui Sun js9gu

library(ISLR)
library(MASS)
library(data.table)
library(rattle)
library(pROC)
library(ROCR)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(party)
library(randomForest)

##################################
# Clean up the data

#select initial variable
mydata<- subset(da35509.0001, select=c("AGE2", "IRSEX", "NEWRACE2", "MJYRTOT", "COCYRTOT", "HEREVER", "INCOME", "EDUCCAT2", "CATAG3"))

#change data format for variable AGE
age.temp=rep(0,55160)
for (i in 1:55160){
  age.temp[i]=substr(mydata$AGE2[i],2,3)
}
age<- as.numeric(age.temp)

#change data format for variable AGE
catag.temp=rep(0,55160)
for (i in 1:55160){
  catag.temp[i]=substr(mydata$CATAG3[i],2,2)
}
catag<- as.numeric(catag.temp)

#change data format for variable IRSEX (gender)
gender.temp=rep(0,55160)
for (i in 1:55160){
  gender.temp[i]=substr(mydata$IRSEX[i], 2,2)
}
gender<- gender.temp

#change the data format of RACE
  # 1 nonhisp white
  # 2 nonhisp Black
  # 3 nonhisp Native Am
  # 4 nonhisp Native HI
  # 5 nonhisp Asian
  # 6 Nonhisp more than one race
  # 7 hispanic
race.temp<- rep(0,55160)
for (i in 1:55160){
  race.temp[i]=substr(mydata$NEWRACE2[i],2,2)
}
race<- race.temp


#change the data format of the variable HEREVER (ever used heroin)
  # 1 yes 2 no
heroin.temp=rep(0,55160)
for (i in 1:55160){
  heroin.temp[i]=substr(mydata$HEREVER[i], 2,2)
}
heroin<- heroin.temp


#construct new dataset
newdata<- cbind(catag, gender, race, mydata$EDUCCAT2, mydata$INCOME, mydata$MJYRTOT, mydata$COCYRTOT, heroin)
newdata[is.na(newdata)]<- 0
colnames(newdata)[4]<- "education"
colnames(newdata)[5]<- "income"
colnames(newdata)[6]<- "mjrtot"
colnames(newdata)[7]<- "coctot"
head(newdata)
newdata<- data.frame(newdata)


#save configurated data
write.csv(newdata, "/Users/lizhuo/Documents/UVA_Statistics/5330_Data Mining/Project/5330_PJ01.csv")


##################################
# Data Summary and Statistics
setwd("/Users/lizhuo/Documents/UVA_Statistics/5330_Data Mining/Project")
library(data.table)
pj=fread("5330_PJ01.csv")
head(pj)
pj=subset(pj, heroin!=0)
pj$mjrtot=as.numeric(pj$mjrtot)
pj$coctot=as.numeric(pj$coctot)
pj$gender=as.factor(pj$gender)
pj$race=as.factor(pj$race)
pj$education=as.factor(pj$education)
pj$income=as.factor(pj$income)
pj$catag=as.factor(pj$catag)
pj$heroin=factor(pj$heroin, levels=c("1", "2"))
pj=subset(pj, select=c("catag", "gender", "race", "education", "income", "mjrtot", "coctot", "heroin"))
summary(pj)

#spliting training and test data: 50/50
pj=subset(pj, heroin!=0)
n=dim(pj)
#set.seed=(81)
train_ind=sample(seq_len(nrow(pj)), n/2)
# set.seed(9)  # optional
training=pj[train_ind,]
testing=pj[-train_ind,]

###################################
# For question 1 and 2, Logit regression and QDA are used. 
# QDA are preferred over LDA due to the reason that the same variance across different classes are not valid. 

###################################
# Question 1 Gateway Drug Hypothesis
## Logit regression
training$heroin=relevel(training$heroin, ref="2")                    # change the default baseline from 1 (yes to Heroin) t0 2 (never take heroin)
testing$heroin=relevel(testing$heroin, ref="2")
logit=glm(heroin~mjrtot+coctot, data=training, family="binomial")
summary(logit)

# check model significance
chi=logit$null.deviance-logit$deviance
df=logit$df.null-logit$df.residual
chisq.p=1-pchisq(chi, df)
chisq.p

# odds ratio for logit regression
CIodds=exp(confint(logit))
oddsratio=data.frame(Lower=CIodds[,1], odds=exp(logit$coefficients), Upper=CIodds[,2])

# confusion matrix for test set
logit.pred=predict(logit, data=testing, type="response")
logit.res=factor(ifelse(logit.pred>0.5, "1", "2"))
logit.table=table(logit.res, testing$heroin)

# classification error
logit_error=1-mean(testing$heroin==logit.res)
logit_error

# sensitivity
sens_logit=logit.table[2,2]/(logit.table[2,2]+logit.table[2,1])   #这个table 最好print out看一下
# specificity
spec_logit=logit.table[1,1]/(logit.table[1,1]+logit.table[1,2])

## Cross Validation 10-fold
n<- nrow(testing)
m<- 10
len<- floor(n/m)
pred.err.log<- matrix(0,1,m)
for(i in 1:m){
  ind<- (len*i - len +1):(len*i)
  trainingi<- testing[-ind, ]
  testingi<- testing[ind,]
  fit<- glm(heroin~mjrtot+coctot, trainingi, family="binomial")
  yhat<- predict(fit, testingi, type="response")
  yhat[yhat>0.5]<- 1
  yhat[yhat<0.5]<- 0
  y<- as.numeric(testingi$heroin)
  y[y==1]<- 0
  y[y==2]<- 1
  pred.err.log[i]<- mean((yhat-y)^2)
}
mean(pred.err.log)


## QDA
qda.f=qda(heroin~mjrtot+coctot, data=training)
qda.pred=predict(qda.f, newdata=testing)
# confusion matrix
qda.table=table(predict(qda.f, testing)$class, testing$heroin)

# classification error
qda_error=1-mean(testing$heroin==predict(qda.f, testing)$class)

#sensitivity
sens_qda=qda.table[2,2]/(qda.table[2,2]+qda.table[2,1])
#secificity
spec_qda=qda.table[1,1]/(qda.table[1,1]+qda.table[1,2])

## Cross Validation
n<- nrow(testing)
m<- 10
len<- floor(n/m)
pred.err.qda<- matrix(0,1,m)
for(i in 1:m){
  ind<- (len*i - len +1):(len*i)
  trainingi<- testing[-ind, ]
  testingi<- testing[ind,]
  fit<- qda(heroin~mjrtot+coctot, trainingi)
  yhat<- predict(fit, testingi)$class
  pred.err.qda[i]<- mean((as.numeric(testingi$heroin) - as.numeric(yhat))^2)
}
mean(pred.err.qda)

# ROC curve
log.pred=predict(logit, data=testing)
qda.pred=predict(qda.f, data=testing)

pred.qda <- predict(qda.f, testing)$pos[,2]
pred.roc.qda <- prediction(pred.qda, testing$heroin)
perf.roc.qda <- performance(pred.roc.qda, "tpr", "fpr")
plot(perf.roc.qda, col="blue")

pred.log<- predict(logit, testing)
pred.roc.log<- prediction(pred.log, testing$heroin)
perf.roc.log <- performance(pred.roc.log, "tpr", "fpr")
plot(perf.roc.log, add=T, col="red" )
legend("bottomright", inset=.05, c("QDA", "Log"), fill=c("blue", "red"), horiz=T)


# Compute AUC for predicting Class with the model
auc.qda <- performance(pred.roc.qda, measure = "auc")
auc.qda <- auc.qda@y.values[[1]]
auc.qda

auc.log <- performance(pred.roc.log, measure = "auc")
auc.log <- auc.log@y.values[[1]]
auc.log


# Dicision Boundaries
# QDA decision boundary
y=training$heroin
train=subset(training,select=c("mjrtot", "coctot", "heroin"))
pi_est=c(mean(y=="1"), mean(y=="2"))
class1<- train[heroin == "2",]
class2<- train[heroin == "1",]
n1<- nrow(class1)
n2<- nrow(class2)
class1=subset(class1, select=c("mjrtot", "coctot"))
class2=subset(class2, select=c("mjrtot", "coctot"))

mu1_est=apply(class1, 2, mean)
mu2_est=apply(class2, 2, mean)

sigma1_est=cov(class1)
sigma2_est=cov(class2)
s1i<- solve(sigma1_est)
s2i<- solve(sigma2_est)
k= log(det(sigma1_est)/det(sigma2_est))/2+(t(mu1_est)%*%s1i%*%mu1_est-t(mu2_est)%*%s2i%*%mu2_est)/2

x01=seq(0, 400, 10)
n01=length(x01)
f<- matrix(0, n01, n01)
for (i in 1:n01){
  for (j in 1:n01){
    xc<- c(x01[i], x01[i])
    f[i, j]<- -t(xc)%*%(s1i-s2i)%*%xc/2 + (t(mu1_est)%*%s1i-t(mu2_est)%*%s2i)%*%xc-k
  }
}
plot(class1$mjrtot, class1$coctot, col="blue", xlim=c(0,365), ylim=c(0, 365), xlab="Number of days using Marijuana", ylab="Number of days using cocaine", main="Decision Boundary for Question 1")
points(class2$mjrtot, class2$coctot, col="red")
contour(x01, x01, f, col="black", levels=c(1), add=T)
#logistic decision boundary
abline(-logit$coef[1]/logit$coef[3], -logit$coef[2]/logit$coef[3], col="orange")
legend("topright", inset=.05, c("QDA", "Log"), fill=c("black","red"), horiz=T)


########################################
# Question 2

## Logistic
training$heroin=relevel(training$heroin, ref="2")                    # change the default baseline from 1 (yes to Heroin) t0 2 (never take heroin)
testing$heroin=relevel(testing$heroin, ref="2")
logit2=glm(heroin~catag+race+education+income+gender, data=training, family="binomial")
summary(logit2)

# check model significance
chi=logit2$null.deviance-logit2$deviance
df=logit2$df.null-logit2$df.residual
chisq.p=1-pchisq(chi, df)
chisq.p

# odds ratio for logit regression
CIodds=exp(confint(logit))
oddsratio=data.frame(Lower=CIodds[,1], odds=exp(logit2$coefficients), Upper=CIodds[,2])

# confusion matrix for test set
logit.pred2=predict(logit2, data=testing, type="response")
logit.res2=factor(ifelse(logit.pred2>0.5, "1", "2"))
logit.table2=table(logit.res2, testing$heroin)

# classification error
logit_error2=1-mean(testing$heroin==logit.res2)
logit_error2

# sensitivity  # the logit regression doesn't predict any "1", so the table function dimension doesn't match, calculate by hand
sens_logit2=logit.table2[2,2]/(logit.table2[2,2]+logit.table2[2,1])
# specificity
spec_logit2=logit.table2[1,1]/(logit.table2[1,1]+logit.table2[1,2])


## Cross Validation 10-fold
n<- nrow(testing)
m<- 10
len<- floor(n/m)
pred.err.log<- matrix(0,1,m)
for(i in 1:m){
  ind<- (len*i - len +1):(len*i)
  trainingi<- testing[-ind, ]
  testingi<- testing[ind,]
  fit<- glm(heroin~catag+race+education+income+gender, trainingi, family="binomial")
  yhat<- predict(fit, testingi, type="response")
  yhat[yhat>0.5]<- 1
  yhat[yhat<0.5]<- 0
  y<- as.numeric(testingi$heroin)
  y[y==1]<- 0
  y[y==2]<- 1
  pred.err.log[i]<- mean((yhat-y)^2)
}
mean(pred.err.log)

# ROC curve
log.pred=predict(logit2, data=testing)

pred.log<- predict(logit2, testing)
pred.roc.log<- prediction(pred.log, testing$heroin)
perf.roc.log <- performance(pred.roc.log, "tpr", "fpr")
plot(perf.roc.log, col="red" )
legend("bottomright", inset=.05, c("Log"), fill=c("red"), horiz=T)

# Compute AUC for predicting Class with the model

auc.log <- performance(pred.roc.log, measure = "auc")
auc.log <- auc.log@y.values[[1]]
auc.log

# pairwise
# series=c(0,0,0.....)   #define the variables for comparision
x=subset(training, select=c("catag", "gender", "race", "education", "income"))
mse=(logit2$residuals)^2/(nrow(training)-2)
std=mse%*%t(series)%*%solve(t(x)%*%x)%*%series



# QDA
#qda.f=qda(heroin~catag+race+education+income, data=training)
#qda.pred=predict(qda.f, newdata=testing)
# confusion matrix
#qda.table=table(predict(qda.f, testing)$class, testing$heroin)

# classification error
#qda_error=1-mean(testing$heroin==predict(qda.f, testing)$class)

#sensitivity
#sens_qda=qda.table[2,2]/(qda.table[2,2]+qda.table[2,1])
#secificity
#spec_qda=qda.table[1,1]/(qda.table[1,1]+qda.table[1,2])


# Tree
tree.q2<- ctree(heroin~catag+race+education+income+gender, data=training)
plot(tree.q2, main="Decision Tree for Drug Use Population Prediction", cex=0.2)
predictions<- predict(model, testing)
table(predictions, testing$heroin)



#################################################
# Question 3
# Considering the interactive effects between all variables in the above two questions
# However, since we are combining 8 explanatory variables, 6 of which are categorical, adding 
# interactive effects between the variables would make the logistic regression 
# So for question 3 Tree will be used in analysis and prediction
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(party)
library(randomForest)

model<- ctree(heroin~., data=training)
plot(model, main="Decision Tree for Drug Use Population Prediction", cex=0.2)
predictions<- predict(model, testing)
table(predictions, testing$heroin)

rf=randomForest(heroin~., data=training, importance=T)
print(rf)
predictions<- predict(rf, testing)
table(testing$heroin, predictions)
partialPlot(rf, training, education, "1")
hist(treesize(rf))









