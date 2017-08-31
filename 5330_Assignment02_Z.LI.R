library(ISLR)
library(MASS)
library(data.table)
library(rattle)
library(pROC)
library(ROCR)
setwd("/Users/lizhuo/Documents/UVA_Statistics/5330_Data Mining/DataSet")

training=read.csv("bank.csv", sep=";")
testing=read.csv("bank_full.csv", sep=";")
names(training)
summary(training)
head(training)

training.a=subset(training, select=c("age", "balance", "day", "duration", "pdays", "campaign", "previous", "y"))
testing.a=subset(testing, select=c("age", "balance", "day", "duration", "pdays", "campaign", "previous", "y"))

##############################################################
#Linear Discriminant Analysis
bank.lda<- lda(y~., data=training.a)
summary(bank.lda)
bank.lda

lda.pred=predict(bank.lda, testing.a)
names(lda.pred)
lda.class=lda.pred$class
table(lda.class, testing.a$y)

#accuracy 
ac.lda<- mean(testing.a$y == lda.pred$class)
ac.lda

#precision/positive predictive value

#negative predictive value

#5-fold cv
n<- nrow(testing.a)
m<- 5
len<- floor(n/m)
pred.err.lda<- matrix(0,1,m)
for(i in 1:m){
  ind<- (len*i - len +1):(len*i)
  trainingi<- testing.a[-ind, ]
  testingi<- testing.a[ind,]
  fit<- lda(y~., trainingi)
  yhat<- predict(fit, testingi)$class
  pred.err.lda[i]<- mean((as.numeric(testingi$y) - as.numeric(yhat))^2)
}
mean(pred.err.lda)



####################################################################
#Quadralic Discriminant Analysis
bank.qda<- qda(y~., data=training.a)
summary(bank.qda)
bank.qda

qda.pred=predict(bank.qda, testing.a)
names(qda.pred)
qda.class=qda.pred$class
table(qda.class, testing.a$y)

#accuracy 
ac.qda<- mean(testing.a$y == qda.pred$class)
ac.qda

#5-fold cv
n<- nrow(testing.a)
m<- 5
len<- floor(n/m)
pred.err.qda<- matrix(0,1,m)
for(i in 1:m){
  ind<- (len*i - len +1):(len*i)
  trainingi<- testing.a[-ind, ]
  testingi<- testing.a[ind,]
  fit<- qda(y~., trainingi)
  yhat<- predict(fit, testingi)$class
  pred.err.qda[i]<- mean((as.numeric(testingi$y) - as.numeric(yhat))^2)
}
mean(pred.err.qda)



#######################################################################
#Logistics Regression
bank.log<- glm(y~., data=training.a, family="binomial")
summary(bank.log)
bank.log

log.pred=predict(bank.log, testing.a, type="response")
table(testing.a$y, log.pred>0.5)


#accuracy 


#5-fold cv
n<- nrow(testing.a)
m<- 5
len<- floor(n/m)
pred.err.log<- matrix(0,1,m)
for(i in 1:m){
  ind<- (len*i - len +1):(len*i)
  trainingi<- testing.a[-ind, ]
  testingi<- testing.a[ind,]
  fit<- glm(y~., trainingi, family="binomial")
  yhat<- predict(fit, testingi, type="response")
  yhat[yhat>0.5]<- 1
  yhat[yhat<0.5]<- 0
  y<- as.numeric(testingi$y)
  y[y==1]<- 0
  y[y==2]<- 1
  pred.err.log[i]<- mean((yhat-y)^2)
}
mean(pred.err.log)




############################################################################
#ROC

pred.lda <- predict(bank.lda, testing.a)$post[,2]
pred.roc.lda <- prediction(pred.lda, testing.a$y)
perf.roc.lda <- performance(pred.roc.lda, "tpr", "fpr")
plot(perf.roc.lda, col="black", main="Fig. 1 ROC for LDA, QDA and Logistic Regression")

pred.qda <- predict(bank.qda, testing.a)$pos[,2]
pred.roc.qda <- prediction(pred.qda, testing.a$y)
perf.roc.qda <- performance(pred.roc.qda, "tpr", "fpr")
plot(perf.roc.qda, add=T, col="blue")

pred.log<- predict(bank.log, testing.a)
pred.roc.log<- prediction(pred.log, testing.a$y)
perf.roc.log <- performance(pred.roc.log, "tpr", "fpr")
plot(perf.roc.log, add=T, col="red" )
legend("bottomright", inset=.05, c("LDA", "QDA", "Log"), fill=c("black", "blue", "red"), horiz=T)

# Compute AUC for predicting Class with the model
auc.lda <- performance(pred.roc.lda, measure = "auc")
auc.lda <- auc.lda@y.values[[1]]
auc.lda

auc.qda <- performance(pred.roc.qda, measure = "auc")
auc.qda <- auc.qda@y.values[[1]]
auc.qda

auc.log <- performance(pred.roc.log, measure = "auc")
auc.log <- auc.log@y.values[[1]]
auc.log

############################################################################
library(lattice)
library(ggplot2)
training.b=subset(training, select=c("balance", "duration", "y"))
testing.b=subset(testing, select=c("balance", "duration", "y"))


bank.lda.b<- lda(y~., data=training.b)
bank.qda.b<- qda(y~., data=training.b, prior=c(1,1)/2)
bank.log.b<- glm(y~., data=training.b, family="binomial")

pred.lda.b<- predict(bank.lda.b, data=testing.b)
pred.qda.b<- predict(bank.qda.b, data=testing.b)
pred.log.b<- predict(bank.log.b, data=testing.b)

plot(testing.b$duration, testing.b$balance, col=c("blue", "green"), xlim=c(0,3000), ylim=c(-1000,70000), xlab="Duration", ylab="Balance", main="Classification Boundaries for LDA, QDA and Logistic Regression")
x= seq(0,3000, 100)
y=seq(-1000, 70000, 100)
z=as.matrix(expand.grid(x,y), 0)
m=length(x)
n=length(y)

x.yes=subset(testing.b, y=="yes", select=c("duration", "balance"))
x.no=subset(testing.b, y=="no", select=c("duration", "balance"))
n.yes=dim(x.yes)
n.no=dim(x.no)
n1=5289
n2=39922


#lda
#prior probabilities
y=as.numeric(testing.b[,3])

pi_est.lda<- c(mean(y==2), mean(y==1))
pi_est.lda


mu1_est.lda<- apply(x.yes, 2, mean)
mu2_est.lda<- apply(x.no, 2, mean)

#covariance matrix
sigma_est.lda<- (cov(x.yes)*(n1-1)+cov(x.no)*(n2-1))/(n1+n2-2)
sigma_est.lda

#intercept
int_est.lda<- log(pi_est.lda[1]/pi_est.lda[2])-.5*t(mu1_est.lda+mu2_est.lda)%*%solve(sigma_est.lda)%*%(mu1_est.lda-mu2_est.lda)
coef_est.lda<- solve(sigma_est.lda)%*%(mu1_est.lda-mu2_est.lda)

int_est.lda
coef_est.lda

abline(-int_est.lda/coef_est.lda[2], -coef_est.lda[1]/coef_est.lda[2])


#logistics
abline(a=-coef(bank.log.b)[1]/coef(bank.log.b)[2], b=-coef(bank.log.b)[3]/coef(bank.log.b)[2], col="red")

#QDA

#we can use contour plot to show the decision boundary
##create a grid for our plotting surface
x <- seq(0,3000,100)
y <- seq(-1000,70000,1000)
z <- as.matrix(expand.grid(x,y),0)
colnames(z)<-c("Duration","Balance")
z<-as.data.frame(z)
m <- length(x)
n <- length(y)
z.qdp<-as.numeric(predict(object=bank.qda.b,newdata=testing.b)$class)

#plot(salmon[,1:2],pch=rep(c(18,20),each=50),col=rep(c(2,4),each=50))
contour(x,y,matrix(z.qdp,m,n), add=TRUE, drawlabels =FALSE, lty=1)
#legend("topleft",legend=c("阿拉斯加","加拿大"),pch=c(18,20),col=c(2,4))


legend("topright", inset=.05, c("LDA", "QDA", "Log"), fill=c("black", "purple", "red"), horiz=T)
