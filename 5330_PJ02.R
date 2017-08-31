###############################################
## STAT 5330 Data Mining Project II

# Zhuo LI
# Jianhui SUN

#### Preparing Data
setwd("/Users/lizhuo/Documents/UVA_Statistics/5330_Data Mining/Project")
library(data.table)
library(MASS)
set.seed(81)
### 

pj02<-fread("5330_PJ02.csv")
summary(pj02)

temp <- fread("2012_public_use_data_aug2016.csv")
pj02 <- subset(temp, select=c("PBA", "SQFT", "WLCNS", "RFCNS", "GLSSPC", "NFLOOR", "YRCON", "WKHRS", "NWKER", "MAINHT", "MAINCL", "MFHTBTU", "MFCLBTU", "HDD65", "CDD65", "MFEXP", "NGEXP", "FKEXP"))
# Variable explanation
# PBA       Principal building activity B5
# SQFT      Square footage B7
# WLCNS     Wall construction material B8
# RFCNS     Roof construction material B9
# GLSSPC    Percent exterior glass char B13
# NFLOOR    Number of floors B17
# YRCON     Year of construction B25 995-before year 1946
# WKHRS     Total hours open per week 0-168 B93
# NWKER     Number of employee B95
# MAINHT    Main heating equipment B152
# MAINCL    Main cooling equipment B226
# MFHTBTU   Major fuel heating use (thous Btu)
# MFCLBTU   Major fuel coolling use(thous Btu)
# HDD65     Heating degree days (base 65)
# CDD65     Cooling degree days (base 65)
# MFEXP     Annual major fuel expenditures $
# ELEXP     Annual electricity expenditures
# NGEXP     Annual natural gas expenditures
# FKEXP     Annual fuel oil expenditures

# for simplicity, we remove all entries with missing values, reducing a data sample to 908
pj02 <- na.omit(pj02)
write.csv(pj02, "5330_PJ02.csv")


# Research question 1 Using Building Charateristics for Analysing Commercial Building Energy Consumption and Prediction
# Research question 2 Choice of Main Heating and Cooling Equipment
# Research question 3 Characteristics of High-Energy-Efficient Building and Low-Energy-Efficient Building


# Spliting data
# Since we will be using different variables for different questions, we will split training/testing dataset after subsetting dataset for each question
# 80/20 training/testing

########################################
# For QUESTION 1 Using Building Charateristics for Analysing Commercial Building Energy Consumption and Prediction
# method: Stepwise, Lasso -- evaulation and prediction of MFEXP (annual major fuel expenditure)
# subsetting and spliting training/testing
q1 <- subset(pj02, select=c("PBA", "SQFT", "WLCNS", "RFCNS", "GLSSPC", "NFLOOR", "WKHRS", "NWKER", "MAINHT", "MAINCL", "HDD65", "CDD65", "MFEXP"))
summary(q1)
index <- sample(1:nrow(q1),round(0.8*nrow(q1)))
train1 <- q1[index,]
test1 <- q1[-index,]
train1 <- as.data.frame(train1)
########################################
#Stepwise
# Full model
regfit.full=lm(MFEXP~., data=train1)
# Null model
regfit.null=lm(MFEXP~1, data=train1)

## Forward
regfit.fwd=step(regfit.null, scope=list(lower=regfit.null, upper=regfit.full), direction="forward")
# model MFEXP ~ SQFT + NWKER + PBA + WKHRS + NFLOOR + MAINHT + CDD65 + MAINCL

## Backward
regfit.bwd=step(regfit.full, direction="backward")
# model MFEXP ~ PBA + SQFT + NFLOOR + WKHRS + NWKER + MAINHT + MAINCL + CDD65

## Hybrid
regfit.both=step(regfit.null, scope=list(upper=regfit.full), direction="both")
# model MFEXP ~ SQFT + NWKER + PBA + WKHRS + NFLOOR + MAINHT + CDD65 + MAINCL

# COMPARISON:
# All three methods gives the same model

# prediction error for the best model on testing set MFEXP ~ SQFT + NWKER + PBA + WKHRS + NFLOOR + MAINHT + CDD65 + MAINCL
pred.step=predict(regfit.both, test1)
pred.err.step=mean(((pred.step-test1$MFEXP)/test1$MFEXP)^2)

# Cross-Validation for best model on the whole dataset
#10-fold cv for forward model
n<- dim(q1)[1]
m<- 10
len<- floor(n/m)
pred.err.fwd<- matrix(0,1,m)
for(k in 1:m){
  ind<- (len*k - len +1):(len*k)
  trainingi<- q1[-ind, ]
  testingi<- q1[ind,]
  fit<- lm(MFEXP ~ SQFT + NWKER + PBA + WKHRS + NFLOOR + MAINHT + CDD65 + MAINCL, trainingi)
  yhat<- predict(fit, testingi)
  pred.err.fwd[k]<- mean((testingi$MFEXP - yhat)^2)
}
pred.err.fwd


## Final model using full dataset
regfit.step=lm(MFEXP ~ SQFT + NWKER + PBA + WKHRS + NFLOOR + MAINHT + CDD65 + MAINCL, q1)
regfit.step$coef


### Further note
model.ls <- lm(MFEXP ~ ., data=train1)
rss.ls <- sum(model.ls$resid^2)/model.ls$df.residual

model.backward <- step(model.ls, direction="backward")
rss.backward <- sum(model.backward$resid^2)/model.backward$df.residual

scope <- list(upper=~PBA+SQFT+WLCNS+RFCNS+GLSSPC+NFLOOR+WKHRS+NWKER+MAINHT+MAINCL+HDD65+CDD65, lower=~.)
model.forward <- step(lm(MFEXP ~ 1, data=train1), scope, direction="forward")
rss.forward <- sum(model.forward$resid^2)/model.forward$df.residual

model.both=step(lm(MFEXP ~ 1, data=train1), scope, direction="both")
rss.both <- sum(model.both$resid^2)/model.both$df.residual


r2 <- list()
AICs <- list()
for(i in 1:12){
  indexes <- combn(12,i)
  currentr2 <- NULL
  currentAIC <- NULL
  for(j in 1:dim(indexes)[2]){
    temp.model <- lm(MFEXP ~ ., data=train1[,c(indexes[,j], 13)])
    currentr2[j] <- summary(temp.model)$r.squared
    currentAIC[j] <- AIC(temp.model)
  }
  r2[[i]] <- currentr2
  AICs[[i]] <- currentAIC
}
compare <- function(set){
  s <- length(set)
  temp <- combn(12,s)
  check <- NULL
  for(i in 1:dim(temp)[2]){
    check[i] <- all(temp[,i]==set)
  }
  return(which(check))
}
backward <- c(compare(1:12), compare(c(1:2,4:12)), compare(c(1:2,4,6:12)), compare(c(1:2,4,6:10,12)), compare(c(1:2,6:10,12)))

#####################################################
# Shinkage & Dimension Reduction Model
library(glmnet)
### Ridge Regression
x<- model.matrix(MFEXP~., train1)
y<- train1$MFEXP
#alpha=0 for ridge, =1 (default) for lasso
fit.ridge<- glmnet(x, y, alpha=0)   #rss + lambda*sum sq of coefficients, smallest lambda==ols coefficients, keep all the bariable, shrink the coefficient to 0
#cv of ridge to determine lambda
cv.ridge<- cv.glmnet(x, y, alpha=0, type.measure="mse")
par(mfrow=c(1,2))
plot(fit.ridge, xvar="lambda")
plot(cv.ridge)  # two lines: one minimum, one std away from min
# prediction error
# validation approach
test.mat<- model.matrix(MFEXP~., data=test1)
val.errors<- rep(NA, length(fit.ridge$lambda))
for (i in 1:length(fit.ridge$lambda)){
  coefi<- fit.ridge$beta[,i]
  pred<- test.mat[,names(coefi)]%*%coefi
  val.errors[i]<- mean((test1$MFEXP-pred)^2)
}
val.errors
par(mfrow=c(1,1))
plot(fit.ridge$lambda, val.errors)
plot(log(fit.ridge$lambda), val.errors)

xnew=model.matrix(MFEXP~., test1)
ynew=test1$MFEXP
pred=predict(fit.ridge, xnew)
rmse=sqrt(apply((ynew-pred)^2, 2, mean))
plot(log(fit.ridge$lambda), rmse, type="b", xlab="log(lambda")
lam.best=fit.ridge$lambda[order(rmse)[1]]
lam.best
coef(fit.ridge, s=lam.best)
# final equation based on full dataset
y.full=q1$MFEXP
x.full<- model.matrix(MFEXP~., q1)
ridge.fin=glmnet(x.full, y.full, alpha=0, lambda=lam.best)

## Second note on Ridge Regression
library(car)
library(MASS)
model.ridge <- lm.ridge(MFEXP ~ ., data=train1, lambda = seq(0,2000,0.1))
plot(seq(0,2000,0.1), model.ridge$GCV, main="GCV of Ridge Regression", type="l", xlab=expression(lambda), ylab="GCV")
# The optimal lambda is given by
lambda.ridge <- seq(0,10,0.1)[which.min(model.ridge$GCV)]
# Plot the coefficients against lambda
colors <- rainbow(13)
matplot(seq(0,2000,0.1), coef(model.ridge)[,-1], xlim=c(0,2100), type="l",xlab=expression(lambda), 
        ylab=expression(hat(beta)), col=colors, lty=1, lwd=2, main="Ridge coefficients")
abline(v=lambda.ridge, lty=2)
abline(h=0, lty=2)
text(rep(10, 13), coef(model.ridge)[length(seq(0,10,0.1)),-1], colnames(train)[-13], pos=4, col=colors)
beta.ridge <- coef(model.ridge)[which.min(model.ridge$GCV),]
resid.ridge <- train1$MFEXP - beta.ridge[1] - as.matrix(train1[,1:12])%*%beta.ridge[2:13]
# To find df
d <- svd(as.matrix(training[,1:8]))$d
df <- 67 - sum(d^2/(lambda.ridge+d^2))
rss.ridge <- sum(resid.ridge^2)/df

## Lasso regression  rss+lambda*sum of absolute value of coefficients
fit.lasso=glmnet(x, y)
plot(fit.lasso, xvar="lambda", label=T)  # top: number of non-zero varialbes, shrinkage and selection
cv.lasso=cv.glmnet(x,y)
plot(fit.lasso, xvar="dev", label=T) #deviance explained == rsq
plot(cv.lasso)
coef(cv.lasso)
# validation approach
test.mat<- model.matrix(MFEXP~., data=test1)
val.errors<- rep(NA, length(fit.lasso$lambda))
for (i in 1:length(fit.lasso$lambda)){
  coefi<- fit.lasso$beta[,i]
  pred<- test.mat[,names(coefi)]%*%coefi
  val.errors[i]<- mean((test1$MFEXP-pred)^2)
}
val.errors

pred=predict(fit.lasso, xnew)
rmse=sqrt(apply((ynew-pred)^2, 2, mean))
plot(log(fit.lasso$lambda), rmse, type="b", xlab="log(lambda")
lam.best=fit.lasso$lambda[order(rmse)[1]]
lam.best
coef(fit.lasso, s=lam.best)
# final equation based on full dataset
y.full=q1$MFEXP
x.full<- model.matrix(MFEXP~., q1)
lasso.fin=glmnet(x.full, y.full, lambda=lam.best)


########################################
# For QUESTION 2 Choice of Main Heating and Cooling Equipment
# method PCA -- predict MAINHT
q2 <- subset(pj02, select=c("PBA", "SQFT", "WLCNS", "RFCNS", "GLSSPC", "NFLOOR", "YRCON", "WKHRS", "NWKER", "HDD65",  "MAINHT"))
train2 <- q2[index,]
test2 <- q2[-index,]
train2 <- as.data.frame(train2)
test2 <- as.data.frame(test2)
library(pls)
library(devtools)

## PCR
pr.out<- prcomp(train2[,1:10], scale=T)
names(pr.out)
biplot(pr.out, scale=0)
# or 
pr.train=prcomp(train2[,1:10], center=T, scale=T)
print(pr.train)  
# The print method returns the standard deviation of each of the four PCs, and their rotation (or loadings), 
# which are the coefficients of the linear combinations of the continuous variables.
plot(pr.train, type="l")
# The plot method returns a plot of the variances (y-axis) associated with the PCs (x-axis). 
# The Figure below is useful to decide how many PCs to retain for further analysis.
summary(pr.train)
# The summary method describe the importance of the PCs.
## Biplot for PCR
pred.pca=predict(pr.train, newdata=test2[,1:10])
install_github("ggbiplot", "vqv")
install.packages("ggbiplot")
library(ggbiplot)
g<- ggbiplot(pr.train, obs.scale=1, var.scale=1, groups=train2$MAINHT, ellipse=T, circle=T)
# g<- g+scale_color_discrete(name="")
g<- g+theme(legend.direction="horizontal", legend.position='top')
print(g)

pcr.fit=pcr(MAINHT~., data=train2, scale=T, validation="CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type="MSEP", type="b")
# The smallest CV error occurs when M=8 components are used
pcr.pred=round(predict(pcr.fit, test2, ncomp=3))
mean(pcr.pred==test2$MAINHT)

# validation approach
test.mat<- model.matrix(MAINHT~., data=test2)
val.errors<- rep(NA, 10)
for (i in 1:10){
  pred<- round(predict(pcr.fit, test2, ncomp=i))
  val.errors[i]<- mean(test2$MAINHT==pred)
}
val.errors
plot(val.errors, type="b", xlab="Number of Components", ylab="Validation Set Prediction Error")
points(which.min(val.errors), val.errors[which.min(val.errors)], col="red", pch=20)
# the Validation set prediction error indicate model with 10 components fit the best
pcr.fit$coefficients[,,10]

# fit pcr on full dataset
# CV approach
y.full=q2$MAINHT
x.full<- model.matrix(MAINHT~., q2)
pcr.fit1=pcr(y.full~x.full, ncomp=10)
summary(pcr.fit)
coef(pcr.fit)
# validation set prediction error approach
pcr.fit2=pcr(y.full~x.full, ncomp=10)
summary(pcr.fit2)
coef(pcr.fit2)

########################################
# For QUESTION 3 Characteristics of High-Energy-Efficient Building and Low-Energy-Efficient Building
# method SVM neural network 
q3 <- subset(pj02, select=c("PBA", "SQFT", "WLCNS", "RFCNS", "GLSSPC", "NFLOOR", "YRCON", "WKHRS", "NWKER", "MAINHT", "MAINCL", "HDD65", "CDD65", "MFEXP"))
# create new variable energy efficiency defined as annual enery expenditure/square footage
q3$Eff <- q3$MFEXP/q3$SQFT
# we define the mean as dividing point, below as high-enenrgy-efficient building, and low-energy-efficient building if above the mean

for(i in 1:nrow(q3)){
if(q3$Eff[i] >= mean(q3$Eff)){q3$RE[i] = 1}
if(q3$Eff[i] < mean(q3$Eff)){q3$RE[i] = 0}
}
q3$RE <- as.factor(q3$RE)
q4 <- subset(q3, select=c("PBA", "WLCNS", "RFCNS", "GLSSPC", "NFLOOR", "YRCON", "WKHRS", "NWKER", "MAINHT", "MAINCL", "HDD65", "CDD65", "Eff"))
q3 <- subset(q3, select=c("WLCNS", "RFCNS", "RE"))

train3 <- q3[index,]
test3 <- q3[-index,]
train3 <- as.data.frame(train3)
test3 <- as.data.frame(test3)

library(e1071)
svmfit <- svm(RE~., data=train3, kernel="radial", gamma=1, scale=F)
plot(svmfit, train3)
summary(svmfit)
tune.out <- tune(svm, RE~., data=train3, kernel="radial", ranges=list(cost=c(0.1,1,10,100,1000), gamma=c(0.5, 1,2,3,4)))
summary(tune.out)
# the optimal cost=0.1, gamma=0.5
library(ROCR)
svmfit.opt<- svm(RE~., data=train3, kernel="radial", cost=0.1, gamma=0.5, decision.values=T )
fitted <- attributes(predict(svmfit.opt, train3, decision.values = T))$decision.values
par(mfrow=c(1,2))
plot(fitted, train3, main="ROC plot for Training Data")


library(neuralnet)
train4 <- q4[index,]
test4 <- q4[-index,]
train4 <- as.data.frame(train4)
test4 <- as.data.frame(test4)
n <- names(train4)
f <- as.formula(paste("Eff ~", paste(n[!n %in% "Eff"], collapse="+")))
nn <- neuralnet(f, data=train4, hidden=3,linear.output = T)   #hidden可以自己改着玩儿。。。。
plot(nn)




