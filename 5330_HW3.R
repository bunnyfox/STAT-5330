#####################################################
# Regression Method Homework

#Zhuo LI zl5sn
#Jianhui Sun js9gu

library(data.table)
library(MASS)

setwd("/Users/lizhuo/Documents/UVA_Statistics/5330_Data Mining/Homework")
data=fread("prostate.data.txt")
head(data)
data$V1<- NULL
summary(data)
data=as.data.frame(data)
# check for missing data
sum(is.na(data))
# Standardize Dataset
set.seed(37)
sdata=scale(data[,c(-10)], center=T, scale=T)
# check whether standardized
colMeans(sdata)
apply(sdata, 2, sd)
sdata=as.data.frame(sdata)
dim(sdata)
# Splitting training and testing datasets
sdata$train=data$train
training=subset(sdata, train=="TRUE")
testing=subset(sdata, train=="FALSE")
training=training[,c(-10)]
testing=testing[,c(-10)]
sdata=sdata[,c(-10)]
sdata=as.data.frame(sdata)

#####################################################
## Best Subset
library(leaps)
regfit.best=regsubsets(lpsa~., data=training, nvmax=8)
reg.sum=summary(regfit.best)

# Plotting RSS, adjusted rsq, Cp and BIC
par(mfrow=c(2,2))
plot(reg.sum$rss, xlab="Number of Variables", ylab="RSS", type="b")
plot(reg.sum$adjr2, xlab="Number of Variables", ylab="Adjusted Rsq", type="b")
which.max(reg.sum$adjr2)
points(which.max(reg.sum$adjr2), reg.sum$adjr2[which.max(reg.sum$adjr2)], col="red", pch=20)
plot(reg.sum$cp, xlab="Number of Variables", ylab="Cp", type="b")
which.min(reg.sum$cp)
points(which.min(reg.sum$cp), reg.sum$cp[which.min(reg.sum$cp)], col="red", pch=20)
which.min(reg.sum$bic)
plot(reg.sum$bic, xlab="Number of Variables", ylab="BIC", type="b")
points(which.min(reg.sum$bic), reg.sum$bic[which.min(reg.sum$bic)], col="red", pch=20)
# or
plot(regfit.best, scale="r2")
plot(regfit.best, scale="adjr2")
plot(regfit.best, scale="Cp")
plot(regfit.best, scale="bic")

# Based on adjusted rsq and Mallow's Cp criteria, the model with 7 predictors are considered to be the best fit. Although BIC indicating otherwise, considering BIC
# penalizes adding more variables, and adding up to 7 variables does increasing the ajdusted rsq by a significant amount

# prediction error
# validation set approach
test.mat=model.matrix(lpsa~., data=testing)
val.errors=rep(NA, 8)
for (i in 1:8){
  coefi=coef(regfit.best, id=i)
  pred=test.mat[,names(coefi)]%*%coefi
  val.errors[i]=mean((testing$lpsa-pred)^2)
}
which.min(val.errors)
# best model based on validation set
regfit.best2=regsubsets(lpsa~., data=sdata, nvmax=8)
coef(regfit.best2,which.min(val.errors))

# cross-validation approach
predict.regsubsets=function(object, newdata, id){
  form=as.formula(object$call[[2]])
  mat=model.matrix(form, newdata)
  coefi=coef(object, id=id)
  xvars=names(coefi)
  mat[,xvars]%*%coefi
}
k=10
folds=sample(1:k, nrow(sdata), replace=T)
cv.errors=matrix(NA, k, 8, dimnames=list(NULL, paste(1:8)))
for (j in 1:k){
  best.fit=regsubsets(lpsa~., data=sdata[folds!=j,], nvmax=8)
  for (i in 1:8){
    pred=predict(best.fit, sdata[folds==j,], id=i)
    cv.errors[j,i]=mean((sdata$lpsa[folds==j]-pred)^2)
  }
}
mean.cv.errors=apply(cv.errors,2, mean)
# plot validation set approach and CV approach
par(mfrow=c(1,2))
plot(val.errors, type="b", xlab="Number of Predictors", ylab="Validation Set Error")
points(which.min(val.errors), val.errors[which.min(val.errors)], col="red", pch=20)
plot(mean.cv.errors, type="b", xlab="Number of Predictors", ylab="Cross-Validation Mean Error")
points(which.min(mean.cv.errors), mean.cv.errors[which.min(mean.cv.errors)], col="red", pch=20)
# CV indicating the model with 7 variables is the best
mean.cv.errors

# final model based on entire dataset
regfit.bestF=regsubsets(lpsa~., data=sdata, nvmax=8)
coef(regfit.bestF, 7)




#####################################################
# Stepwise
# Full model
regfit.full=lm(lpsa~., data=training)
# Null model
regfit.null=lm(lpsa~1, data=training)

## Forward
regfit.fwd=step(regfit.null, scope=list(lower=regfit.null, upper=regfit.full), direction="forward")
# model lm(formula = lpsa ~ lcavol + lweight + svi + lbph, data = training)


## Backward
regfit.bwd=step(regfit.full, direction="backward")
# model lm(formula = lpsa ~ lcavol + lweight + age + lbph + svi + lcp + pgg45, data = training)


## Hybrid
regfit.both=step(regfit.null, scope=list(upper=regfit.full), direction="both")
# model lm(formula = lpsa ~ lcavol + lweight + svi + lbph, data = training)

# COMPARISON:
# Both forward and hybrid give the same final model, the backward method gives a different one

# prediction error for the best model on testing set    lpsa ~ lcavol + lweight + svi + lbph
pred.step=predict(regfit.both, testing)
pred.err.step=mean((pred.step-testing$lpsa)^2)
# prediction error for the backward model on testing set    lpsa ~ lcavol + lweight + age + lbph + svi + lcp + pgg45
pred.stepb=predict(regfit.bwd, testing)
pred.err.stepb=mean((pred.stepb-testing$lpsa)^2)

# Cross-Validation for best model
#10-fold cv
n<- dim(sdata)[1]
m<- 10
len<- floor(n/m)
pred.err<- matrix(0,1,m)
for(k in 1:m){
  ind<- (len*k - len +1):(len*k)
  trainingi<- sdata[-ind, ]
  testingi<- sdata[ind,]
  fit<- lm(lpsa~lcavol + lweight + svi + lbph, trainingi)
  yhat<- predict(fit, testingi)
  pred.err[k]<- mean((testingi$lpsa - yhat)^2)
}
pred.err


# further compare three models
anova(regfit.bwd, regfit.both)   # not sigfinicant when adding three extra variables based on paritial F-test

## Final model using full dataset
regfit.step=lm(lpsa ~ lcavol + lweight + svi + lbph, sdata)
regfit.step$coef


### Further note
model.ls <- lm(lpsa ~ ., data=training)
rss.ls <- sum(model.ls$resid^2)/model.ls$df.residual

model.backward <- step(model.ls, direction="backward")
rss.backward <- sum(model.backward$resid^2)/model.backward$df.residual
# So backward selection using AIC drops "gleason" from the model. The final AIC
# is AIC -58.33.

scope <- list(upper=~lcavol+lweight+age+lbph+svi+lcp+gleason+pgg45, lower=~.)
model.forward <- step(lm(lpsa ~ 1, data=training), scope, direction="forward")
rss.forward <- sum(model.forward$resid^2)/model.forward$df.residual
# So we conclude that forward selection using AIC keeps lcavol, lweight, svi and lbph.
# The AIC is -57.06.

model.both=step(lm(lpsa ~ 1, data=training), scope, direction="both")
rss.both <- sum(model.both$resid^2)/model.both$df.residual
# So we keep lpsa ~ lcavol + lweight + svi + lbph
# The AIC is -57.06

r2 <- list()
AICs <- list()
for(i in 1:8){
  indexes <- combn(8,i)
  currentr2 <- NULL
  currentAIC <- NULL
  for(j in 1:dim(indexes)[2]){
    temp.model <- lm(lpsa ~ ., data=training[,c(indexes[,j], 9)])
    currentr2[j] <- summary(temp.model)$r.squared
    currentAIC[j] <- AIC(temp.model)
  }
  r2[[i]] <- currentr2
  AICs[[i]] <- currentAIC
}
compare <- function(set){
  s <- length(set)
  temp <- combn(8,s)
  check <- NULL
  for(i in 1:dim(temp)[2]){
    check[i] <- all(temp[,i]==set)
  }
  return(which(check))
}
backward <- compare(c(1:6,8))
forward <- c(compare(1), compare(1:2), compare(c(1,2,5)), compare(c(1,2,4,5)))
both <- c(compare(1), compare(1:2), compare(c(1,2,5)), compare(c(1,2,4,5)))

r2.b <- c(r2[[7]][backward], r2[[8]])
r2.f <- c(r2[[1]][forward[1]], r2[[2]][forward[2]], r2[[3]][forward[3]], r2[[4]][forward[4]])
r2.bo <- c(r2[[1]][both[1]], r2[[2]][both[2]], r2[[3]][both[3]], r2[[4]][both[4]])
AICs.b <- c(AICs[[7]][backward], AICs[[8]])
AICs.f <- c(AICs[[1]][forward[1]], AICs[[2]][forward[2]], AICs[[3]][forward[3]], AICs[[4]][forward[4]])
AICs.bo <- c(AICs[[1]][both[1]], AICs[[2]][both[2]], AICs[[3]][both[3]], AICs[[4]][both[4]])

# how backward/forward performs!  # forward and hybrid gives the same results and follow the same steps, thus eliminated hybrid for the following analysis
x11(width=10, height=5)
layout(matrix(1:2, ncol=2))
plot(0, xlim=c(0,9), ylim=c(0,0.8), type="n", ylab=expression(r^2), main="Fitting criteria")
for(i in 1:8){
  points(rep(i, length(r2[[i]])), r2[[i]], pch=21, bg="Grey")
}
points(7:8, r2.b, bg="Red", col="Red", pch=21, type="o")
points(1:4, r2.f, bg="Blue", col="Blue", pch=21, type="o")
plot(0, xlim=c(0,9), ylim=c(130,217), type="n", ylab="AIC", main="AIC")
for(i in 1:8){
  points(rep(i, length(AICs[[i]])), AICs[[i]], pch=21, bg="Grey")
}
points(7:8, AICs.b, bg="Red", col="Red", pch=21, type="o")
points(1:4, AICs.f, bg="Blue", col="Blue", pch=21, type="o")


#####################################################
# Shinkage & Dimension Reduction Model
library(glmnet)

### Ridge Regression
x<- model.matrix(lpsa~., training)
y<- training$lpsa
#alpha=0 for ridge, =1 (default) for lasso
fit.ridge<- glmnet(x, y, alpha=0)   #rss + lambda*sum sq of coefficients, smallest lambda==ols coefficients, keep all the bariable, shrink the coefficient to 0
#cv of ridge to determine lambda
cv.ridge<- cv.glmnet(x, y, alpha=0, type.measure="mse")
par(mfrow=c(1,2))
plot(fit.ridge, xvar="lambda")
plot(cv.ridge)  # two lines: one minimum, one std away from min
# prediction error
# validation approach
test.mat<- model.matrix(lpsa~., data=testing)
val.errors<- rep(NA, length(fit.ridge$lambda))
for (i in 1:length(fit.ridge$lambda)){
  coefi<- fit.ridge$beta[,i]
  pred<- test.mat[,names(coefi)]%*%coefi
  val.errors[i]<- mean((testing$lpsa-pred)^2)
}
val.errors
par(mfrow=c(1,1))
plot(fit.ridge$lambda, val.errors)
plot(log(fit.ridge$lambda), val.errors)

xnew=model.matrix(lpsa~., testing)
ynew=testing$lpsa
pred=predict(fit.ridge, xnew)
rmse=sqrt(apply((ynew-pred)^2, 2, mean))
plot(log(fit.ridge$lambda), rmse, type="b", xlab="log(lambda")
lam.best=fit.ridge$lambda[order(rmse)[1]]
lam.best
coef(fit.ridge, s=lam.best)
# final equation based on full dataset
y.full=sdata$lpsa
x.full<- model.matrix(lpsa~., sdata)
ridge.fin=glmnet(x.full, y.full, alpha=0, lambda=lam.best)

## Second note on Ridge Regression
library(car)
library(MASS)
model.ridge <- lm.ridge(lpsa ~ ., data=training, lambda = seq(0,1000,0.1))
plot(seq(0,1000,0.1), model.ridge$GCV, main="GCV of Ridge Regression", type="l", xlab=expression(lambda), ylab="GCV")
# The optimal lambda is given by
lambda.ridge <- seq(0,10,0.1)[which.min(model.ridge$GCV)]
# Plot the coefficients against lambda
colors <- rainbow(8)
matplot(seq(0,1000,0.1), coef(model.ridge)[,-1], xlim=c(0,1100), type="l",xlab=expression(lambda), 
        ylab=expression(hat(beta)), col=colors, lty=1, lwd=2, main="Ridge coefficients")
abline(v=lambda.ridge, lty=2)
abline(h=0, lty=2)
text(rep(10, 9), coef(model.ridge)[length(seq(0,10,0.1)),-1], colnames(training)[-9], pos=4, col=colors)
beta.ridge <- coef(model.ridge)[which.min(model.ridge$GCV),]
resid.ridge <- training$lpsa - beta.ridge[1] - as.matrix(training[,1:8])%*%beta.ridge[2:9]
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
test.mat<- model.matrix(lpsa~., data=testing)
val.errors<- rep(NA, length(fit.lasso$lambda))
for (i in 1:length(fit.lasso$lambda)){
  coefi<- fit.lasso$beta[,i]
  pred<- test.mat[,names(coefi)]%*%coefi
  val.errors[i]<- mean((testing$lpsa-pred)^2)
}
val.errors

pred=predict(fit.lasso, xnew)
rmse=sqrt(apply((ynew-pred)^2, 2, mean))
plot(log(fit.lasso$lambda), rmse, type="b", xlab="log(lambda")
lam.best=fit.lasso$lambda[order(rmse)[1]]
lam.best
coef(fit.lasso, s=lam.best)
# final equation based on full dataset
y.full=sdata$lpsa
x.full<- model.matrix(lpsa~., sdata)
lasso.fin=glmnet(x.full, y.full, lambda=lam.best)

### Second note on Lasso
library(lars)
y <- as.numeric(training[,9])
x <- as.matrix(training[,1:8])
model.lasso <- lars(x, y, type="lasso")
lambda.lasso <- c(model.lasso$lambda,0)
beta <- coef(model.lasso)
library(colorspace)
colors <- rainbow_hcl(8, c = 65, l = 65)
matplot(lambda.lasso, beta, xlim=c(8,-2), type="o", pch=20, xlab=expression(lambda), 
        ylab=expression(hat(beta)), col=colors)
text(rep(-0, 9), beta[9,], colnames(x), pos=4, col=colors)
abline(v=lambda.lasso[4], lty=2)
abline(h=0, lty=2)
#
beta.lasso <- beta[4,]
resid.lasso <- training$lpsa - predict(model.lasso, as.matrix(training[,1:8]), s=4, type="fit")$fit
rss.lasso <- sum(resid.lasso^2)/(67-4)



## Principal Component
library(pls)
library(devtools)
## PCR
boxplot(training)
pr.out<- prcomp(training[,1:8], scale=T)
names(pr.out)
biplot(pr.out, scale=0)
# or 
pr.train=prcomp(training[,1:8], center=T, scale=T)
print(pr.train)  
# The print method returns the standard deviation of each of the four PCs, and their rotation (or loadings), 
# which are the coefficients of the linear combinations of the continuous variables.
plot(pr.train, type="l")
# The plot method returns a plot of the variances (y-axis) associated with the PCs (x-axis). 
# The Figure below is useful to decide how many PCs to retain for further analysis.
summary(pr.train)
# The summary method describe the importance of the PCs.
## Biplot for PCR
pred.pca=predict(pr.train, newdata=testing[,1:8])
install_github("ggbiplot", "vqv")
library(ggbiplot)
g<- ggbiplot(pr.train, obs.scale=1, var.scale=1, groups=training$lpsa, ellipse=T, circle=T)
# g<- g+scale_color_discrete(name="")
g<- g+theme(legend.direction="horizontal", legend.position='top')
print(g)

pcr.fit=pcr(lpsa~., data=training, scale=T, validation="CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type="MSEP", type="b")
# The smallest CV error occurs when M=8 components are used
pcr.pred=predict(pcr.fit, testing, ncomp=8)
mean((pcr.pred-testing$lpsa)^2)

# validation approach
test.mat<- model.matrix(lpsa~., data=testing)
val.errors<- rep(NA, 8)
for (i in 1:8){
  coefi<- pcr.fit$coefficients[,,i]
  pred<- test.mat[,names(coefi)]%*%coefi
  val.errors[i]<- mean((testing$lpsa-pred)^2)
}
val.errors
plot(val.errors, type="b", xlab="Number of Components", ylab="Validation Set Prediction Error")
points(which.min(val.errors), val.errors[which.min(val.errors)], col="red", pch=20)
# the Validation set prediction error indicate model with 7 components fit the best
pcr.fit$coefficients[,,7]

# fit pcr on full dataset
# CV approach
y.full=sdata$lpsa
x.full<- model.matrix(lpsa~., sdata)
pcr.fit1=pcr(y.full~x.full, ncomp=8)
summary(pcr.fit)
coef(pcr.fit)
# validation set prediction error approach
pcr.fit2=pcr(y.full~x.full, ncomp=7)
summary(pcr.fit2)
coef(pcr.fit2)

### Second Note on PCR
model.pcr <- pcr(lpsa ~ .,7, data = training)
summary(model.pcr)
beta.pcr <- drop(coef(model.pcr))
resid.pcr <- drop(model.pcr$resid)[,7]
rss.pcr <- sum(resid.pcr^2)/(67-7)

##################
# Partial Least Square
pls.fit=plsr(lpsa~., data=training, validation="CV")
summary(pls.fit)
validationplot(pls.fit, val.type = "MSEP")
# Cross-validation indicating the best model would be with 8 components
pls.pred=predict(pls.fit, testing, ncomp=8)
mean((pls.pred-testing$lpsa)^2)
pls.fit.full=plsr(y.full~x.full, ncomp=8)
summary(pls.fit.full)
coef(pls.fit.full)
# validation set test average prediction error approach
test.mat<- model.matrix(lpsa~., data=testing)
val.errors<- rep(NA, 8)
for (i in 1:8){
  coefi<- pls.fit$coefficients[,,i]
  pred<- test.mat[,names(coefi)]%*%coefi
  val.errors[i]<- mean((testing$lpsa-pred)^2)
}
val.errors
plot(val.errors, type="b", xlab="Number of Components", ylab="Validation Set Prediction Error")
points(which.min(val.errors), val.errors[which.min(val.errors)], col="red", pch=20)
# validation set test average prediction error indicating that the model with 3 components are the best
pls.fit.full2=plsr(y.full~x.full, ncomp=3)
summary(pls.fit.full2)
coef(pls.fit.full2)

## Second note on PLS
model.pls <- plsr(lpsa ~ .,3, data = training, method = "oscorespls")
summary(model.pls)
beta.pls <- drop(coef(model.pls))
resid.pls <- drop(model.pls$resid)[,3]
rss.pls <- sum(resid.pls^2)/(67-3)




#########################################################################################
# #########################
# # COMPARISON OF FITTING #
# #########################
#
# choose the smallest one
rss.ls
rss.backward
rss.forward
rss.ridge
rss.lasso
rss.pcr
rss.pls


# ############################
# # COMPARISON OF PREDICTION #
# ############################
#

y.new <- testing$lpsa

pss.ls <- sum((y.new - predict(model.ls, testing[,1:8]))^2)
pss.backward <- sum((y.new - predict(model.backward, testing[,1:8]))^2)
pss.forward <- sum((y.new - predict(model.forward, testing[,1:8]))^2)
pss.ridge <- sum((y.new - beta.ridge[1] - as.matrix(testing[,1:8])%*%beta.ridge[2:9])^2)
pss.lasso <- sum((y.new - predict(model.lasso, as.matrix(testing[,1:8]), s=4, type="fit")$fit)^2)
pss.pcr <- sum((y.new - drop(predict(model.pcr, testing[,1:8], 7)))^2)
pss.pls <- sum((y.new - drop(predict(model.pls, testing[,1:8], 3)))^2)


pss.ls
pss.backward
pss.forward
pss.ridge
pss.lasso
pss.pcr
pss.pls

