library(mvtnorm)
library(copula)
library(MASS)
library(fCopulae)
library(fGarch)
library(mnormt)

nasdaq = read.csv("O:\\18WIN\\STATS 509\\HW4\\Nasdaq_wklydata_92-12.csv",header = T)
sp400 = read.csv("O:\\18WIN\\STATS 509\\HW4\\SP400Mid_wkly_92-12.csv",header = T)
nas_adjclose = nasdaq$Adj.Close
sp_adjclose = sp400$Adj.Close
nas_log = diff(log(nas_adjclose))
sp_log = diff(log(sp_adjclose))

#a) multivariate normal distribution
myxlim = c(-0.2,0.2)
myylim = c(-0.2,0.2)
mean_nas = mean(nas_log)
sd_nas = sd(nas_log)
mean_sp = mean(sp_log)
sd_sp = sd(sp_log)
mu = c(mean(nas_log),mean(sp_log))
mu
sigma = var(cbind(nas_log,sp_log))
sigma

n = length(nas_log)
nsim = rmvnorm(n,mu,sigma)
par(mfrow=c(1,2))
plot(nas_log,sp_log,xlim=myxlim,ylim=myylim,xlab="Nasdaq",ylab="SP400")
plot(nsim,xlim=myxlim,ylim=myylim,xlab="X-simulation",ylab="Y-Simulation")
qqnorm(nas_log,main="NASDAQ log return")
qqline(nas_log)
qqnorm(sp_log,main="SP400 log return")
qqline(sp_log)
#empirical
cdf_nas = pnorm(nas_log,mean_nas,sd_nas)
cdf_sp = pnorm(sp_log,mean_sp,sd_sp)
dem = pempiricalCopula(cdf_nas,cdf_sp)
contour(dem$x,dem$y,dem$z,main="Multivariate Normal",col='blue',lty=1,lwd=1,nlevel=20)
#theoretical
theo_norm = mvrnorm(1e6,mu,sigma)
theo_1 = theo_norm[,1]
theo_2 = theo_norm[,2]
cdf_theo_1 = pnorm(theo_1,mean(theo_1),sd(theo_1))
cdf_theo_2 = pnorm(theo_2,mean(theo_2),sd(theo_2))
dem_theo = pempiricalCopula(cdf_theo_1,cdf_theo_2)
contour(dem_theo$x,dem_theo$y,dem_theo$z,main="Multivariate Normal",col='red',lty=2,lwd=1,add=TRUE,nlevel=20)

#b) multivariate t distribution
combination = cbind(nas_log, sp_log)
df = seq(1, 8, 0.01)
n = length(df)
loglik_max = rep(0, n)
for(i in 1:n){
  fit = cov.trob(combination, nu = df[i])
  mu = as.vector(fit$center)
  sigma = matrix(fit$cov, nrow = 2)
  loglik_max[i] = sum(log(dmt(combination, mean=fit$center, S=fit$cov, df=df[i])))
}
plot(df, loglik_max, xlab='nu', ylab='Profile-likelihood')
##df
nuest = df[which.max(loglik_max)]
nuest
##CI of df
position = which((loglik_max[which.max(loglik_max)]-loglik_max) <= 0.5*qchisq(0.95, 1))
lower_bound = df[position[1]]
upper_bound = df[position[length(position)]]
c(lower_bound,upper_bound)

par(mfrow=c(1,2))
quantv = (1/n)*seq(.5,n-.5,1)
qqplot(sort(nas_log),qt(quantv,nuest),main="NASDAQ Q-Q Plot")
abline(lm(qt(c(.25,.75),nuest)~quantile(nas_log,c(.25,.75))))
qqplot(sort(sp_log),qt(quantv,nuest),main="SP400 Q-Q Plot")
abline(lm(qt(c(.25,.75),nuest)~quantile(sp_log,c(.25,.75))))
#empirical
cdf_nas_t = pstd(nas_log,mean_nas,sd_nas,nuest)
cdf_sp_t = pstd(sp_log,mean_sp,sd_sp,nuest)
dem_t = pempiricalCopula(cdf_nas_t,cdf_sp_t)
contour(dem_t$x,dem_t$y,dem_t$z,main="Multivariate-t",col='blue',lty=1,lwd=1,nlevel=20)
#theoretical
mu = c(mean(nas_log),mean(sp_log))
sigma = var(cbind(nas_log,sp_log))
lambda = sigma*(nuest-2)/nuest
theo_t = rmt(1e6,mu,lambda,nuest)
theo_1 = theo_t[,1]
theo_2 = theo_t[,2]
cdf_theo_1 = pstd(theo_1,mean(theo_1),sd(theo_1),nuest)
cdf_theo_2 = pstd(theo_2,mean(theo_2),sd(theo_2),nuest)
dem_theo = pempiricalCopula(cdf_theo_1,cdf_theo_2)
contour(dem_theo$x,dem_theo$y,dem_theo$z,col='red',lty=2,lwd=1,add=TRUE,nlevel=20)

#d)-normal
mu = c(mean(nas_log),mean(sp_log))
sigma = var(cbind(nas_log,sp_log))
theo_norm = mvrnorm(1e6,mu,sigma)
datat = 0.5*theo_norm[,1]+0.5*theo_norm[,2]
VaR = -quantile(datat,0.001)*1e7
VaR
#d)-t
theo_t = rmt(1e6,mu,lambda,nuest)
datat = 0.5*theo_t[,1]+0.5*theo_t[,2]
VaR = -quantile(datat,0.001)*1e7
VaR

#e)-t
set.seed(2015)
w = seq(0,1,0.01)
n = length(w)
VaRv=rep(0,n)
exp_return = rep(0,n)
var=rep(0,n)
data_sim = rmt(1e4,mu,lambda,nuest)
for(i in 1:n){
  datat = w[i]*data_sim[,1]+(1-w[i])*data_sim[,2]
  VaRv[i] = -quantile(datat,0.002)
  exp_return[i] = mean(datat)
  var[i] = sd(nas_log)^2*w[i]^2+sd(sp_log)^2*(1-w[i])^2+2*sd(nas_log)*sd(sp_log)*cor(nas_log,sp_log)*w[i]*(1-w[i])
}
w_exp = w[which.max(exp_return)]
exp_max = exp_return[which.max(exp_return)]*1e7
w_var = w[which.min(var)]
var_min = var[which.min(var)]
wmax = w[which.min(VaRv)]
VaR = VaRv[which.min(VaRv)]*1e7
w_exp;w_var;wmax

#e)-norm
set.seed(2015)
w = seq(0,1,0.01)
n = length(w)
VaRv=rep(0,n)
exp_return = rep(0,n)
var=rep(0,n)
data_sim = mvrnorm(1e4,mu,sigma)
for(i in 1:n){
  datat = w[i]*data_sim[,1]+(1-w[i])*data_sim[,2]
  VaRv[i] = -quantile(datat,0.002)
  exp_return[i] = mean(datat)
  var[i] = sd(nas_log)^2*w[i]^2+sd(sp_log)^2*(1-w[i])^2+2*sd(nas_log)*sd(sp_log)*cor(nas_log,sp_log)*w[i]*(1-w[i])
}
w_exp = w[which.max(exp_return)]
exp_max = exp_return[which.max(exp_return)]*1e7
w_var = w[which.min(var)]
var_min = var[which.min(var)]
wmax = w[which.min(VaRv)]
VaR = VaRv[which.min(VaRv)]*1e7
w_exp;w_var;wmax