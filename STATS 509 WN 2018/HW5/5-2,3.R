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
len = length(nas_log)

# estimate seperate t distribution
est.nas = as.numeric(fitdistr(nas_log,"t")$estimate)
est.sp = as.numeric(fitdistr(sp_log,"t")$estimate)
est.nas;est.sp
par(mfrow=c(1,2))
quantv = (1/len)*seq(.5,len-.5,1)
qqplot(sort(nas_log),qt(quantv,est.nas[3]),main="NASDAQ Q-Q Plot")
abline(lm(qt(c(.25,.75),est.nas[3])~quantile(nas_log,c(.25,.75))))
qqplot(sort(sp_log),qt(quantv,est.sp[3]),main="SP400 Q-Q Plot")
abline(lm(qt(c(.25,.75),est.sp[3])~quantile(sp_log,c(.25,.75))))
est.nas[2] = est.nas[2]*sqrt(est.nas[3]/(est.nas[3]-2))
est.sp[2] = est.sp[2]*sqrt(est.sp[3]/(est.sp[3]-2))
# fit t-copula
data = cbind(pstd(nas_log,mean=est.nas[1],sd=est.nas[2],nu=est.nas[3]),pstd(sp_log,mean=est.sp[1],sd=est.sp[2],nu=est.sp[3]))
cor_tau = cor(data[,1],data[,2])
cop_t = tCopula(cor_tau,dim=2,dispstr="un",df=4)
ft = fitCopula(cop_t,optim.method="L-BFGS-B",data=data,start=c(cor_tau,5))
summary(ft)
# empirical and theoretical cdf
dem = pempiricalCopula(data[,1],data[,2])
contour(dem$x,dem$y,dem$z,main="Multivariate Student t",col='blue',lty=1,lwd=1,nlevel=20)
ct = tCopula(ft@estimate[1],dim=2,dispstr = "un",df=ft@estimate[2])
utdis = rCopula(100000,ct)
demt = pempiricalCopula(utdis[,1],utdis[,2])
contour(demt$x,demt$y,demt$z,col='red',lty=2,lwd=1,add=TRUE,nlevel=20)
# AIC
AIC_t_copula = -2*ft@loglik+2*2
AIC_nas = -2*fitdistr(nas_log,"t")$loglik+2*2
AIC_sp = -2*fitdistr(sp_log,"t")$loglik+2*2
AIC_t_copula+AIC_nas+AIC_sp

# Problem 3-a,b
set.seed(2015)
uvsim = rCopula(1e5,ct)
w = seq(-1,1,0.001)
n = length(w)
VaRv=rep(0,n)
var=rep(0,n)
data_sim = cbind(qstd(uvsim[,1],mean=est.nas[1],sd=est.nas[2],nu=est.nas[3]),qstd(uvsim[,2],mean=est.sp[1],sd=est.sp[2],nu=est.sp[3]))
for(i in 1:n){
  datat = w[i]*data_sim[,1]+(1-w[i])*data_sim[,2]
  VaRv[i] = -quantile(datat,0.005)
  var[i] = sd(nas_log)^2*w[i]^2+sd(sp_log)^2*(1-w[i])^2+2*sd(nas_log)*sd(sp_log)*cor(nas_log,sp_log)*w[i]*(1-w[i])
}
w_var = w[which.min(var)]
var_min = var[which.min(var)]
c(w_var,var_min)
wmax = w[which.min(VaRv)]
VaR = VaRv[which.min(VaRv)]
c(wmax,VaR)
shortfall = 0
count = 0
data_sim_2 = wmax*data_sim[,1] + (1-wmax)*data_sim[,2]
n = length(data_sim_2)
for (i in 1:n){
  if (data_sim_2[i] > VaR){
    shortfall = shortfall + data_sim_2[i]
    count = count + 1
  }
}
exp_shortfall = shortfall / count
exp_shortfall

# 3-(c)
VaR_nas = qstd(0.003, est.nas[1], est.nas[2], est.nas[3])
VaR_sp = qstd(0.003, est.sp[1], est.sp[2], est.sp[3])
return_nas = data_sim[,1]
return_sp = data_sim[,2]
count = 0
for (i in 1:1e5){
  if ( return_nas[i] < VaR_nas && return_sp[i] < VaR_sp){
    count = count + 1
  }
}
P = count/1e5
P
