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

est.nas = as.numeric(fitdistr(nas_log,"t")$estimate)
est.sp = as.numeric(fitdistr(sp_log,"t")$estimate)
est.nas[2] = est.nas[2]*sqrt(est.nas[3]/(est.nas[3]-2))
est.sp[2] = est.sp[2]*sqrt(est.sp[3]/(est.sp[3]-2))
data = cbind(pstd(nas_log,mean=est.nas[1],sd=est.nas[2],nu=est.nas[3]),pstd(sp_log,mean=est.sp[1],sd=est.sp[2],nu=est.sp[3]))
cor_tau = cor(data[,1],data[,2])

fnorm = fitCopula(data=data,copula=normalCopula(dim=2),optim.method="BFGS")
summary(fnorm)

dem = pempiricalCopula(data[,1],data[,2])
contour(dem$x,dem$y,dem$z,main="Multivariate Normal",col='blue',lty=1,lwd=1,nlevel=20)
cn = normalCopula(fnorm@estimate[1],dim=2,dispstr = "un")
utdis = rCopula(100000,cn)
demt = pempiricalCopula(utdis[,1],utdis[,2])
contour(demt$x,demt$y,demt$z,col='red',lty=2,lwd=1,add=TRUE,nlevel=20)

#b) multivariate t distribution
par(mfrow=c(1,2))
quantv = (1/n)*seq(.5,n-.5,1)
qqplot(sort(nas_log),qt(quantv,est.nas[3]),main="NASDAQ Q-Q Plot")
abline(lm(qt(c(.25,.75),est.nas[3])~quantile(nas_log,c(.25,.75))))
qqplot(sort(sp_log),qt(quantv,est.sp[3]),main="SP400 Q-Q Plot")
abline(lm(qt(c(.25,.75),est.sp[3])~quantile(sp_log,c(.25,.75))))

cop_t = tCopula(cor_tau,dim=2,dispstr="un",df=4)
ft = fitCopula(cop_t,optim.method="L-BFGS-B",data=data,start=c(cor_tau,5))
summary(ft)

dem = pempiricalCopula(data[,1],data[,2])
contour(dem$x,dem$y,dem$z,main="Multivariate Student t",col='blue',lty=1,lwd=1,nlevel=20)
ct = tCopula(ft@estimate[1],dim=2,dispstr = "un",df=ft@estimate[2])
utdis = rCopula(100000,ct)
demt = pempiricalCopula(utdis[,1],utdis[,2])
contour(demt$x,demt$y,demt$z,col='red',lty=2,lwd=1,add=TRUE,nlevel=20)

data1=cbind(nas_log,sp_log)
df = seq(1, 7, 0.01)
n = length(df)
loglik = rep(0,n)
for(i in 1:n){
  fit = cov.trob(data1,nu=df[i])
  mu = as.vector(fit$center)
  sigma = matrix(fit$cov,nrow=4)
  loglik[i] = sum(log(dmt(data1, mean=fit$center,S=fit$cov,df=df[i])))
}
plot(df,loglik)
which.min(abs(max(loglik) - 0.5*qchisq(0.95, 1)-loglik))
which.min(abs(max(loglik) - qchisq(0.95, 1)-loglik[-144]))
mean(df[144],df[145])
mean(df[247],df[248])

#c)
AIC_norm = -2*fnorm@loglik+2*2
AIC_t = -2*ft@loglik+2*2
c(AIC_norm,AIC_t)

#d)-normal
cn = tCopula(fnorm@estimate[1],dim=2,dispstr="un")
uvsim = rCopula(100000,cn)
data_sim = cbind(qstd(uvsim[,1],mean=est.nas[1],sd=est.nas[2],nu=est.nas[3]),qstd(uvsim[,2],mean=est.sp[1],sd=est.sp[2],nu=est.sp[3]))
datat = 0.5*data_sim[,1]+0.5*data_sim[,2]
VaR = -quantile(datat,0.001)*1e7
VaR

#d)-t
ct = tCopula(ft@estimate[1],dim=2,dispstr="un",df=ft@estimate[2])
uvsim = rCopula(100000,ct)
data_sim = cbind(qstd(uvsim[,1],mean=est.nas[1],sd=est.nas[2],nu=est.nas[3]),qstd(uvsim[,2],mean=est.sp[1],sd=est.sp[2],nu=est.sp[3]))
datat = 0.5*data_sim[,1]+0.5*data_sim[,2]
VaR = -quantile(datat,0.001)*1e7
VaR

#e)-max expected return
set.seed(2015)
ct = tCopula(ft@estimate[1],dim=2,dispstr="un",df=ft@estimate[2])
uvsim = rCopula(1e5,ct)
w = seq(0,1,0.00001)
n = length(w)
VaRv=rep(0,n)
exp_return = rep(0,n)
var=rep(0,n)
data_sim = cbind(qstd(uvsim[,1],mean=est.nas[1],sd=est.nas[2],nu=est.nas[3]),qstd(uvsim[,2],mean=est.sp[1],sd=est.sp[2],nu=est.sp[3]))
for(i in 1:n){
  datat = w[i]*data_sim[,1]+(1-w[i])*data_sim[,2]
  VaRv[i] = -quantile(datat,0.002)
  exp_return[i] = mean(datat)
  var[i] = sd(nas_log)^2*w[i]^2+sd(sp_log)^2*(1-w[i])^2+2*sd(nas_log)*sd(sp_log)*cor(nas_log,sp_log)*w[i]*(1-w[i])
}
w_exp = w[which.max(exp_return)]
exp_max = exp_return[which.max(exp_return)]*1e7
c(w_exp,exp_max)
w_var = w[which.min(var)]
var_min = var[which.min(var)]
c(w_var,var_min)
wmax = w[which.min(VaRv)]
VaR = VaRv[which.min(VaRv)]*1e7
c(wmax,VaR)