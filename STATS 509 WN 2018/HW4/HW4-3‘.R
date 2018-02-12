ct = tCopula(ft@estimate[1],dim=2,dispstr="un",df=ft@estimate[2])
set.seed(2015)
uvsim = rCopula(100000,ct)
w = seq(0,1,0.01)
n = length(w)
VaRv = rep(0,n)
data_sim = cbind(qstd(uvsim[,1],mean=est.nas[1],sd=est.nas[2],nu=est.nas[3]),qstd(uvsim[,2],mean=est.sp[1],sd=est.sp[2],nu=est.sp[3]))
for(i in 1:n){
  datat = w[i]*data_sim[,1]+(1-w[i])*data_sim[,2]
  VaRv[i] = -quantile(datat,0.001)
}
plot(w,VaRv,xlab="w",ylab="VaR",main="VaR vs. w")
wmax = w[which.min(VaRv)]
VaR = VaRv[which.min(VaRv)]*1e7
c(wmax,VaR)


cn = tCopula(fnorm@estimate[1],dim=2,dispstr="un")
set.seed(2015)
uvsim = rCopula(100000,cn)
w = seq(0,1,0.01)
n = length(w)
VaRv = rep(0,n)
data_sim = cbind(qstd(uvsim[,1],mean=est.nas[1],sd=est.nas[2],nu=est.nas[3]),qstd(uvsim[,2],mean=est.sp[1],sd=est.sp[2],nu=est.sp[3]))
for(i in 1:n){
  datat = w[i]*data_sim[,1]+(1-w[i])*data_sim[,2]
  VaRv[i] = -quantile(datat,0.001)
}
plot(w,VaRv,xlab="w",ylab="VaR",main="VaR vs. w")
wmax = w[which.min(VaRv)]
VaR = VaRv[which.min(VaRv)]*1e7
c(wmax,VaR)