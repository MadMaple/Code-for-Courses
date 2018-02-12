library(VaRES)
#c
w = seq(0, 1, by=0.01)
mu = 0.02*w+0.03*(1-w)
var = 0.03^2*w^2+0.04^2*(1-w)^2+2*0.03*0.04*0.5*w*(1-w)
q = rep(0,length(w))
for (i in 1:length(w)) {
  q[i] = esnormal(0.005,mu[i],sqrt(var[i]))
}
num = which.max(q)
"weight";w[num]
"expected_shortfall";(-q[num])*1e6
"VaR";-qnorm(0.005, mu[num], sqrt(var[num]))*1e6
#d
shortfall = rep(0,1000)
exp_shortfall = rep(0,length(w))
set.seed(2015)
for (i in 1:length(w)) {
  lambda = sqrt(var[i])*sqrt(4/6)
  t_std = qt(0.005, 6)
  t_new = t_std*lambda+mu[i]
  q[i] = exp(t_new)-1
  rt = rt(1000, 6)
  rt_new = rt*lambda+mu[i]
  for (n in 1:1000){
    return = exp(rt_new)-1
    index = which(return < q[i])
    shortfall[n] = mean(return[index])
  }
  exp_shortfall[i] = mean(na.omit(shortfall))
}
num = which.min(exp_shortfall)
"Weight";w[num]
"expected shorfall";-exp_shortfall[num]*1e6
"VaR";-q[num]*1e6

