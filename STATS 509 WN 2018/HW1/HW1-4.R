dat = read.csv("O:\\18WIN\\STATS 509\\datasets\\MCD_PriceDaily.csv",header=T)
nd = dim(dat)
n = nd[1]
r = dat$Adj.Close[-1]/dat$Adj.Close[-n] - 1
logr = diff(log(dat$Adj.Close))
plot(r,logr,xlab="returns",ylab="log_returns")
abline(0,1)

niter = 1e4
below = rep(0,niter)
set.seed(2015)
for (i in 1:niter)
{
  r = rnorm(20,mean = mean(logr),sd = sd(logr))
  logPrice = log(93.07) + cumsum(r)
  minlogP = min(logPrice)
  below[i] = as.numeric((minlogP) < log(85)) #or 84.5 in problem 17
}
mean(below)