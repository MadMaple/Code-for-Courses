#a
qdexp <- function(p,mu,lambda){
  quant1 <- qexp(0*p,lambda) + mu
  pn <- p[p<.5]
  pp <- p[p>.5]
  quant1[p>.5] <- qexp(2*pp-1,lambda) + mu
  quant1[p<.5] <- -qexp(1-2*pn,lambda) + mu
  quant1 
}
dat = read.csv("O:\\18WIN\\STATS 509\\HW1\\Nasdaq_daily_Jan1_2012_Dec31_2017.csv",header = T)
logr = diff(log(dat$Adj.Close))
mu = mean(logr)
sd = sd(logr)
lambda = sqrt(2)/sd
p = 0.005
q = qdexp(p,mu,lambda)
mu;sd;q
quantile(logr,0.005)

#b
rdexp <- function(n,mu,lambda){
  rexp <- rexp(n,lambda)
  rbin <- 2*rbinom(n,1,.5)-1
  x <- rexp*rbin+mu
}

niter = 1e4
exp_shortfall = rep(0,niter)
set.seed(2015)
for (m in 1:niter)
{
  r = rdexp(1508,mu,lambda)
  index = which(r < q)
  exp_shortfall[m] = mean(r[index])
}
mean(na.omit(exp_shortfall))*1e7

#ca
posloss = -logr[logr < 0]
n = length(posloss)
mean = mean(posloss)
std = sd(posloss)
lambda = 1 / mean
p_pos = (n - 0.005*length(logr))/n
q_pos = qexp(p_pos,rate = lambda)
q_pos
quantile(posloss,p_pos)

#cb
rdexp <- function(n,mu,lambda){
  rexp <- rexp(n,lambda)
  rbin <- 2*rbinom(n,1,.5)-1
  x <- rexp*rbin+mu
}

niter = 1e4
exp_shortfall = rep(0,niter)
set.seed(2015)
for (m in 1:niter)
{
  r = rexp(n,rate = lambda)
  index = which(r > q_pos)
  exp_shortfall[m] = mean(r[index])
}
mean(na.omit(exp_shortfall))*1e7