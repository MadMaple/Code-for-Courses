x = seq(-3,3,by = 0.01)
bias.1 = (pnorm(x+1.732*0.1) - pnorm(x-1.732*0.1))/(3.464*0.1) - dnorm(x)
bias.2 = (pnorm(x+1.732*0.2) - pnorm(x-1.732*0.2))/(3.464*0.2) - dnorm(x)
bias.4 = (pnorm(x+1.732*0.4) - pnorm(x-1.732*0.4))/(3.464*0.4) - dnorm(x)
plot(x,bias.1,type = "l",ylim = c(-0.03,0.015),lwd = 2,ylab = "Bias")
lines(x,bias.2,lty = 2,lwd = 2)
lines(x,bias.4,lty = 3,lwd = 2)
legend("bottomright", c("b = 0.1","b = 0.2","b = 0.4"),lty = c(1,2,3),lwd = 2)
abline(h = 0)