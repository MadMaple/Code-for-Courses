dat = read.csv("O:\\18WIN\\STATS 509\\HW1\\Nasdaq_daily_Jan1_2012_Dec31_2017.csv",header = T)
d = as.Date(dat$Date,format="%m/%d/%y")
logr = diff(log(dat$Adj.Close))
plot(d,dat$Adj.Close,type = "l",xlab="Time",ylab="Adjusted_Closing_Price")
plot(d[-1],100*logr,type="l",xlab="Time",ylab="Log_Return (%)")

library(fBasics)
summary(logr)
skewness(logr)
kurtosis(logr)
par(mfrow=c(1,2))
hist(logr)
boxplot(logr)