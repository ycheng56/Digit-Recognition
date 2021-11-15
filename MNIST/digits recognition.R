wd<-getwd()
path= wd
#10-classclassification, Digit 0-9
load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  x <<- load_image_file('train-images-idx3-ubyte')
  xTest <<- load_image_file('t10k-images-idx3-ubyte')
  
  y <<- load_label_file('train-labels-idx1-ubyte')
  yTest <<- load_label_file('t10k-labels-idx1-ubyte')  
}


show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

x <<- load_image_file('train-images-idx3-ubyte')
xTest <<- load_image_file('t10k-images-idx3-ubyte')
y <<- load_label_file('train-labels-idx1-ubyte')
yTest <<- load_label_file('t10k-labels-idx1-ubyte')
show_digit(x[2,])

library(softmaxreg)
## Normalize Input Data
x = x/255
xTest = xTest/255
model1= softmaxReg(x, y, hidden = c(), funName = 'sigmoid', maxit = 15, rang = 0.1,type = "class", algorithm = "sgd", rate = 0.01, batch = 1000)
loss1= model1$loss
#Test Accuracy
yFit= predict(model1, newdata = xTest)
yFit= yFit-1
table(yTest,yFit)
mean(yTest==yFit)

model2= softmaxReg(x, y, hidden = c(), funName = 'sigmoid', maxit = 15, rang = 0.1,type = "class", algorithm = "adagrad", rate = 0.01, batch =1000)
loss2= model2$loss
model3= softmaxReg(x, y, hidden = c(), funName = 'sigmoid', maxit = 15, rang = 0.1,type = "class", algorithm = "rmsprop", rate = 0.01, batch =1000)
loss3= model3$loss
model4= softmaxReg(x, y, hidden = c(), funName = 'sigmoid', maxit = 15, rang = 0.1,type = "class", algorithm = "momentum", rate = 0.01, batch= 1000)
loss4= model4$loss
model5= softmaxReg(x, y, hidden = c(), funName = 'sigmoid', maxit = 15, rang = 0.1,type = "class", algorithm = "nag", rate = 0.01, batch = 1000)
loss5= model5$loss 

yFit2= predict(model2, newdata = xTest)-1
yFit3= predict(model3, newdata = xTest)-1
yFit4= predict(model4, newdata = xTest)-1
yFit5= predict(model5, newdata = xTest)-1

model.nn=softmaxReg(x, y, hidden = c(500,300), funName = 'sigmoid', maxit = 10, rang = 0.1,type = "class", algorithm = "rmsprop", rate = 0.01, batch =1000)
loss.nn= model.nn$loss
yFit.nn= predict(model.nn, newdata = xTest)-1
mean(yTest==yFit.nn)

par(mfrow=c(1,1))
#plot the loss convergence
iteration= c(1:length(loss1))
myplot= plot(iteration, loss1, xlab = "iteration", ylab = "loss",ylim = c(0, max(loss1,loss2,loss3,loss4,loss5) + 0.01),
             type = "p", col ="black", cex = 0.7)
title("ConvergenceComparision Between Learning Algorithms")
points(iteration,loss2, col = "red", pch = 2, cex = 0.7)
points(iteration,loss3, col = "blue", pch = 3, cex = 0.7)
points(iteration,loss4, col = "green", pch = 4, cex = 0.7)
points(iteration,loss5, col = "magenta", pch = 5, cex = 0.7)
points(c(1:length(loss.nn)),loss.nn,col = "yellow", pch = 6, cex = 0.7)
legend("topright",c("SGD", "Adagrad", "RMSprop","Momentum", "NAG","Hidden layer"),
       col = c("black", "red","blue", "green", "magenta","yellow"),pch = c(1,2,3,4,5,6),cex=0.7)
save.image()


par(mfrow=c(2,3))
show_digit(xTest[1,],xlab=yFit3[1])
show_digit(xTest[2,],xlab=yFit3[2])
show_digit(xTest[3,],xlab=yFit3[3])
show_digit(xTest[4,],xlab=yFit3[4])
show_digit(xTest[5,],xlab=yFit3[5])
show_digit(xTest[9,],xlab=yFit3[9])

