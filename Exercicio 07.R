rm(list=ls())
library('plot3D')
library('mlbench')

fnormal1var <- function(x, m, r) {
  (1 / sqrt(2 * pi * r^2)) * exp(-0.5 * ((x - m)^2 / r^2))
}

set.seed(124)  # Para reprodutibilidade

N1 <- 100
N2 <- 100
u1 <- 2
u2 <- 4
sd_1 <- 0.7
sd_2 <- 0.7
# Geração completa
xc1_all <- cbind(rnorm(N1, mean = u1, sd = sd_1), rep(0, N1))
xc2_all <- cbind(rnorm(N2, mean = u2, sd = sd_2), rep(1, N2))

# Separação treino/teste (90% treino)
train_idx1 <- sample(1:N1, size = 0.9 * N1)
train_idx2 <- sample(1:N2, size = 0.9 * N2)

xc1 <- xc1_all[train_idx1, , drop=FALSE]
xc2 <- xc2_all[train_idx2, , drop=FALSE]
test1 <- xc1_all[-train_idx1, , drop=FALSE]
test2 <- xc2_all[-train_idx2, , drop=FALSE]

# Plots de treino
plot(xc1, col='blue', xlim=c(0,6), ylim=c(0,1), xlab='', ylab='')
par(new=T)
plot(xc2, col='red', xlim=c(0,6), ylim=c(0,1), xlab='', ylab='')

# KDE
x_min <- min(c(xc1[,1], xc2[,1])) - 1
x_max <- max(c(xc1[,1], xc2[,1])) + 1
xgrid <- seq(x_min, x_max, length.out = 500)

h <- 0.1

pkdetotal1 <- rep(0, length(xgrid))
for(i in 1:nrow(xc1)) {
  pkde <- fnormal1var(xgrid, xc1[i], h) / nrow(xc1)
  par(new=T)
  #plot(xgrid, pkde, type='l', col='blue', xlab='X', ylab='', xlim=c(0,6), ylim=c(0,1))
  pkdetotal1 <- pkdetotal1 + pkde
}
par(new=T)
plot(xgrid, pkdetotal1, type='l', col='blue', xlab='X', ylab='', xlim=c(0,6), ylim=c(0,1))

pkdetotal2 <- rep(0, length(xgrid))
for(i in 1:nrow(xc2)) {
  pkde <- fnormal1var(xgrid, xc2[i], h) / nrow(xc2)
  par(new=T)
  #plot(xgrid, pkde, type='l', col='red', xlab='X', ylab='', xlim=c(0,6), ylim=c(0,1))
  pkdetotal2 <- pkdetotal2 + pkde
}
par(new=T)
plot(xgrid, pkdetotal2, type='l', col='red', xlab='X', ylab='', xlim=c(0,6), ylim=c(0,1))

par(new=T)

pc1 <- nrow(xc1) / (nrow(xc1) + nrow(xc2))
pc2 <- nrow(xc2) / (nrow(xc1) + nrow(xc2))

fx1 <- pkdetotal1
fx2 <- pkdetotal2

px <- fx1 * pc1 + fx2 * pc2
pc1x <- (fx1*pc1) / px
pc2x <- (fx2*pc2) / px

yhat <- sign(pc1x - pc2x)
par(new = T)
plot(xgrid, yhat, col='black', xlim = c(0, 6), ylim = c(0, 1), type= 'l', xlab='', ylab='')

# === Classificação dos dados de teste ===

# Funções de densidade
dens1 <- function(x) mean(sapply(xc1[,1], function(xi) fnormal1var(x, xi, h)))
dens2 <- function(x) mean(sapply(xc2[,1], function(xi) fnormal1var(x, xi, h)))

# Dados de teste
test_data <- rbind(test1, test2)
true_labels <- c(rep(0, nrow(test1)), rep(1, nrow(test2)))

pred_labels <- sapply(test_data[,1], function(x) {
  p1 <- dens1(x) * pc1
  p2 <- dens2(x) * pc2
  if (p1 > p2) 0 else 1
})

# Cálculo da acurácia
accuracy <- mean(pred_labels == true_labels)
cat("Acurácia no conjunto de teste:", round(accuracy * 100, 2), "%\n")
