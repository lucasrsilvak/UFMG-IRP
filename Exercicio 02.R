# Clear environment
rm(list = ls())
library('plot3D')

myk <- 3
myh <- 0.1

# Funções auxiliares
pdfnvar <- function(x, m, K, n) {
  ((1 / (sqrt((2 * pi))^n * (det(K))))) * exp(-0.5 * (t(x - m) %*% (solve(K))) %*% (x - m))
}

myknnR <- function(X, Y, xt, k) {
  N <- dim(X)[1]
  n <- dim(X)[2]
  
  seqi <- seq(1, N, 1)
  mdist <- matrix(nrow = N, ncol = 1)
  
  for (i in seqi) {
    xc <- X[i, ]
    mdist[i] <- sqrt(sum((xc - t(xt))^2))
  }
  
  ordmdist <- order(mdist)
  ordY <- Y[ordmdist]
  
  yxt <- sign(sum(ordY[1:k]))
  retlist <- list(yxt, mdist, ordY)
  return(retlist)
}

myknnR2 <- function(X, Y, xt, k, h) {
  N <- dim(X)[1]
  n <- dim(X)[2]
  
  xtmat <- matrix(xt, nrow = N, ncol = n, byrow = TRUE)
  seqk <- order(rowSums((X - xtmat) * (X - xtmat)))
  K <- h * diag(n)
  
  xt <- matrix(xt, nrow = n, ncol = 1)
  sumk <- 0
  for (i in 1:k)
    sumk <- sumk + Y[seqk[i]] * pdfnvar(X[seqk[i], ], xt, K, n)
  yhat <- sign(sumk)
  
  return(yhat)
}

gera_gaussianas_2classes_2D <- function(N1, M1, SD1, N2, M2, SD2, seed) {
  set.seed(seed)
  
  xc1 <- matrix(rnorm(N1 * 2), ncol = 2) * SD1 + t(matrix(M1, ncol = N1, nrow = 2))
  xc2 <- matrix(rnorm(N2 * 2), ncol = 2) * SD2 + t(matrix(M2, ncol = N2, nrow = 2))
  
  y1 <- array(1, c(N1, 1))
  y2 <- array(-1, c(N2, 1))
  
  X <- rbind(xc1, xc2)
  Y <- rbind(y1, y2)
  
  retlist <- list(X, Y)
  return(retlist)
}

plota_sup_2D <- function(X, Y, lab1, lab2, M, seqi, seqj, color) {
  contour2D(M, seqi, seqj, add = TRUE, col = color, levels = 0, lwd = 2)
}

seqi <- seq(0.06, 6, 0.06)
seqj <- seq(0.06, 6, 0.06)

M <- gera_gaussianas_2classes_2D(100, c(2, 2), 1, 100, c(4, 4), 1, seed = 42)
X <- M[[1]]
Y <- M[[2]]

# Inicializa o gráfico vazio
plot(X[which(Y > 0), 1], X[which(Y > 0), 2], col = 'red', pch = 'o', 
     xlab = "X", ylab = "Y", xlim = c(0, max(X[, 1])), ylim = c(0, max(X[, 2])))

# Adiciona os pontos da classe Y < 0 (azul, "+")
points(X[which(Y < 0), 1], X[which(Y < 0), 2], col = 'blue', pch = '+')

# Geração da superfície de decisão
G <- matrix(0, nrow = length(seqi), ncol = length(seqj))
ci <- 0
for (i in seqi) {
  ci <- ci + 1
  cj <- 0
  for (j in seqj) {
    cj <- cj + 1
    G[ci, cj] <- myknnR2(X, Y, matrix(c(i, j), ncol = 2), k = myk, h = myh)[[1]]
  }
}

# Plotando a superfície de decisão
plota_sup_2D(X, Y, "X", "Y", G, seqi, seqj, "black")

# Projeção para X e Y
projetaX <- function(X, Y, xt, k, h) {
  N <- dim(X)[1]
  n <- dim(X)[2]
  
  xtmat <- matrix(xt, nrow = N, ncol = n, byrow = TRUE)
  seqk <- order(rowSums((X - xtmat) * (X - xtmat)))
  K <- h * diag(n)
  
  xt <- matrix(xt, nrow = n, ncol = 1)
  sumk1 <- 0
  sumk2 <- 0
  for (i in 1:k) {
    if (Y[seqk[i]] == -1)
      sumk1 <- sumk1 + pdfnvar(X[seqk[i], ], xt, K, n) # X - Afinidade com a Classe X
    else 
      sumk2 <- sumk2 + pdfnvar(X[seqk[i], ], xt, K, n) # Y - Afinidade com a Classe Y
  }
  return(list(sumk1, sumk2))
}

# Definindo p1 e p2 para armazenar os resultados da projeção
p1 <- numeric(200)  # Vetor de 200 elementos para armazenar os valores de sumk1
p2 <- numeric(200)  # Vetor de 200 elementos para armazenar os valores de sumk2

# Calculando os valores de p1 e p2 para cada ponto
for (i in 1:200) {
  retx <- projetaX(X, Y, as.matrix(X[i,]), myk, myh)
  p1[i] <- retx[[1]]
  p2[i] <- retx[[2]]
}

xlim_vals <- range(p1)
ylim_vals <- range(p2)

plot(c(0, 5), c(0, 5), type = "n", xlim = xlim_vals, ylim = ylim_vals, xlab = "X", ylab = "Y")

points(p1[Y == -1], p2[Y == -1], col = "red", pch = "o", cex = 1)  

points(p1[Y == 1], p2[Y == 1], col = "blue", pch = "+", cex = 1)   

abline(a = 0, b = 1, col = "black", lwd = 2)  # Linha y = x
