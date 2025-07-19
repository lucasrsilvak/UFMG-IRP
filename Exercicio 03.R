rm(list = ls())
library('plot3D')

# x = variaveis do problema
# w = constantes que vc quer ajustar para otimizar o problema

perceptron <- function(x, w) {
  result <- sum(x * w) # 1*w0 + w1x1 + w2x2 
  return(ifelse(result > 0, 1, -1)) # se w0 + w1x1 + w2x2 > 0 retorna 1, senão retorna -1
}

treinaperceptron <- function(X, y, eta = 0.01, tol = 5, maxepocs = 10000) {
  N <- nrow(X) # (n de amostras)
  n <- ncol(X) # x1 = x,x2 = y (2 variaveis)
  X_bias <- cbind(1, X) # x (3 variaveis) = [1, x1, x2]
  w <- rep(0, n + 1) # w [w0, w1, w2]
  
  # 1*x0 + w1x1 + w2x2
  
  # xw = w0 + w1x1 + w2x2
  
  previous_error <- Inf
  
  for (epoch in 1:maxepocs) { # while ()
    error_count <- 0
    total_error <- 0
    
    for (i in 1:N) { # N = numero de amostras
      y_pred <- perceptron(X_bias[i,], w) # retorna o valor -1 ou 1 do perceptron
      if (y_pred != y[i]) { # se errou
        error_count <- error_count + 1 # errei
        w <- w + eta * (y[i] - y_pred) * X_bias[i,] # ajusta o w 
        # eta = passo de treinamento (se for muito rápido oscila)
      }
      total_error <- total_error + (y[i] - y_pred)^2 #erro quadratico
    }
    
    if (abs(previous_error) < tol) {
      print(paste("Convergiu na época", epoch))
      break
    }
    
    previous_error <- total_error
    cat("Época:", epoch, "Erros:", error_count, "\n")
  }
  
  return(w)
}

gera_gaussianas_2classes_2D <- function(N1, M1, SD1, N2, M2, SD2, seed) {
  set.seed(seed)
  
  xc1 <- matrix(rnorm(N1 * 2), ncol = 2) * SD1 + t(matrix(M1, ncol = N1, nrow = 2))
  xc2 <- matrix(rnorm(N2 * 2), ncol = 2) * SD2 + t(matrix(M2, ncol = N2, nrow = 2))
  
  y1 <- rep(1, N1)
  y2 <- rep(-1, N2)
  
  X <- rbind(xc1, xc2)
  Y <- c(y1, y2)
  
  return(list(X, Y))  # Retorna os dados e as classes
}

plota_sup_2D <- function(X, Y, lab1, lab2, M, seqi, seqj) {
  plot(X[which(Y > 0), 1], X[which(Y > 0), 2], col = 'red', pch = 'o', 
       xlab = lab1, ylab = lab2, xlim = c(0, max(X[, 1])), ylim = c(0, max(X[, 2])))
  par(new = TRUE)
  plot(X[which(Y < 0), 1], X[which(Y < 0), 2], col = 'blue', pch = '+', 
       xlab = "", ylab = "", xlim = c(0, max(X[, 1])), ylim = c(0, max(X[, 2])))
  par(new = TRUE)
  
  contour2D(M, seqi, seqj, xlim = c(0, max(X[, 1])), ylim = c(0, max(X[, 2])), 
            xlab = '', ylab = '', levels = 0)
}

seqi <- seq(0.06, 6, 0.06)
seqj <- seq(0.06, 6, 0.06)
G <- matrix(0, nrow = length(seqi), ncol = length(seqj))

M <- gera_gaussianas_2classes_2D(100, c(2, 2), 0.9, 100, c(4, 4), 0.9, seed = 42)
X <- M[[1]]
Y <- M[[2]]

weights <- treinaperceptron(X, Y)

ci <- 0
for (i in seqi) {
  ci <- ci + 1
  cj <- 0
  for (j in seqj) {
    cj <- cj + 1
    G[ci, cj] <- perceptron(c(1, i, j), weights)
  }
}

plota_sup_2D(X, Y, "X", "Y", G, seqi, seqj)