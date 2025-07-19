rm(list=ls())

# Função Perceptron
perceptron <- function(X_aug, Y) {
  max_iter <- 1000
  w <- rep(0, ncol(X_aug))  
  w_history <- list()
  
  for (epoch in 1:max_iter) {
    updated <- FALSE
    index = sample(nrow(X_aug))
    for (i in index) {
      if (Y[i] * (t(w) %*% X_aug[i,]) <= 0) {
        w <- w + Y[i] * X_aug[i,]
        w_history[[length(w_history)+1]] <- w
        updated <- TRUE
      }
    }
    if (!updated) break
  }
  return(w_history)
}

# Geração dos dados
n <- 2
N1 <- 50
sd1 <- 0.2
xc1 <- matrix(rnorm(N1*n, sd=sd1), nrow=N1, ncol=n) + matrix(c(2,2), nrow=N1, ncol=n)
N2 <- 50
sd2 <- 0.2
xc2 <- matrix(rnorm(N2*n, sd=sd2), nrow=N2, ncol=n) + matrix(c(4,4), nrow=N2, ncol=n)

# Preparo dos dados
X <- rbind(xc1, xc2)
X_aug <- cbind(1, X)
Y <- rbind(matrix(-1, nrow=N1), matrix(1, nrow=N2))

# Plot dos dados
plot(xc1[,1], xc1[,2], col='blue', xlim=c(0,6), ylim=c(0,6), xlab='', ylab='',)
points(xc2[,1], xc2[,2], col='red')

# Cálculo da margem geométrica
calc_margin <- function(w, X_aug, Y) {
  margins <- Y * (X_aug %*% w)
  w_sem_bias <- w[-1]
  norm_w <- sqrt(sum(w_sem_bias^2))
  return(min(margins) / norm_w)
}

# Busca pelo melhor w com maior margem
best_margin <- -Inf
w_melhor <- NULL
ws_todos <- list()

for (k in 1:100) {
  ws <- perceptron(X_aug, Y)
  ws_todos <- c(ws_todos, ws)
  for (w in ws) {
    margin <- calc_margin(w, X_aug, Y)
    if (!is.nan(margin) && margin > best_margin) {
      best_margin <- margin
      w_melhor <- w
    }
  }
}

# Plot de todas as fronteiras ruins (cinza claro)
for (w in ws_todos) {
  abline(a=-w[1]/w[3], b=-w[2]/w[3], col=rgb(0.7, 0.7, 0.7, 0.05), lwd=1)
}

w_melhor = c(-6,1,1)


# Plot da melhor fronteira (dourado)
abline(a=-w_melhor[1]/w_melhor[3], b=-w_melhor[2]/w_melhor[3], col='gold', lwd=3)


cat("Maior margem encontrada:", best_margin, "\n")