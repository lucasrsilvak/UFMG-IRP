rm(list=ls())

library("mlbench")

data("BreastCancer")
data2 <- BreastCancer
data2 <- data2[complete.cases(data2),]

# Função Perceptron
perceptron <- function(x, w) {
  result <- sum(x * w)
  return(ifelse(result > 0, 1, 0))
}

# Função de Treinamento
treinaperceptron <- function(X, y, eta = 0.01, tol = 1e-4, maxepocs = 1000) {
  N <- nrow(X)
  n <- ncol(X)
  X_bias <- cbind(1, X)
  w <- rep(0, n + 1)
  
  previous_error <- Inf
  
  for (epoch in 1:maxepocs) {
    error_count <- 0
    total_error <- 0
    
    for (i in 1:N) {
      y_pred <- perceptron(X_bias[i,], w)
      if (y_pred != y[i]) {
        error_count <- error_count + 1
        w <- w + eta * (y[i] - y_pred) * X_bias[i,]
      }
      total_error <- total_error + (y[i] - y_pred)^2
    }
    
    if (abs(previous_error - total_error) < tol) {
      break
    }
    
    previous_error <- total_error
  }
  
  return(w)
}

X <- data2[,2:10]
X <- as.data.frame(lapply(X, function(col) as.numeric(as.character(col))))
Y <- ifelse(data2[,11] == "malignant", 1, 0)

set.seed(42)
folds <- sample(rep(1:10, length.out = nrow(X)))

acuracias <- c()

for (k in 1:10) {
  X_treino <- X[folds != k,]
  Y_treino <- Y[folds != k]
  X_teste  <- X[folds == k,]
  Y_teste  <- Y[folds == k]
  
  modelo <- treinaperceptron(X_treino, Y_treino, eta = 0.01, tol = 0.01, maxepocs = 300)
  
  X_teste_bias <- cbind(1, X_teste)
  
  Y_pred <- apply(X_teste_bias, 1, function(x) perceptron(x, modelo))
  
  acuracia <- mean(Y_pred == Y_teste)
  acuracias <- c(acuracias, acuracia)
  
  cat("Fold", k, "- Acurácia:", acuracia, "\n")
}

cat("Acurácia Média:", mean(acuracias), "\n")
cat("Desvio Padrão Médio:", sd(acuracias), "\n")
