rm(list = ls())
library(mlbench)
library(kernlab)

set.seed(126)

data(Glass)
x <- scale(Glass[, -10])
y <- Glass$Type

C_val <- 237
sigma_val <- 0.01

n_repet <- 10
nfolds <- 10

# Vetores para armazenar resultados
acc_repet <- numeric(n_repet)
sd_repet <- numeric(n_repet)

# Loop das repetições
for (r in 1:n_repet) {
  set.seed(100 + r)  # Semente diferente a cada repetição
  folds <- sample(rep(1:nfolds, length.out = nrow(x)))
  
  accs_fold <- numeric(nfolds)
  
  for (k in 1:nfolds) {
    test_idx <- which(folds == k)
    train_idx <- setdiff(1:nrow(x), test_idx)
    
    modelo <- ksvm(x[train_idx, ], y[train_idx],
                   type = "C-svc",
                   kernel = "rbfdot",
                   C = C_val,
                   kpar = list(sigma = sigma_val),
                   scaled = FALSE)
    
    ypred <- predict(modelo, x[test_idx, ])
    accs_fold[k] <- mean(ypred == y[test_idx])
  }
  
  # Guarda média e desvio padrão da repetição
  acc_repet[r] <- mean(accs_fold)
  sd_repet[r] <- sd(accs_fold)
  
  cat(sprintf("Repetição %d - Acurácia média: %.4f - Desvio padrão: %.4f\n", 
              r, acc_repet[r], sd_repet[r]))
}

# Resultados finais
cat("\n============================\n")
cat("Acurácia média geral:", mean(acc_repet), "\n")
cat("Desvio padrão geral das médias:", sd(acc_repet), "\n")
