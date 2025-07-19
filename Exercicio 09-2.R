rm(list = ls())

set.seed(126)

# Bibliotecas
library(mlbench)
library(kernlab)
library(ggplot2)
library(reshape2)

# Carrega e normaliza dados
data(Glass)
x <- scale(Glass[, -10])
y <- Glass$Type

# Geração de hiperparâmetros
C_vals <- 10^seq(-2, 3, length.out = 25)        # 0.01 a 100
sigma_vals <- 10^seq(-3, 0, length.out = 25)    # 0.001 a 1

# Inicializa matriz de acurácias
acc_matrix <- matrix(0, nrow = length(C_vals), ncol = length(sigma_vals),
                     dimnames = list(paste0("C=", round(C_vals, 4)),
                                     paste0("σ=", round(sigma_vals, 4))))

# Validação cruzada 10-fold
folds <- sample(rep(1:10, length.out = nrow(x)))

# Loop para preencher matriz
for (i in seq_along(C_vals)) {
  for (j in seq_along(sigma_vals)) {
    accs <- numeric(10)
    for (k in 1:10) {
      test_idx <- which(folds == k)
      train_idx <- setdiff(1:nrow(x), test_idx)
      
      model <- ksvm(x[train_idx, ], y[train_idx],
                    type = "C-svc",
                    kernel = "rbfdot",
                    C = C_vals[i],
                    kpar = list(sigma = sigma_vals[j]),
                    scaled = FALSE)
      
      pred <- predict(model, x[test_idx, ])
      accs[k] <- mean(pred == y[test_idx])
    }
    acc_matrix[i, j] <- mean(accs)
  }
}

# Transforma matriz para data frame longo (long format)
acc_df <- melt(acc_matrix)
colnames(acc_df) <- c("C", "Sigma", "Acuracia")

# Plota heatmap com ggplot
ggplot(acc_df, aes(x = Sigma, y = C, fill = Acuracia)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Acuracia, 2)), size = 3) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  labs(title = "Acurácia Média por C e Sigma (SVM RBF)",
       x = "Sigma",
       y = "C",
       fill = "Acurácia")
