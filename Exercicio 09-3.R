rm(list = ls())

set.seed(126)

library(mlbench)
library(kernlab)
library(ggplot2)
library(reshape2)

# Carrega dados e normaliza
data(Glass)
x <- scale(Glass[, -10])
y <- Glass$Type

# Hiperparâmetros
C_vals <- 10^seq(-2, 3, length.out = 20)      # C: 0.01 a 100
deg_vals <- 1:6                              # grau do polinômio
scale_val <- 1                               # fator de escala fixo

# Matriz para armazenar acurácias médias
acc_matrix <- matrix(0, nrow = length(C_vals), ncol = length(deg_vals),
                     dimnames = list(paste0("C=", C_vals),
                                     paste0("deg=", deg_vals)))

# Validação cruzada 10-fold
set.seed(42)
folds <- sample(rep(1:10, length.out = nrow(x)))

# Loop para testar combinações
for (i in seq_along(C_vals)) {
  for (j in seq_along(deg_vals)) {
    accs <- numeric(10)
    for (k in 1:10) {
      test_idx <- which(folds == k)
      train_idx <- setdiff(1:nrow(x), test_idx)
      
      model <- ksvm(x[train_idx, ], y[train_idx],
                    type = "C-svc",
                    kernel = "polydot",
                    C = C_vals[i],
                    kpar = list(degree = deg_vals[j], scale = scale_val),
                    scaled = FALSE)
      
      pred <- predict(model, x[test_idx, ])
      accs[k] <- mean(pred == y[test_idx])
    }
    acc_matrix[i, j] <- mean(accs)
  }
}

# Converte para data frame longo para plotagem
acc_df <- melt(acc_matrix)
colnames(acc_df) <- c("C", "Degree", "Acuracia")

# Plot heatmap
ggplot(acc_df, aes(x = Degree, y = C, fill = Acuracia)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Acuracia, 2)), size = 3) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  theme_minimal() +
  labs(title = "Acurácia Média (SVM com Kernel Polinomial)",
       x = "Grau do polinômio",
       y = "C",
       fill = "Acurácia")
