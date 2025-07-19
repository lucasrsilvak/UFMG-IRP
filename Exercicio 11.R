# Limpeza do ambiente
rm(list = ls())

# Pacotes necessários
library(dslabs)    # para carregar o MNIST
library(ggplot2)   # para gráficos
library(caret)     # para divisão e avaliação
library(MASS)      # para LDA
library(e1071)     # para SVM

# 1 - Carregar a base de dados MNIST
mnist <- read_mnist()

x <- mnist$train$images
y <- mnist$train$labels

# Para acelerar o processamento, vamos usar um subconjunto (ex.: 5000 amostras)
set.seed(123)
idx <- sample(1:nrow(x), 5000)

x <- x[idx, ]
y <- y[idx]

# 2 - Visualizar algumas imagens
MostraImagem <- function(vetor) {
  img <- matrix(vetor, nrow = 28, byrow = TRUE)
  image(1:28, 1:28, img[,28:1], col = gray.colors(255), axes = FALSE)
}

par(mfrow = c(2,5))
for (i in 1:10) {
  MostraImagem(x[i,])
}

# 3 - PCA com diferentes números de componentes
meanx <- colMeans(x)
Xs <- scale(x, center = meanx, scale = FALSE)

pca <- prcomp(Xs, center = FALSE, scale. = FALSE)

var_exp <- pca$sdev^2 / sum(pca$sdev^2)
var_cum <- cumsum(var_exp)

# Scree plot
plot(var_cum, type = 'b', xlab = 'Número de Componentes', ylab = 'Variância acumulada', main = 'Variância explicada')
abline(h = 0.95, col = 'red')

# Variância explicada por alguns números de componentes
for (n in c(10, 30, 50, 100)) {
  cat(sprintf("Componentes: %d - Variância Explicada: %.2f%%\n", n, 100 * var_cum[n]))
}

# 4 - Determinar número de componentes para ~95% de variância
num_comp <- which(var_cum >= 0.95)[1]
cat("Número de componentes para ~95% de variância:", num_comp, "\n")

# 5 - Treinar classificador SVM com dados reduzidos
treina_avalia <- function(n_comp) {
  X_proj <- Xs %*% pca$rotation[, 1:n_comp]
  
  set.seed(123)
  trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
  
  X_train <- X_proj[trainIndex, ]
  X_test <- X_proj[-trainIndex, ]
  
  y_train <- y[trainIndex]
  y_test <- y[-trainIndex]
  
  modelo <- svm(x = X_train, y = as.factor(y_train))
  pred <- predict(modelo, X_test)
  
  acc <- sum(pred == y_test) / length(y_test)
  return(acc)
}

# Avaliação com diferentes números de componentes
for (n in c(10, 30, 50, 100, ncol(x))) {
  acc <- treina_avalia(n)
  cat(sprintf("Componentes: %d - Acurácia: %.2f%%\n", n, 100 * acc))
}

# 6 - Visualização 2D com as duas primeiras componentes
X_proj2d <- Xs %*% pca$rotation[, 1:2]
df_proj <- data.frame(PC1 = X_proj2d[,1], PC2 = X_proj2d[,2], Classe = as.factor(y))

ggplot(df_proj, aes(x = PC1, y = PC2, color = Classe)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "MNIST - Projeção PCA 2D")

# Fim
