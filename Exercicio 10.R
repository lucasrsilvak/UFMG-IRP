# Limpeza do ambiente
rm(list = ls())

# Pacotes necessários
require(RnavGraphImageData)
require(MASS)       # Para classificador Bayesiano
require(caret)      # Para particionar os dados

# 1 - Carregar a base de dados Olivetti
data(faces)
faces <- t(faces)  # Transpor para ter imagens nas linhas

# 2 - Função para visualizar as imagens
MostraImagem <- function(x) {
  rotate <- function(x) t(apply(x, 2, rev))
  img <- matrix(x, nrow = 64)
  cor <- rev(gray(50:1 / 50))
  image(rotate(img), col = cor)
}

# Visualizar exemplo
# MostraImagem(faces[1,])

# 3 - Gerar rótulos (40 classes, cada uma com 10 imagens)
y <- rep(1:40, each = 10)

# 4 - PCA manual

# Centralizar os dados
meanx <- colMeans(faces)
Xs <- scale(faces, center = meanx, scale = FALSE)  # centralização

# Matriz de covariância
S <- cov(Xs)

# Decomposição em autovalores/autovetores
eigS <- eigen(S)

# Visualização dos autovalores (scree plot)
plot(eigS$values, type = 'b', xlab = 'Eixo PCA', ylab = "Autovalor", main = "Autovalores")
plot(eigS$values[1:50], type = 'b', xlab = 'Eixo PCA', ylab = "Autovalor", col = 'red', main = "Top 50 Autovalores")

# 5 - Projeção dos dados nos componentes principais
projX <- Xs %*% eigS$vectors

# 6 - Determinar o número de componentes que explicam ~95% da variância
var_exp <- eigS$values / sum(eigS$values)
var_cum <- cumsum(var_exp)
plot(var_cum, type = 'b', xlab = 'Número de Componentes', ylab = 'Variância Explicada', main = 'Variância Cumulativa')

# Selecionar número de componentes para ~95% de variância explicada
num_comp <- which(var_cum >= 0.95)[1]
cat("Número de componentes selecionados:", num_comp, "\n")

# 7 - Subset dos dados projetados
projX_reduced <- projX[, 1:num_comp]

# 8 - Escolha da classe alvo (por exemplo, classe 1)
classe_alvo <- 1
y_bin <- ifelse(y == classe_alvo, classe_alvo, 0)  # binário: alvo x não-alvo

# 9 - Repetição do experimento 10 vezes
set.seed(123)  # Reprodutibilidade
resultados <- numeric(10)

for (i in 1:10) {
  
  # Divisão treino/teste mantendo proporção
  trainIndex <- createDataPartition(y_bin, p = 0.5, list = FALSE)
  
  X_train <- projX_reduced[trainIndex, ]
  X_test <- projX_reduced[-trainIndex, ]
  
  y_train <- y_bin[trainIndex]
  y_test <- y_bin[-trainIndex]
  
  # Classificador Bayesiano Gaussiano
  modelo <- lda(x = X_train, grouping = as.factor(y_train))
  
  pred <- predict(modelo, X_test)$class
  
  # Cálculo da acurácia
  acuracia <- sum(pred == as.factor(y_test)) / length(y_test)
  resultados[i] <- acuracia * 100  # percentual
}

# 10 - Resultados finais
cat("Resultados das 10 execuções (%):\n")
print(resultados)

cat("Média da acurácia (%):", mean(resultados), "\n")
cat("Desvio padrão da acurácia (%):", sd(resultados), "\n")
cat("Número de componentes utilizados:", num_comp, "\n")
