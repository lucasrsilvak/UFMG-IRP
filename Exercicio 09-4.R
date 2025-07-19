rm(list = ls())
library(e1071)
library(mlbench)
library(RSNNS)

# Carregar dados
data(Glass)
x <- scale(Glass[, -10])
y_original <- Glass$Type
y <- ifelse(y_original == 1, 1, -1)

# Dividir dados
set.seed(126)
xysplit <- splitForTrainingAndTest(x, y, 0.3)
x_train <- xysplit$inputsTrain
y_train <- xysplit$targetsTrain
x_test <- xysplit$inputsTest
y_test <- xysplit$targetsTest

# Treinar SVM (kernel radial)
svm_model <- svm(
  x = x_train,
  y = as.factor(y_train),
  kernel = "radial",
  gamma = 0.1,
  cost = 237,
  scale = FALSE
)

# Extrair vetores de suporte
sv_indices <- svm_model$index
sv <- x_train[sv_indices, , drop = FALSE]

# Função do kernel RBF
rbf_kernel <- function(x, y, gamma) {
  dist_sq <- sum((x - y)^2)
  exp(-gamma * dist_sq)
}

# Escolher dois vetores de suporte para projeção
sv1 <- sv[1, ]
sv2 <- sv[2, ]

# Calcular projeções no espaço das similaridades
proj <- matrix(NA, nrow = nrow(x_train), ncol = 2)
for (i in 1:nrow(x_train)) {
  proj[i, 1] <- rbf_kernel(x_train[i, ], sv1, gamma = 0.1)
  proj[i, 2] <- rbf_kernel(x_train[i, ], sv2, gamma = 0.1)
}

# Cores para as classes
colvec <- ifelse(y_train == 1, "blue", "red")

# Plot básico
plot(proj[, 1], proj[, 2],
     col = colvec,
     pch = ifelse(1:nrow(x_train) %in% sv_indices, 17, 16),
     xlab = "Similaridade com SV1",
     ylab = "Similaridade com SV2",
     main = "Espaço das Similaridades da SVM")

legend("topright", legend = c("Classe 1", "Classe 2", "Vetores de Suporte"),
       col = c("blue", "red", "black"),
       pch = c(16, 16, 17))
