rm(list=ls())
library(mlbench)
library(class)
library(caret)

set.seed(123)

pdfnvar <- function(x, m, K, n) {
  (1 / sqrt((2*pi)^n * det(K))) * exp(-0.5 * t(x - m) %*% solve(K) %*% (x - m))
}

mymix <- function(x, inlist) {
  ng <- length(inlist)
  klist <- list()
  mlist <- list()
  pglist <- list()
  nglist <- list()
  n <- ncol(inlist[[1]])
  
  for (i in 1:ng) {
    klist[[i]] <- cov(inlist[[i]])
    mlist[[i]] <- colMeans(inlist[[i]])
    nglist[[i]] <- nrow(inlist[[i]])
  }
  
  N <- sum(unlist(nglist))
  for (i in 1:ng) {
    pglist[[i]] <- nglist[[i]] / N
  }
  
  Px <- 0
  for (i in 1:ng) {
    Px <- Px + pglist[[i]] * pdfnvar(x, mlist[[i]], klist[[i]], n)
  }
  
  return(Px)
}

# Passo 1: Dados
espiral <- mlbench.spirals(400, cycles=1, sd=0.05)
x <- espiral$x
y <- as.numeric(espiral$classes)

# Passo 2: Validação cruzada
folds <- createFolds(y, k=10)
accuracies <- numeric(10)

ks <- 5

for (f in 1:10) {
  test_idx <- folds[[f]]
  x_train <- x[-test_idx,]
  y_train <- y[-test_idx]
  x_test <- x[test_idx,]
  y_test <- y[test_idx]
  
  # Passo 3: Misturas de Gaussianas por K-means
  clusters_class1 <- kmeans(x_train[y_train == 1,], centers=ks)
  clusters_class2 <- kmeans(x_train[y_train == 2,], centers=ks)
  
  list_class1 <- list()
  list_class2 <- list()
  
  for (k in 1:ks) {
    list_class1[[k]] <- x_train[y_train == 1,][clusters_class1$cluster == k,]
    list_class2[[k]] <- x_train[y_train == 2,][clusters_class2$cluster == k,]
  }
  
  # Plot intermediário das Gaussianas (cores por cluster e triângulos nos centróides)
  colors1 <- rainbow(ks)
  colors2 <- rainbow(ks)
  
  plot(x_train, type='n', xlim=c(-1.5,1.5), ylim=c(-1.5,1.5),
       xlab="x", ylab="y", main=paste("Clusters Gaussianos - Fold", f))
  
  for (k in 1:ks) {
    points(list_class1[[k]], col=colors1[k], pch=19)
    mu1 <- colMeans(list_class1[[k]])
    points(mu1[1], mu1[2], pch=17, cex=1, col='black')
    
    points(list_class2[[k]], col=colors2[k], pch=19)
    mu2 <- colMeans(list_class2[[k]])
    points(mu2[1], mu2[2], pch=17, cex=1, col='black')
  }
  
  # Passo 4: Classificação Bayesiana
  y_pred <- numeric(length(y_test))
  for (i in 1:length(y_test)) {
    xt <- matrix(x_test[i,], ncol=1)
    p1 <- mymix(xt, list_class1)
    p2 <- mymix(xt, list_class2)
    y_pred[i] <- ifelse(p1 > p2, 1, 2)
  }
  
  # Passo 5: Acurácia do fold
  acc <- mean(y_pred == y_test) * 100
  accuracies[f] <- acc
  cat("Fold", f, "- Acurácia:", round(acc, 2), "%\n")
}

# Passo 6: Resultados finais
cat("\nAcurácia média:", round(mean(accuracies), 2), "%\n")
cat("Desvio padrão:", round(sd(accuracies), 2), "%\n")

# Passo 7: Plot da melhor separação
best_fold <- which.max(accuracies)
test_idx <- folds[[best_fold]]
x_train <- x[-test_idx,]
y_train <- y[-test_idx]

# Repete mistura para o melhor fold
clusters_class1 <- kmeans(x_train[y_train == 1,], centers=ks)
clusters_class2 <- kmeans(x_train[y_train == 2,], centers=ks)

list_class1 <- list()
list_class2 <- list()
for (k in 1:ks) {
  list_class1[[k]] <- x_train[y_train == 1,][clusters_class1$cluster == k,]
  list_class2[[k]] <- x_train[y_train == 2,][clusters_class2$cluster == k,]
}

seqx <- seq(-1.5, 1.5, length=100)
seqy <- seq(-1.5, 1.5, length=100)
grid <- expand.grid(x=seqx, y=seqy)

z <- apply(grid, 1, function(pt) {
  xt <- matrix(pt, ncol=1)
  p1 <- mymix(xt, list_class1)
  p2 <- mymix(xt, list_class2)
  return(ifelse(p1 > p2, 1, 2))
})

z_matrix <- matrix(z, nrow=100, ncol=100)

contour(seqx, seqy, z_matrix, levels=c(1.5), drawlabels=FALSE, col='black', lwd=2)
points(x_train, col=ifelse(y_train==1,'blue','red'), pch=19)
