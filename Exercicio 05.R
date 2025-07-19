rm(list = ls())
library('plot3D')
library('mlbench')
library("rgl")
library("RSNNS")

# Carregando dados
data(PimaIndiansDiabetes)
dados <- na.omit(PimaIndiansDiabetes)

# Inputs e targets
xall <- as.matrix(dados[, 1:8])
yall <- ifelse(dados$diabetes == "pos", 1, -1)

# Embaralhar os dados
shuffle_idx <- sample(1:nrow(xall))
xall <- xall[shuffle_idx, ]
yall <- yall[shuffle_idx]

# Separação treino/teste
xysplit <- splitForTrainingAndTest(xall, yall, 0.3)
xall <- xysplit$inputsTrain
yall <- xysplit$targetsTrain
xtst <- xysplit$inputsTest
ytst <- xysplit$targetsTest

# Separar por classe
idxc1 <- which(yall == 1)
idxc2 <- which(yall == -1)
xc1_tr <- xall[idxc1, ]
xc2_tr <- xall[idxc2, ]

idxct1 <- which(ytst == 1)
idxct2 <- which(ytst == -1)
xc1_tst <- xtst[idxct1, ]
xc2_tst <- xtst[idxct2, ]

# Probabilidades a priori
Nall <- nrow(xall)
Nall_tst <- nrow(xtst)

pc1 <- nrow(xc1_tr) / Nall
pc2 <- nrow(xc2_tr) / Nall

# KDE multivariado
kdemulti <- function(xi, xall, h) {
  N <- nrow(xall)
  n <- ncol(xall)
  
  xirow <- matrix(xi, ncol = n, nrow = 1)
  xirep <- matrix(xirow, ncol = n, nrow = N, byrow = TRUE)
  matdif <- (xall - xirep)^2
  dximat <- rowSums(matdif) / (h^2)
  emat <- exp(-dximat / 2)
  pxi <- sum(emat) / (N * (sqrt(2 * pi) * h)^n)
  
  return(pxi)
}

# Teste de diferentes valores de h
seqh <- seq(0.01, 10, 0.05)
yhat_tst <- numeric(Nall_tst)
acc_tst <- numeric(length(seqh))
ch <- 0

for (h in seqh) {
  ch <- ch + 1
  for (i in 1:Nall_tst) {
    pxc1_tst <- kdemulti(xtst[i, ], xc1_tr, h)
    pxc2_tst <- kdemulti(xtst[i, ], xc2_tr, h)
    
    # Tratamento para divisão por zero
    if (pxc2_tst == 0) {
      yhat_tst[i] <- 1
    } else {
      yhat_tst[i] <- 1 * ((pxc1_tst / pxc2_tst) > (pc2 / pc1))
    }
  }
  yhat_bin <- (yhat_tst - 0.5) * 2
  acc_tst[ch] <- mean(yhat_bin == ytst)
}

# Melhor largura de banda
best_h_index <- which.max(acc_tst)
best_h <- seqh[best_h_index]

# Plot da acurácia
plot(seqh, acc_tst, type = 'l', col = 'blue',
     xlab = 'Largura do Kernel (h)', ylab = 'Acurácia',
     main = 'Acurácia em função de h', ylim = c(0, 1))

# Projeção 2D usando apenas as duas primeiras features
seqx1x2 <- seq(0, 6, 0.1)
lseq <- length(seqx1x2)
MZ2 <- matrix(nrow = lseq, ncol = lseq)

# Reduz treino para 2 dimensões
xc1_2d <- xc1_tr[, 1:2]
xc2_2d <- xc2_tr[, 1:2]

for (i in 1:lseq) {
  for (j in 1:lseq) {
    x_point <- c(seqx1x2[i], seqx1x2[j])
    pxc1 <- kdemulti(x_point, xc1_2d, best_h)
    pxc2 <- kdemulti(x_point, xc2_2d, best_h)
    MZ2[i, j] <- 1 * (pxc1 > (pc2 / pc1) * pxc2)
  }
}

# Contorno da região de decisão
contour(seqx1x2, seqx1x2, MZ2,
        xlim = c(0, 6), ylim = c(0, 6),
        xlab = "x1", ylab = "x2", nlevels = 1)
par(new = TRUE)
plot(xc1_tst[, 1], xc1_tst[, 2], col = 'red', pch = 1,
     xlim = c(0, 6), ylim = c(0, 6), xlab = "", ylab = "")
par(new = TRUE)
plot(xc2_tst[, 1], xc2_tst[, 2], col = 'blue', pch = 2,
     xlim = c(0, 6), ylim = c(0, 6), xlab = "", ylab = "")

# Espaço de verossimilhança pxc1 vs pxc2 para h ótimo
pxc1vec <- numeric(Nall_tst)
pxc2vec <- numeric(Nall_tst)

for (i in 1:Nall_tst) {
  pxc1vec[i] <- kdemulti(xtst[i, ], xc1_tr, best_h)
  pxc2vec[i] <- kdemulti(xtst[i, ], xc2_tr, best_h)
}

pxc1c2 <- cbind(pxc1vec, pxc2vec)
colvec <- c('red', 'blue')
plot(pxc1c2[, 1], pxc1c2[, 2],
     col = colvec[1 + (ytst + 1) / 2],
     xlab = "pxc1", ylab = "pxc2",
     main = paste("Verossimilhanças para h =", round(best_h, 2)))

# Espaço pxc1 vs pxc2 para diferentes valores de h
h_to_plot <- seqh[seq(1, length(seqh), length.out = 9)]  # 9 valores
par(mfrow = c(3, 3))  # Grid de 3x3

for (h in h_to_plot) {
  pxc1vec <- numeric(Nall_tst)
  pxc2vec <- numeric(Nall_tst)
  
  for (i in 1:Nall_tst) {
    pxc1vec[i] <- kdemulti(xtst[i, ], xc1_tr, h)
    pxc2vec[i] <- kdemulti(xtst[i, ], xc2_tr, h)
  }
  
  pxc1c2 <- cbind(pxc1vec, pxc2vec)
  plot(pxc1c2[, 1], pxc1c2[, 2],
       col = colvec[1 + (ytst + 1) / 2],
       xlab = "pxc1", ylab = "pxc2",
       main = paste("h =", round(h, 2)))
}

#table.predict.real()