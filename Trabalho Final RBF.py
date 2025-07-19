import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import (
    load_breast_cancer, load_digits, load_wine, load_iris,
    load_diabetes, fetch_openml
)
from sklearn.metrics import pairwise_distances, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
import os

# === Configuração Inicial ===
os.makedirs("results", exist_ok=True)
np.random.seed(123)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 12,
})

# === Funções de Visualização ===
def plot_espaco_verossimilhanca(pall_train, y_train, h, y_kmeans, nome_dataset, kernel, sufixo=''):
    plt.scatter(pall_train[y_train == 0, 0], pall_train[y_train == 0, 1], color='red', label='Classe 0', alpha=0.6)
    plt.scatter(pall_train[y_train == 1, 0], pall_train[y_train == 1, 1], color='blue', label='Classe 1', alpha=0.6)

    erros = (y_train != y_kmeans)
    plt.scatter(
        pall_train[erros][:, 0],
        pall_train[erros][:, 1],
        facecolors='none', edgecolors='black', marker='o', s=80,
        label='Erro do KMeans'
    )

    plt.xlabel('Afinidade com classe 0')
    plt.ylabel('Afinidade com classe 1')
    plt.title(f'Espaço de Verossimilhança - {nome_dataset} - Kernel={kernel}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{nome_dataset}_espaco_verossimilhanca.png")
    plt.close()

def plot_spiral(X, y):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Classe 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Classe 1')
    plt.title("Dataset Espiral Original")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/spiral_dataset.pdf")
    plt.close()

# === Funções Auxiliares ===
def alinhar_labels(y_true, y_pred):
    acc1 = accuracy_score(y_true, y_pred)
    acc2 = accuracy_score(y_true, 1 - y_pred)
    return y_pred if acc1 >= acc2 else 1 - y_pred

def calcular_metrica_combinada(pall_train, y_train):
    idx0 = (y_train == 0)
    idx1 = (y_train == 1)
    V1 = pall_train[idx0].mean(axis=0)
    V2 = pall_train[idx1].mean(axis=0)
    norma_diff = np.linalg.norm(V1 - V2)
    produto_interno = np.dot(V1, V2)
    norma_v1 = np.linalg.norm(V1)
    norma_v2 = np.linalg.norm(V2)
    cos_theta = produto_interno / (norma_v1 * norma_v2) if norma_v1 and norma_v2 else 0
    return norma_diff * cos_theta

def calcular_espaco_verossimilhanca(X, y, h, kernel='rbf'):
    if kernel == 'rbf':
        dists = pairwise_distances(X)
        K = np.exp(-dists**2 / h**2)
    elif kernel == 'sigmoid':
        gamma = 1 / h**2
        coef0 = 0.5
        K = np.tanh(gamma * X @ X.T + coef0)
    else:
        raise ValueError("Kernel não reconhecido")

    idx0 = (y == 0)
    idx1 = (y == 1)
    afinidade_classe0 = K[:, idx0].sum(axis=1) / idx0.sum()
    afinidade_classe1 = K[:, idx1].sum(axis=1) / idx1.sum()

    pall = np.column_stack((afinidade_classe0, afinidade_classe1))
    return pall, K

def gap_statistic(X, n_refs=10, random_state=123):
    np.random.seed(random_state)
    shape = X.shape
    tops = X.max(axis=0)
    bottoms = X.min(axis=0)
    dists_ref = []

    for _ in range(n_refs):
        random_points = np.random.uniform(bottoms, tops, size=shape)
        kmeans = KMeans(n_clusters=1, n_init=5, random_state=random_state).fit(random_points)
        dists_ref.append(kmeans.inertia_)

    ref_log_disp = np.mean(np.log(dists_ref))
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit(X)
    orig_disp = kmeans.inertia_

    gap = -(ref_log_disp - np.log(orig_disp + 1e-10))
    return gap, kmeans.labels_

# === Geração de Dados Sintéticos ===
def gerar_spirals(n_samples=500, noise=0):
    n = np.sqrt(np.random.rand(n_samples)) * 390 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples) * noise
    X1 = np.vstack([d1x, d1y]).T
    X2 = np.vstack([np.cos(n)*n + np.random.rand(n_samples) * noise, -np.sin(n)*n + np.random.rand(n_samples) * noise]).T
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    return X, y

def gerar_gaussians(n_samples=300, centers=[(2, 2), (4, 4)], std=0.6):
    n_per_class = n_samples // 2
    X0 = np.random.normal(loc=centers[0], scale=std, size=(n_per_class, 2))
    X1 = np.random.normal(loc=centers[1], scale=std, size=(n_per_class, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Classe 0')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Classe 1')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Amostras de duas Gaussianas')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/gaussiansplot.png')
    plt.close()
    return X, y, ['x1', 'x2']

# === Processamento Principal ===
def processar_modelos_com_teste(X_train, y_train, X_test, y_test, seqh, kernel='rbf', C=1.0, nome_dataset='dataset'):
    scores_gap, espacos, labels_kmeans = [], [], []
    scores_svm_train, scores_svm_test, gaps = [], [], []
    metricas_combinadas = []

    for h in seqh:
        try:
            if kernel == 'rbf':
                gamma = 1 / h**2
                K_train = rbf_kernel(X_train, X_train, gamma=gamma)
                K_test = rbf_kernel(X_test, X_train, gamma=gamma)
            elif kernel == 'sigmoid':
                gamma = 1 / h
                coef0 = 1.0
                K_train = np.tanh(gamma * X_train @ X_train.T + coef0)
                K_test = np.tanh(gamma * X_test @ X_train.T + coef0)
            else:
                raise ValueError("Kernel não reconhecido")

            svc = SVC(C=C, kernel='precomputed')
            svc.fit(K_train, y_train)

            y_pred_train = alinhar_labels(y_train, svc.predict(K_train))
            y_pred_test = alinhar_labels(y_test, svc.predict(K_test))

            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            gap_train_test = acc_train - acc_test

            pall_train, _ = calcular_espaco_verossimilhanca(X_train, y_pred_train, h, kernel)

            if np.any(np.std(pall_train, axis=0) < 1e-6):
                scores_gap.append(None)
                espacos.append(None)
                labels_kmeans.append(None)
                scores_svm_train.append(acc_train)
                scores_svm_test.append(acc_test)
                gaps.append(gap_train_test)
                metricas_combinadas.append(None)
                continue

            gap_score, km_labels = gap_statistic(pall_train)
            y_pred_km = alinhar_labels(y_train, km_labels)
            metrica_combinada = calcular_metrica_combinada(pall_train, y_pred_km)

            scores_gap.append(gap_score)
            espacos.append(pall_train)
            labels_kmeans.append(y_pred_km)
            scores_svm_train.append(acc_train)
            scores_svm_test.append(acc_test)
            gaps.append(gap_train_test)
            metricas_combinadas.append(metrica_combinada)

        except Exception as e:
            print(f"Erro com h={h:.2f}: {e}")
            scores_gap.append(None)
            espacos.append(None)
            labels_kmeans.append(None)
            scores_svm_train.append(None)
            scores_svm_test.append(None)
            gaps.append(None)
            metricas_combinadas.append(None)

    return scores_gap, espacos, labels_kmeans, scores_svm_train, scores_svm_test, gaps, metricas_combinadas

# === Dataset Loader ===
def carregar_dataset(nome='breast_cancer'):
    if nome == 'breast_cancer':
        data = load_breast_cancer()
        return data.data, data.target, data.feature_names
    elif nome == 'digits':
        data = load_digits()
        return data.data, (data.target >= 5).astype(int), data.feature_names
    elif nome == 'wine':
        data = load_wine()
        return data.data, (data.target == 0).astype(int), data.feature_names
    elif nome == 'iris':
        data = load_iris()
        return data.data, (data.target != 0).astype(int), data.feature_names
    elif nome == 'diabetes':
        data = load_diabetes()
        return data.data, (data.target > np.median(data.target)).astype(int), data.feature_names
    elif nome == 'spiral':
        X, y = gerar_spirals(n_samples=300, noise=2.5)
        plot_spiral(X, y)
        return X, y, ['x1', 'x2']
    elif nome == 'spambase':
        d = fetch_openml(name="spambase", as_frame=False, parser='liac-arff')
        return d.data, d.target.astype(int), d.feature_names
    elif nome == 'gaussians':
        return gerar_gaussians(n_samples=300, centers=[(2, 2), (4, 4)], std=0.75)
    else:
        raise ValueError("Dataset não reconhecido")

# === Loop Principal ===
def rodar_para_dataset(nome_dataset, kernel='rbf'):
    print(f"\nRodando experimento para: {nome_dataset.upper()}")

    X, y, feature_names = carregar_dataset(nome_dataset)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    seqh = np.arange(0.01, 1, 0.01)
    C = 1.0

    scores_gap, espacos, labels_km, scores_svm_train, scores_svm_test, gaps, metricas_combinadas = processar_modelos_com_teste(
        X_train, y_train, X_test, y_test, seqh, kernel, C, nome_dataset
    )

    valid_gap = [v for v in scores_gap if v is not None]
    valid_metrica = [v for v in metricas_combinadas if v is not None]

    gap_norm = [(v - min(valid_gap)) / (max(valid_gap) - min(valid_gap)) if v is not None else None for v in scores_gap]
    metrica_norm = [(v - min(valid_metrica)) / (max(valid_metrica) - min(valid_metrica)) if v is not None else None for v in metricas_combinadas]

    alpha = 0.75
    soma_ponderada = [None if (g is None or m is None) else alpha * g + (1 - alpha) * m for g, m in zip(gap_norm, metrica_norm)]

    valid_indices = [i for i, v in enumerate(soma_ponderada) if v is not None]
    min_index = valid_indices[np.argmax([soma_ponderada[i] for i in valid_indices])]
    min_h = seqh[min_index]

    plt.figure(figsize=(10, 6))
    plt.plot(seqh, scores_svm_train, linestyle=':', label="Acurácia Treino")
    plt.plot(seqh, scores_svm_test, label="Acurácia Teste")
    plt.plot(seqh, gap_norm, linestyle=':', label="Gap Statistic Normalizado")
    plt.plot(seqh, metrica_norm, linestyle=':', label="Ushikoshi Normalizado")
    plt.plot(seqh, soma_ponderada, label=f"Soma Ponderada (a={alpha})")
    plt.axvline(x=min_h, color='red', linestyle='--', label=f'Máximo @ h={min_h:.2f}')
    plt.title(f"{nome_dataset.upper()} - Métricas vs h - Kernel {kernel.upper()}")
    plt.xlabel("h (parâmetro do kernel)")
    plt.ylabel("Métricas / Acurácia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{nome_dataset}_metricas_combinadas_{kernel}_GAP.png")
    plt.close()

    # Plot do espaço de verossimilhança no melhor h
    pall_melhor_h = espacos[min_index]
    if pall_melhor_h is not None:
        plot_espaco_verossimilhanca(pall_melhor_h, y_train, min_h, labels_km[min_index], nome_dataset, kernel)

# === Execução ===
datasets = ['breast_cancer', 'digits', 'wine', 'iris', 'diabetes', 'spiral', 'spambase', 'gaussians']
for dataset in datasets:
    rodar_para_dataset(dataset, kernel='rbf')