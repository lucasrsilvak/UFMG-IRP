import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import (
    load_breast_cancer, load_digits, load_wine, load_iris,
    load_diabetes, fetch_openml
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# === Configurações de plot ===
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

np.random.seed(1234)  # Reprodutibilidade

# === Funções Auxiliares ===
def plot_espaco_verossimilhanca(pall_train, y_train, y_kmeans, h, j, nome_dataset, kernel, sufixo=''):
    plt.figure(figsize=(6, 6))
    plt.scatter(pall_train[y_train == 0, 0], pall_train[y_train == 0, 1], color='red', label='Classe 0', alpha=0.6)
    plt.scatter(pall_train[y_train == 1, 0], pall_train[y_train == 1, 1], color='blue', label='Classe 1', alpha=0.6)
    erros = (y_train != y_kmeans)
    plt.scatter(pall_train[erros, 0], pall_train[erros, 1], facecolors='none', edgecolors='black', marker='o', s=80, label='Erro do KMeans')
    plt.xlabel('Afinidade com classe 0')
    plt.ylabel('Afinidade com classe 1')
    plt.title(f'Espaço de Verossimilhança - {nome_dataset} - h={h:.2f} c={j:.2f} - Kernel={kernel}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/{nome_dataset}_espaco_verossimilhanca_{kernel}_h{h:.2f}_c{j:.2f}_{sufixo}.pdf")
    plt.close()

def alinhar_labels(y_true, y_pred):
    acc1 = accuracy_score(y_true, y_pred)
    acc2 = accuracy_score(y_true, 1 - y_pred)
    return y_pred if acc1 >= acc2 else 1 - y_pred

def calcular_espaco_verossimilhanca(X, y, gamma, coef0):
    if np.sum(y == 0) == 0 or np.sum(y == 1) == 0:
        return None, None
    K = np.tanh(gamma * X @ X.T + coef0)
    idx0 = (y == 0)
    idx1 = (y == 1)
    afinidade_classe0 = K[:, idx0].sum(axis=1) / idx0.sum()
    afinidade_classe1 = K[:, idx1].sum(axis=1) / idx1.sum()
    pall = np.column_stack((afinidade_classe0, afinidade_classe1))
    return pall, K

def calcular_metrica_combinada(pall_train, y_train):
    idx0 = (y_train == 0)
    idx1 = (y_train == 1)
    V1 = pall_train[idx0].mean(axis=0)
    V2 = pall_train[idx1].mean(axis=0)
    norma_diff = np.linalg.norm(V1 - V2)
    produto_interno = np.dot(V1, V2)
    norma_v1 = np.linalg.norm(V1)
    norma_v2 = np.linalg.norm(V2)
    cos_theta = 0 if norma_v1 == 0 or norma_v2 == 0 else produto_interno / (norma_v1 * norma_v2)
    return norma_diff * cos_theta

def calcular_gap_statistic(X, n_refs=10, n_clusters=2):
    from sklearn.metrics import pairwise_distances

    def within_cluster_dispersion(X, labels):
        W = 0
        for k in range(n_clusters):
            cluster_k = X[labels == k]
            if len(cluster_k) > 1:
                dists = pairwise_distances(cluster_k)
                W += np.sum(dists) / (2.0 * len(cluster_k))
        return W

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=123)
    km.fit(X)
    Wk = within_cluster_dispersion(X, km.labels_)

    bounds = np.vstack([np.min(X, axis=0), np.max(X, axis=0)]).T
    Wk_refs = np.zeros(n_refs)
    for i in range(n_refs):
        X_ref = np.random.uniform(bounds[:, 0], bounds[:, 1], size=X.shape)
        km_ref = KMeans(n_clusters=n_clusters, n_init=10, random_state=123 + i)
        km_ref.fit(X_ref)
        Wk_refs[i] = within_cluster_dispersion(X_ref, km_ref.labels_)

    gap = np.log(np.mean(Wk_refs)) - np.log(Wk)
    return gap

def normalizar_grid(grid):
    valid = grid[~np.isnan(grid)]
    if len(valid) == 0 or np.nanmax(valid) == np.nanmin(valid):
        return np.zeros_like(grid)
    return (grid - np.nanmin(valid)) / (np.nanmax(valid) - np.nanmin(valid))

# === Função Principal ===
def rodar_busca_sigmoid(nome_dataset):
    print(f"\nRodando busca 2D (gamma x coef0) para: {nome_dataset.upper()}")

    def carregar_dataset(nome):
        if nome == 'breast_cancer':
            data = load_breast_cancer()
            return data.data, data.target
        elif nome == 'digits':
            d = load_digits()
            return d.data, (d.target >= 5).astype(int)
        elif nome == 'wine':
            d = load_wine()
            return d.data, (d.target == 0).astype(int)
        elif nome == 'iris':
            d = load_iris()
            return d.data, (d.target != 0).astype(int)
        elif nome == 'diabetes':
            d = load_diabetes()
            return d.data, (d.target > np.median(d.target)).astype(int)
        elif nome == 'spambase':
            d = fetch_openml(name="spambase", as_frame=False, parser='liac-arff')
            return d.data, d.target.astype(int)
        elif nome == 'spiral':
            def gerar_spirals(n_samples=300, noise=0.1):
                n = np.sqrt(np.random.rand(n_samples)) * 390 * (2 * np.pi) / 360
                d1x = -np.cos(n) * n + np.random.rand(n_samples) * noise
                d1y = np.sin(n) * n + np.random.rand(n_samples) * noise
                X1 = np.vstack([d1x, d1y]).T
                X2 = np.vstack([
                    np.cos(n) * n + np.random.rand(n_samples) * noise,
                    -np.sin(n) * n + np.random.rand(n_samples) * noise
                ]).T
                X = np.vstack([X1, X2])
                y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
                return X, y
            return gerar_spirals()
        else:
            raise ValueError("Dataset desconhecido")

    X, y = carregar_dataset(nome_dataset)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123)

    gammas = np.logspace(-3, 1, 15)
    coef0s = np.linspace(-1, 1, 10)

    gap_grid, metrica_grid, acc_train_grid, acc_test_grid = [], [], [], []

    for g in gammas:
        gap_row, metrica_row, acc_train_row, acc_test_row = [], [], [], []
        for c0 in coef0s:
            try:
                K_train = np.tanh(g * X_train @ X_train.T + c0)
                K_test = np.tanh(g * X_test @ X_train.T + c0)
                clf = SVC(kernel='precomputed', C=1.0)
                clf.fit(K_train, y_train)

                y_pred_train = alinhar_labels(y_train, clf.predict(K_train))
                y_pred_test = alinhar_labels(y_test, clf.predict(K_test))

                acc_train = accuracy_score(y_train, y_pred_train)
                acc_test = accuracy_score(y_test, y_pred_test)

                pall, _ = calcular_espaco_verossimilhanca(X_train, y_pred_train, g, c0)
                if pall is None:
                    gap_row.append(np.nan)
                    metrica_row.append(np.nan)
                    acc_train_row.append(np.nan)
                    acc_test_row.append(np.nan)
                    continue

                y_kmeans = alinhar_labels(y_train, KMeans(n_clusters=2, n_init=10, random_state=123).fit_predict(pall))
                gap = calcular_gap_statistic(pall, n_refs=10, n_clusters=2)
                metrica = calcular_metrica_combinada(pall, y_kmeans)

                gap_row.append(gap)
                metrica_row.append(metrica)
                acc_train_row.append(acc_train)
                acc_test_row.append(acc_test)

            except Exception as e:
                print(f"Erro com gamma={g:.3f}, coef0={c0:.3f}: {e}")
                gap_row.append(np.nan)
                metrica_row.append(np.nan)
                acc_train_row.append(np.nan)
                acc_test_row.append(np.nan)

        gap_grid.append(gap_row)
        metrica_grid.append(metrica_row)
        acc_train_grid.append(acc_train_row)
        acc_test_grid.append(acc_test_row)

    gap_grid = np.array(gap_grid)
    metrica_grid = np.array(metrica_grid)
    acc_train_grid = np.array(acc_train_grid)
    acc_test_grid = np.array(acc_test_grid)

    gap_norm = normalizar_grid(gap_grid)
    metrica_norm = normalizar_grid(metrica_grid)
    alpha = 0.75
    score_grid = alpha * gap_norm + (1 - alpha) * metrica_norm

    best_idx = np.unravel_index(np.nanargmax(score_grid), score_grid.shape)
    melhor_g = gammas[best_idx[0]]
    melhor_c0 = coef0s[best_idx[1]]
    melhor_score = score_grid[best_idx]
    melhor_acc_train = acc_train_grid[best_idx]
    melhor_acc_test = acc_test_grid[best_idx]

    # === Plots ===
    def salvar_imagem(grid, cmap, titulo, nome):
        plt.figure(figsize=(10, 6))
        plt.imshow(grid, aspect='auto', origin='lower',
                   extent=[coef0s[0], coef0s[-1], gammas[0], gammas[-1]],
                   cmap=cmap, interpolation='none')
        plt.colorbar()
        plt.xlabel("coef0")
        plt.ylabel("gamma")
        plt.title(titulo)
        plt.tight_layout()
        plt.savefig(nome)
        plt.close()

    salvar_imagem(score_grid, 'viridis', f"{nome_dataset.upper()} - Score Combinado", f"results/{nome_dataset}_grid_sigmoid.pdf")
    salvar_imagem(acc_test_grid, 'plasma', f"{nome_dataset.upper()} - Acurácia de Teste", f"results/{nome_dataset}_grid_acuracia_teste.pdf")
    salvar_imagem(gap_grid, 'viridis', f"{nome_dataset.upper()} - Gap Statistic", f"results/{nome_dataset}_grid_gap.pdf")
    salvar_imagem(metrica_grid, 'cividis', f"{nome_dataset.upper()} - Métrica Combinada", f"results/{nome_dataset}_grid_metrica_combinada.pdf")

    # === Superfície 3D ===
    G, C0 = np.meshgrid(coef0s, gammas)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(C0, G, score_grid, cmap='viridis', edgecolor='k')
    ax.set_xlabel('gamma')
    ax.set_ylabel('coef0')
    ax.set_zlabel('Score combinado (normalizado)')
    ax.set_title(f"{nome_dataset.upper()} - Superfície do Score (Sigmoid)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig(f"results/{nome_dataset}_superficie_sigmoid.pdf")
    plt.close()

    print(f"\nMelhor combinação encontrada:")
    print(f"  gamma = {melhor_g:.4f}")
    print(f"  coef0 = {melhor_c0:.4f}")
    print(f"  Score combinado (normalizado) = {melhor_score:.4f}")
    print(f"  Acurácia Treino: {melhor_acc_train:.4f}")
    print(f"  Acurácia Teste:  {melhor_acc_test:.4f}")

# === Execução ===
datasets = ['breast_cancer', 'digits', 'wine', 'iris', 'diabetes', 'spiral', 'spambase', 'gaussians']
for ds in datasets:
    rodar_busca_sigmoid(ds)
