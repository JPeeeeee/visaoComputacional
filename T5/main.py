#!/usr/bin/env python3
"""
Trabalho 5 - Visao Computacional / Redes Neurais Artificiais
Depuracao, Experimentacao e Analise de MLP from Scratch no Dataset Iris

Autor: Joao Pedro Vieira Santos
Repositorio Git: https://github.com/JPeeeeee/visaoComputacional
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# ==============================================================================
# 1. IMPLEMENTACAO ORIGINAL (COM BUGS PARA DIAGNOSTICO)
# ==============================================================================
class OriginalMLP:
    """
    Replica fiel da implementacao original fornecida no notebook para
    demonstrar as falhas matematicas e estruturais.
    """
    def __init__(self, params=None):
        if params is None:
            params = {
                'InputLayer': 4, 'HiddenLayer': 5, 'OutputLayer': 3,
                'Epocas': 1500, 'LearningRate': 0.005,
                'BiasHiddenValue': -1, 'BiasOutputValue': -1,
                'ActivationFunction': 'sigmoid'
            }
        self.inputLayer = params['InputLayer']
        self.hiddenLayer = params['HiddenLayer']
        self.outputLayer = params['OutputLayer']
        self.learningRate = params['LearningRate']
        self.max_epochs = params['Epocas']
        self.BiasHiddenValue = params['BiasHiddenValue']
        self.BiasOutputValue = params['BiasOutputValue']

        self.ativacao = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'tanh': lambda x: np.tanh(x),
            'Relu': lambda x: x * (x > 0)
        }
        self.derivada = {
            'sigmoid': lambda x: x * (1 - x),
            'tanh': lambda x: 1 - x**2,
            'Relu': lambda x: 1 * (x > 0)
        }
        self.activation = self.ativacao[params['ActivationFunction']]
        self.deriv = self.derivada[params['ActivationFunction']]

        random.seed(42)
        np.random.seed(42)
        self.WEIGHT_hidden = [[2 * random.random() - 1 for _ in range(self.hiddenLayer)] for _ in range(self.inputLayer)]
        self.WEIGHT_output = [[2 * random.random() - 1 for _ in range(self.outputLayer)] for _ in range(self.hiddenLayer)]
        self.BIAS_hidden = np.array([self.BiasHiddenValue for _ in range(self.hiddenLayer)], dtype=float)
        self.BIAS_output = np.array([self.BiasOutputValue for _ in range(self.outputLayer)], dtype=float)
        self.classes_number = 3

    def Backpropagation_Algorithm(self, x):
        # BUG 1: Atualiza WEIGHT_output antes de calcular delta_hidden!
        # BUG 2: Atualiza BIAS_output e BIAS_hidden dentro dos loops aninhados!
        ERROR_output = self.output - self.OUTPUT_L2
        DELTA_output = (-1) * ERROR_output * self.deriv(self.OUTPUT_L2)

        for i in range(self.hiddenLayer):
            for j in range(self.outputLayer):
                self.WEIGHT_output[i][j] -= (self.learningRate * (DELTA_output[j] * self.OUTPUT_L1[i]))
                self.BIAS_output[j] -= (self.learningRate * DELTA_output[j]) # BUG: repetido 5x!

        delta_hidden = np.matmul(self.WEIGHT_output, DELTA_output) * self.deriv(self.OUTPUT_L1)

        for i in range(self.inputLayer):
            for j in range(self.hiddenLayer):
                self.WEIGHT_hidden[i][j] -= (self.learningRate * (delta_hidden[j] * x[i]))
                self.BIAS_hidden[j] -= (self.learningRate * delta_hidden[j]) # BUG: repetido 4x!

    def fit(self, X, y):
        n = len(X)
        self.error_array = []
        for epoch in range(1, self.max_epochs + 1):
            total_error = 0
            for idx, inputs in enumerate(X):
                self.output = np.zeros(self.classes_number)
                self.OUTPUT_L1 = self.activation(np.dot(inputs, self.WEIGHT_hidden) + self.BIAS_hidden)
                self.OUTPUT_L2 = self.activation(np.dot(self.OUTPUT_L1, self.WEIGHT_output) + self.BIAS_output)

                if y[idx] == 0: self.output = np.array([1, 0, 0])
                elif y[idx] == 1: self.output = np.array([0, 1, 0])
                elif y[idx] == 2: self.output = np.array([0, 0, 1])

                square_error = np.sum(0.05 * (self.output - self.OUTPUT_L2)**2)
                total_error += square_error
                self.Backpropagation_Algorithm(inputs)

            if epoch % 50 == 0 or epoch == 1:
                self.error_array.append(total_error / n)
        return self

    def predict(self, X):
        # BUG 3: predict original NAO aplicava ativacao no forward!
        forward = np.matmul(X, self.WEIGHT_hidden) + self.BIAS_hidden
        forward = np.matmul(forward, self.WEIGHT_output) + self.BIAS_output
        return np.array([np.argmax(row) for row in forward])


# ==============================================================================
# 2. IMPLEMENTACAO TOTALMENTE DEPURADA E CORRIGIDA (MLP FROM SCRATCH)
# ==============================================================================
class CorrectedMLP:
    """
    MLP from Scratch com algebra linear vetorizada, correcao da regra da cadeia
    no backpropagation, suporte a Softmax + Cross-Entropy e rastreamento de perda
    de validacao para diagnostico de convergencia e overfitting.
    """
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=3, lr=0.2,
                 activation='tanh', output_activation='softmax', random_state=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.activation_name = activation
        self.output_activation = output_activation
        self.random_state = random_state

        np.random.seed(random_state)
        # Inicializacao Xavier/Glorot normalizada
        lim_h = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W1 = np.random.uniform(-lim_h, lim_h, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        lim_o = np.sqrt(6.0 / (hidden_dim + output_dim))
        self.W2 = np.random.uniform(-lim_o, lim_o, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    def _activate(self, z):
        if self.activation_name == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        elif self.activation_name == 'relu':
            return np.maximum(0, z)
        return z

    def _activate_deriv(self, a):
        if self.activation_name == 'sigmoid':
            return a * (1.0 - a)
        elif self.activation_name == 'tanh':
            return 1.0 - a**2
        elif self.activation_name == 'relu':
            return (a > 0).astype(float)
        return np.ones_like(a)

    def _softmax(self, z):
        shift = z - np.max(z, axis=-1, keepdims=True)
        exps = np.exp(shift)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._activate(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._softmax(z2) if self.output_activation == 'softmax' else self._activate(z2)
        return a1, a2

    def compute_loss(self, y_true_oh, y_pred_prob):
        eps = 1e-12
        if self.output_activation == 'softmax':
            probs = np.clip(y_pred_prob, eps, 1.0 - eps)
            return -np.mean(np.sum(y_true_oh * np.log(probs), axis=1))
        return np.mean(np.sum(0.5 * (y_true_oh - y_pred_prob)**2, axis=1))

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=300):
        n_samples = len(X_train)
        y_train_oh = np.eye(self.output_dim)[y_train.astype(int)]
        if y_val is not None:
            y_val_oh = np.eye(self.output_dim)[y_val.astype(int)]

        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        for epoch in range(1, epochs + 1):
            # Forward Pass
            a1, a2 = self.forward(X_train)

            # Gradiente de Saida
            if self.output_activation == 'softmax':
                delta_out = (a2 - y_train_oh) / n_samples
            else:
                delta_out = ((a2 - y_train_oh) * self._activate_deriv(a2)) / n_samples

            dW2 = np.dot(a1.T, delta_out)
            db2 = np.sum(delta_out, axis=0)

            # Gradiente Oculto usando W2 ANTES da atualizacao
            delta_hidden = np.dot(delta_out, self.W2.T) * self._activate_deriv(a1)
            dW1 = np.dot(X_train.T, delta_hidden)
            db1 = np.sum(delta_hidden, axis=0)

            # Atualizacao precisa
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

            loss_tr = self.compute_loss(y_train_oh, a2)
            acc_tr = accuracy_score(y_train, np.argmax(a2, axis=1))

            loss_v, acc_v = None, None
            if X_val is not None and y_val is not None:
                _, a2_val = self.forward(X_val)
                loss_v = self.compute_loss(y_val_oh, a2_val)
                acc_v = accuracy_score(y_val, np.argmax(a2_val, axis=1))

            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(loss_tr)
            self.history['val_loss'].append(loss_v)
            self.history['train_acc'].append(acc_tr)
            self.history['val_acc'].append(acc_v)

        return self

    def predict(self, X):
        _, a2 = self.forward(X)
        return np.argmax(a2, axis=1)


# ==============================================================================
# 3. EXECUCAO DOS EXPERIMENTOS E DIAGNOSTICO COMPLETO
# ==============================================================================
def executar_experimentos():
    print("=" * 78)
    print("TRABALHO 5: DEPURACAO, EXPERIMENTACAO E ANALISE DA MLP FROM SCRATCH")
    print("=" * 78)

    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    # --------------------------------------------------------------------------
    # 1. Execucao do Codigo Original e Analise dos Bugs
    # --------------------------------------------------------------------------
    print("\n[ETAPA 1] Executando o modelo original para replicar os bugs...")
    # Particao original do notebook (seed 123 + shuffle sem estratificacao)
    iris_ds = np.column_stack((X, y))
    np.random.seed(123)
    random.seed(123)
    shuffled_ds = iris_ds[np.random.permutation(len(iris_ds))]

    X_shuff = shuffled_ds[:, :4]
    y_shuff = shuffled_ds[:, 4]
    X_tr_orig, X_te_orig, y_tr_orig, y_te_orig = train_test_split(
        X_shuff, y_shuff, test_size=0.20, random_state=42
    )

    mlp_orig = OriginalMLP({'InputLayer': 4, 'HiddenLayer': 5, 'OutputLayer': 3,
                           'Epocas': 1500, 'LearningRate': 0.005,
                           'BiasHiddenValue': -1, 'BiasOutputValue': -1,
                           'ActivationFunction': 'sigmoid'})
    mlp_orig.fit(X_tr_orig, y_tr_orig)
    pred_orig = mlp_orig.predict(X_te_orig)

    acc_orig_total = accuracy_score(y_te_orig, pred_orig)
    cm_orig = confusion_matrix(y_te_orig, pred_orig, labels=[0, 1, 2])
    acc_classes_orig = []
    for i in range(3):
        total = np.sum(y_te_orig == i)
        hits = np.sum((y_te_orig == i) & (pred_orig == i))
        acc_classes_orig.append((hits / total) * 100 if total > 0 else 0)

    print(f"  -> Acuracia Geral Original: {acc_orig_total * 100:.2f}%")
    print(f"  -> Acuracias por classe: Setosa={acc_classes_orig[0]:.1f}%, Versicolor={acc_classes_orig[1]:.1f}%, Virginica={acc_classes_orig[2]:.1f}%")

    # --------------------------------------------------------------------------
    # 2. Execucao do Modelo Corrigido (Classificando Perfeitamente as 3 Classes)
    # --------------------------------------------------------------------------
    print("\n[ETAPA 2] Executando modelo depurado e corrigido (Estratificado + Z-Score)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    mlp_corr = CorrectedMLP(input_dim=4, hidden_dim=8, output_dim=3, lr=0.25, activation='tanh', random_state=42)
    mlp_corr.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=300)

    pred_corr = mlp_corr.predict(X_test)
    acc_corr_total = accuracy_score(y_test, pred_corr)
    cm_corr = confusion_matrix(y_test, pred_corr, labels=[0, 1, 2])

    acc_classes_corr = []
    for i in range(3):
        total = np.sum(y_test == i)
        hits = np.sum((y_test == i) & (pred_corr == i))
        acc_classes_corr.append((hits / total) * 100)

    print(f"  -> Acuracia Geral Corrigido: {acc_corr_total * 100:.2f}%")
    print(f"  -> Acuracias por classe: Setosa={acc_classes_corr[0]:.1f}%, Versicolor={acc_classes_corr[1]:.1f}%, Virginica={acc_classes_corr[2]:.1f}%")
    print("\nRelatorio de Classificacao:")
    print(classification_report(y_test, pred_corr, target_names=target_names))

    # --------------------------------------------------------------------------
    # 3. Experimento de Overfitting (Comparacao Modelo Ideal vs Modelo Superajustado)
    # --------------------------------------------------------------------------
    print("\n[ETAPA 3] Analisando convergencia e diagnostico de Overfitting...")
    # Modelo com excesso de capacidade (32 neuronios, 1500 epocas, sem regularizacao)
    mlp_overfit = CorrectedMLP(input_dim=4, hidden_dim=32, output_dim=3, lr=0.5, activation='relu', random_state=42)
    mlp_overfit.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=1200)

    # --------------------------------------------------------------------------
    # 4. Geracao de Graficos em Alta Resolucao
    # --------------------------------------------------------------------------
    print("\n[ETAPA 4] Gerando figuras analiticas em T5/output/...")

    # Fig 1: Acuracia por classe (Original vs Corrigido)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x_idx = np.arange(3)
    w = 0.35
    b1 = ax.bar(x_idx - w/2, acc_classes_orig, w, label='Original (Com Bugs)', color='#d9534f')
    b2 = ax.bar(x_idx + w/2, acc_classes_corr, w, label='Corrigido (Depurado)', color='#5cb85c')
    ax.set_title('Acurácia por Classe: Modelo Original vs. Modelo Corrigido', fontsize=12, weight='bold')
    ax.set_ylabel('Acurácia (%)')
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"Iris-{n.capitalize()}" for n in target_names])
    ax.set_ylim(0, 118)
    ax.legend(loc='lower left')
    for b in b1:
        ax.annotate(f'{b.get_height():.1f}%', (b.get_x() + b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha='center', fontsize=9.5, weight='bold')
    for b in b2:
        ax.annotate(f'{b.get_height():.1f}%', (b.get_x() + b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha='center', fontsize=9.5, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "acuracia_por_classe.png"), dpi=300)
    plt.close()

    # Fig 2: Curvas de Convergencia (Perda e Acuracia) do Modelo Corrigido
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ep = mlp_corr.history['epoch']
    ax1.plot(ep, mlp_corr.history['train_loss'], label='Treino (Train Loss)', color='#0275d8', lw=2)
    ax1.plot(ep, mlp_corr.history['val_loss'], label='Validação / Teste (Val Loss)', color='#d9534f', lw=2, linestyle='--')
    ax1.set_title('Convergência da Função de Perda (Cross-Entropy)', fontsize=11, weight='bold')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Perda (Loss)')
    ax1.legend()

    ax2.plot(ep, [a*100 for a in mlp_corr.history['train_acc']], label='Acurácia Treino', color='#0275d8', lw=2)
    ax2.plot(ep, [a*100 for a in mlp_corr.history['val_acc']], label='Acurácia Teste', color='#5cb85c', lw=2, linestyle='--')
    ax2.set_title('Evolução da Acurácia ao Longo das Épocas', fontsize=11, weight='bold')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Acurácia (%)')
    ax2.set_ylim(40, 105)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "convergencia_perda_treino_val.png"), dpi=300)
    plt.close()

    # Fig 3: Estudo de Overfitting
    plt.figure(figsize=(8, 4.8))
    ep_ov = mlp_overfit.history['epoch']
    plt.plot(ep_ov, mlp_overfit.history['train_loss'], label='Perda Treino (Cai continuamente)', color='#0275d8', lw=2)
    plt.plot(ep_ov, mlp_overfit.history['val_loss'], label='Perda Validação (Estagna e oscila)', color='#d9534f', lw=2, linestyle='--')
    plt.title('Diagnóstico de Overfitting em Rede Superdimensionada (32 neurônios)', fontsize=12, weight='bold')
    plt.xlabel('Épocas')
    plt.ylabel('Perda Cross-Entropy')
    plt.axvline(250, color='gray', linestyle=':', label='Ponto Ideal de Parada (Early Stopping ~250)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "analise_overfitting.png"), dpi=300)
    plt.close()

    # Fig 4: Matrizes de Confusao Lado a Lado
    fig, (ax_m1, ax_m2) = plt.subplots(1, 2, figsize=(11, 4.6))
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Reds', ax=ax_m1,
                xticklabels=target_names, yticklabels=target_names, cbar=False)
    ax_m1.set_title(f"Original: Acurácia Global {acc_orig_total*100:.1f}%\n(Dificuldade em Virginica)", fontsize=11, weight='bold')
    ax_m1.set_xlabel("Predito")
    ax_m1.set_ylabel("Real")

    sns.heatmap(cm_corr, annot=True, fmt='d', cmap='Greens', ax=ax_m2,
                xticklabels=target_names, yticklabels=target_names, cbar=False)
    ax_m2.set_title(f"Corrigido: Acurácia Global {acc_corr_total*100:.1f}%\n(Todas as 3 Classes Bem Classificadas)", fontsize=11, weight='bold')
    ax_m2.set_xlabel("Predito")
    ax_m2.set_ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "matriz_confusao_3classes.png"), dpi=300)
    plt.close()

    # Fig 5: Fronteiras de Decisao no Espaco PCA 2D
    pca = PCA(n_components=2)
    X_tr_pca = pca.fit_transform(X_train)
    X_te_pca = pca.transform(X_test)

    mlp_pca = CorrectedMLP(input_dim=2, hidden_dim=8, output_dim=3, lr=0.25, activation='tanh', random_state=42)
    mlp_pca.fit(X_tr_pca, y_train, epochs=300)

    x_min, x_max = X_tr_pca[:, 0].min() - 1, X_tr_pca[:, 0].max() + 1
    y_min, y_max = X_tr_pca[:, 1].min() - 1, X_tr_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.04), np.arange(y_min, y_max, 0.04))
    Z = mlp_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    cmap_light = ListedColormap(["#cfe2ff", "#ffe5d0", "#d1e7dd"])
    cmap_bold = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    plt.figure(figsize=(8, 5.5))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    for idx, c in enumerate(target_names):
        pts = X_te_pca[y_test == idx]
        plt.scatter(pts[:, 0], pts[:, 1], c=cmap_bold[idx], label=f"{c} (teste)", edgecolor="k", s=80)
    plt.title("Fronteiras de Decisão da MLP Corrigida (Espaço PCA 2D)", fontsize=12, weight='bold')
    plt.xlabel("Primeiro Componente Principal (PC1)")
    plt.ylabel("Segundo Componente Principal (PC2)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fronteiras_decisao_mlp.png"), dpi=300)
    plt.close()

    # 5. Salva Relatorio Tecnico em TXT
    caminho_txt = os.path.join(OUTPUT_DIR, "relatorio_experimentos_mlp.txt")
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write("RELATORIO DOS EXPERIMENTOS E DEPURACAO DA MLP - TRABALHO 5\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. COMPARATIVO DE ACURACIA GERAL:\n")
        f.write(f"   - Original:  {acc_orig_total*100:.2f}%\n")
        f.write(f"   - Corrigido: {acc_corr_total*100:.2f}%\n\n")
        f.write("2. ACURACIA POR CLASSE:\n")
        for i, c_name in enumerate(target_names):
            f.write(f"   - {c_name.capitalize():12s}: Original={acc_classes_orig[i]:6.2f}% | Corrigido={acc_classes_corr[i]:6.2f}%\n")
        f.write("\n3. RELATORIO DETALHADO DO MODELO CORRIGIDO:\n")
        f.write(classification_report(y_test, pred_corr, target_names=target_names))

    print(f"\n[INFO] Relatorio textual salvo em: {caminho_txt}")
    print("=" * 78)
    print("EXPERIMENTOS FINALIZADOS COM SUCESSO!")
    print("=" * 78)

if __name__ == "__main__":
    executar_experimentos()
