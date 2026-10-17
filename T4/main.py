#!/usr/bin/env python3
"""
Trabalho 4 - Visao Computacional / Aprendizado de Maquina
Analise Exploratoria, PCA e Classificacao K-NN no Iris Dataset (UCI)

Autor: Joao Pedro Vieira Santos
Repositorio Git: https://github.com/JPeeeeee/visaoComputacional
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuracoes de diretorio e estilo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

# ==============================================================================
# ETAPA 1: CARREGAMENTO DOS DADOS
# ==============================================================================
def carregar_dados():
    iris = load_iris()
    nomes_colunas = [
        "comprimento_sepala",
        "largura_sepala",
        "comprimento_petala",
        "largura_petala"
    ]
    df = pd.DataFrame(iris.data, columns=nomes_colunas)
    df["categoria"] = [iris.target_names[i] for i in iris.target]
    
    # Salvar copia em CSV
    caminho_csv = os.path.join(DATA_DIR, "iris.csv")
    df.to_csv(caminho_csv, index=False)
    print(f"[INFO] Dataset Iris carregado com {len(df)} amostras e salvo em: {caminho_csv}")
    return df, nomes_colunas, iris.target_names

# ==============================================================================
# ETAPA 2: ESTATISTICA DESCRITIVA (MEDIA, DESVIO, MODA, FREQUENCIAS)
# ==============================================================================
def calcular_moda(series):
    m = stats.mode(series, keepdims=True)
    return float(m.mode[0])

def analisar_estatisticas(df, colunas):
    print("\n" + "=" * 78)
    print("ETAPA 1: ESTATISTICAS DESCRITIVAS GERAIS E POR CATEGORIA")
    print("=" * 78)

    # 1. Frequencia de cada categoria
    freq_absoluta = df["categoria"].value_counts()
    freq_relativa = df["categoria"].value_counts(normalize=True) * 100
    df_freq = pd.DataFrame({
        "Frequencia Absoluta": freq_absoluta,
        "Frequencia Relativa (%)": freq_relativa
    })
    print("\n--- Frequencia das Categorias ---")
    print(df_freq)

    # 2. Estatisticas Gerais
    stats_geral = []
    for col in colunas:
        stats_geral.append({
            "Variavel": col,
            "Media": df[col].mean(),
            "Desvio Padrao": df[col].std(),
            "Moda": calcular_moda(df[col]),
            "Min": df[col].min(),
            "Max": df[col].max()
        })
    df_stats_geral = pd.DataFrame(stats_geral).set_index("Variavel")
    print("\n--- Estatisticas Gerais (Todas as Categorias) ---")
    print(df_stats_geral.round(3))

    # 3. Estatisticas por Categoria
    linhas_cat = []
    for cat, grupo in df.groupby("categoria"):
        for col in colunas:
            linhas_cat.append({
                "Categoria": cat,
                "Variavel": col,
                "Media": grupo[col].mean(),
                "Desvio Padrao": grupo[col].std(),
                "Moda": calcular_moda(grupo[col]),
                "Min": grupo[col].min(),
                "Max": grupo[col].max()
            })
    df_stats_cat = pd.DataFrame(linhas_cat)
    print("\n--- Estatisticas por Categoria ---")
    print(df_stats_cat.to_string(index=False))

    # Salvar em arquivo de texto
    caminho_out = os.path.join(OUTPUT_DIR, "estatisticas_descritivas.txt")
    with open(caminho_out, "w", encoding="utf-8") as f:
        f.write("RELATORIO DE ESTATISTICAS DESCRITIVAS - DATASET IRIS\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. FREQUENCIA POR CATEGORIA:\n")
        f.write(df_freq.to_string() + "\n\n")
        f.write("2. ESTATISTICAS GERAIS (GLOBAL):\n")
        f.write(df_stats_geral.round(3).to_string() + "\n\n")
        f.write("3. ESTATISTICAS POR CATEGORIA:\n")
        f.write(df_stats_cat.round(3).to_string(index=False) + "\n")
    print(f"\n[INFO] Estatisticas salvas em: {caminho_out}")

# ==============================================================================
# ETAPA 3: RELACAO ENTRE VARIAVEIS E DISTRIBUICAO NAS 4 DIMENSOES
# ==============================================================================
def plotar_relacoes_e_distribuicoes(df, colunas):
    print("\n" + "=" * 78)
    print("ETAPA 2: ANALISE DE RELACOES E DISTRIBUICAO 4D")
    print("=" * 78)

    cores = {"setosa": "#1f77b4", "versicolor": "#ff7f0e", "virginica": "#2ca02c"}

    # 1. Relacao Comprimento x Largura da Sepala
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="comprimento_sepala",
        y="largura_sepala",
        hue="categoria",
        palette=cores,
        s=70,
        alpha=0.9,
        edgecolor="k"
    )
    plt.title("Relação entre Comprimento e Largura da Sépala", fontsize=13, weight="bold")
    plt.xlabel("Comprimento da Sépala (cm)")
    plt.ylabel("Largura da Sépala (cm)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "relacao_sepala.png"), dpi=300)
    plt.close()
    print("  -> Grafico salvo: relacao_sepala.png")

    # 2. Relacao Comprimento x Largura da Petala
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="comprimento_petala",
        y="largura_petala",
        hue="categoria",
        palette=cores,
        s=70,
        alpha=0.9,
        edgecolor="k"
    )
    plt.title("Relação entre Comprimento e Largura da Pétala", fontsize=13, weight="bold")
    plt.xlabel("Comprimento da Pétala (cm)")
    plt.ylabel("Largura da Pétala (cm)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "relacao_petala.png"), dpi=300)
    plt.close()
    print("  -> Grafico salvo: relacao_petala.png")

    # 3. Distribuicao 4D Completa (Pairplot de todas as dimensoes)
    pair = sns.pairplot(
        df,
        hue="categoria",
        palette=cores,
        diag_kind="kde",
        plot_kws={"alpha": 0.8, "s": 40, "edgecolor": "none"},
        height=2.2
    )
    pair.fig.subplots_adjust(top=0.94)
    pair.fig.suptitle("Distribuição Multidimensional das 4 Variáveis do Iris", fontsize=14, weight="bold")
    pair.savefig(os.path.join(OUTPUT_DIR, "distribuicao_4d_pairplot.png"), dpi=300)
    plt.close()
    print("  -> Grafico salvo: distribuicao_4d_pairplot.png")

    # 4. Boxplots das 4 variaveis por categoria
    df_melt = pd.melt(df, id_vars=["categoria"], value_vars=colunas, var_name="variavel", value_name="valor")
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_melt, x="variavel", y="valor", hue="categoria", palette=cores)
    plt.title("Boxplot Comparativo das 4 Características por Categoria", fontsize=13, weight="bold")
    plt.xlabel("Variável")
    plt.ylabel("Medida (cm)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "distribuicao_boxplots.png"), dpi=300)
    plt.close()
    print("  -> Grafico salvo: distribuicao_boxplots.png")

# ==============================================================================
# ETAPA 4: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ==============================================================================
def aplicar_pca(df, colunas):
    print("\n" + "=" * 78)
    print("ETAPA 3: PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 78)

    X = df[colunas].values
    y = df["categoria"].values

    # Padronizacao (media=0, desvio=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA com todos os 4 componentes para analise de variancia
    pca_completo = PCA(n_components=4)
    pca_completo.fit(X_scaled)

    var_expl = pca_completo.explained_variance_ratio_ * 100
    var_acum = np.cumsum(var_expl)

    print("Variância Explicada por Componente:")
    for i in range(4):
        print(f"  PC{i+1}: {var_expl[i]:.2f}% (Acumulada: {var_acum[i]:.2f}%)")

    # Projecao em 2D
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["categoria"] = y

    # Matriz de pesos (Loadings)
    loadings = pd.DataFrame(
        pca_2d.components_.T,
        columns=["PC1", "PC2"],
        index=colunas
    )
    print("\nPesos dos Componentes Principais (Loadings):")
    print(loadings.round(3))

    # Grafico de dispersao PCA 2D
    cores = {"setosa": "#1f77b4", "versicolor": "#ff7f0e", "virginica": "#2ca02c"}
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_pca,
        x="PC1",
        y="PC2",
        hue="categoria",
        palette=cores,
        s=80,
        alpha=0.9,
        edgecolor="k"
    )
    plt.title(
        f"Projeção PCA 2D do Dataset Iris (Variância Total Explicada: {var_acum[1]:.2f}%)",
        fontsize=13,
        weight="bold"
    )
    plt.xlabel(f"PC1 ({var_expl[0]:.2f}% da variância)")
    plt.ylabel(f"PC2 ({var_expl[1]:.2f}% da variância)")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_2d.png"), dpi=300)
    plt.close()
    print("  -> Grafico salvo: pca_2d.png")

    return X_scaled, X_pca, y, var_expl, loadings

# ==============================================================================
# ETAPA 5: CLASSIFICACAO COM K-NN (80% TREINO / 20% TESTE)
# ==============================================================================
def classificar_knn(X_scaled, X_pca, y):
    print("\n" + "=" * 78)
    print("ETAPA 4: CLASSIFICACAO COM K-NN (DIVISAO 80% TREINO / 20% TESTE)")
    print("=" * 78)

    # Divisao estratificada: 80% treino (120 amostras), 20% teste (30 amostras)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train_pca, X_test_pca, _, _ = train_test_split(
        X_pca, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Tamanho do conjunto de treinamento: {len(X_train)} amostras (80%)")
    print(f"Tamanho do conjunto de teste:        {len(X_test)} amostras (20%)")
    print(f"Distribuicao no teste: {pd.Series(y_test).value_counts().to_dict()}")

    # Avaliar diferentes valores de K
    k_valores = [1, 3, 5, 7, 9, 11]
    resultados = []

    for k in k_valores:
        # K-NN nas 4 dimensoes originais
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred_test = knn.predict(X_test)
        acc_test = accuracy_score(y_test, pred_test)

        # K-NN no espaco PCA 2D
        knn_pca = KNeighborsClassifier(n_neighbors=k)
        knn_pca.fit(X_train_pca, y_train)
        pred_test_pca = knn_pca.predict(X_test_pca)
        acc_pca = accuracy_score(y_test, pred_test_pca)

        resultados.append({
            "k": k,
            "Acuracia (4D Original)": acc_test,
            "Acuracia (PCA 2D)": acc_pca
        })

    df_res = pd.DataFrame(resultados).set_index("k")
    print("\n--- Desempenho do K-NN para diferentes valores de K ---")
    print((df_res * 100).round(2).astype(str) + "%")

    # Modelo Escolhido: K=3 ou K=5 (padrao de alta performance)
    melhor_k = 3
    modelo_final = KNeighborsClassifier(n_neighbors=melhor_k)
    modelo_final.fit(X_train, y_train)
    y_pred = modelo_final.predict(X_test)

    acc_final = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=["setosa", "versicolor", "virginica"])
    cm = confusion_matrix(y_test, y_pred, labels=["setosa", "versicolor", "virginica"])

    print(f"\n--- Relatorio de Classificacao (K={melhor_k}, 4D Original) ---")
    print(f"Acuracia Global: {acc_final * 100:.2f}%\n")
    print(rep)

    # Plotar Matriz de Confusao
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["setosa", "versicolor", "virginica"],
        yticklabels=["setosa", "versicolor", "virginica"],
        cbar=False
    )
    plt.title(f"Matriz de Confusão K-NN (K={melhor_k}) - Acurácia: {acc_final*100:.1f}%", fontsize=12, weight="bold")
    plt.xlabel("Classe Predita")
    plt.ylabel("Classe Real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "matriz_confusao.png"), dpi=300)
    plt.close()
    print("  -> Grafico salvo: matriz_confusao.png")

    # Plotar Fronteira de Decisao do K-NN no espaco PCA 2D
    modelo_pca_final = KNeighborsClassifier(n_neighbors=melhor_k)
    modelo_pca_final.fit(X_train_pca, y_train)

    # Grid de resolucao
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.03), np.arange(y_min, y_max, 0.03))

    mapa_classes = {"setosa": 0, "versicolor": 1, "virginica": 2}
    Z = modelo_pca_final.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([mapa_classes[val] for val in Z]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(["#cfe2ff", "#ffe5d0", "#d1e7dd"])
    cmap_bold = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    
    # Pontos de teste
    for idx, c in enumerate(["setosa", "versicolor", "virginica"]):
        pontos = X_test_pca[y_test == c]
        plt.scatter(pontos[:, 0], pontos[:, 1], c=cmap_bold[idx], label=f"{c} (teste)", edgecolor="k", s=80)

    plt.title(f"Fronteiras de Decisão do K-NN (K={melhor_k}) no Espaço PCA 2D", fontsize=13, weight="bold")
    plt.xlabel("Primeiro Componente Principal (PC1)")
    plt.ylabel("Segundo Componente Principal (PC2)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "knn_fronteira_decisao.png"), dpi=300)
    plt.close()
    print("  -> Grafico salvo: knn_fronteira_decisao.png")

    # Salvar resultados em TXT
    caminho_res = os.path.join(OUTPUT_DIR, "resultados_classificacao.txt")
    with open(caminho_res, "w", encoding="utf-8") as f:
        f.write("RESULTADOS DA CLASSIFICACAO K-NN - DATASET IRIS\n")
        f.write("=" * 60 + "\n\n")
        f.write("COMPARACAO DE ACURACIA POR VALOR DE K:\n")
        f.write(df_res.to_string() + "\n\n")
        f.write(f"MODELO FINAL (K={melhor_k}):\n")
        f.write(f"Acuracia Teste (4D): {acc_final * 100:.2f}%\n\n")
        f.write("RELATORIO DE METRICAS:\n")
        f.write(rep + "\n\n")
        f.write("MATRIZ DE CONFUSAO:\n")
        f.write(str(cm) + "\n")
    print(f"\n[INFO] Resultados de classificacao salvos em: {caminho_res}")

# ==============================================================================
# EXECUCAO PRINCIPAL
# ==============================================================================
def main():
    print("=" * 78)
    print("TRABALHO 4: EXPERIMENTOS DE CLASSIFICACAO NO IRIS DATASET")
    print("=" * 78)

    df, colunas, classes = carregar_dados()
    analisar_estatisticas(df, colunas)
    plotar_relacoes_e_distribuicoes(df, colunas)
    X_scaled, X_pca, y, var_expl, loadings = aplicar_pca(df, colunas)
    classificar_knn(X_scaled, X_pca, y)

    print("\n" + "=" * 78)
    print("EXECUCAO CONCLUIDA COM SUCESSO! Todos os resultados e graficos estao em T4/output/")
    print("=" * 78 + "\n")

if __name__ == "__main__":
    main()
