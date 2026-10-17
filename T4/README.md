# Trabalho 4: Análise Exploratória, PCA e Classificação K-NN no Iris Dataset

Este repositório contém o código e o relatório do Trabalho 4 da disciplina de **Visão Computacional / Aprendizado de Máquina**, focado na análise estatística, redução de dimensionalidade (PCA) e classificação supervisionada com K-NN no clássico dataset [Iris (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/iris).

---

## 🎯 Objetivos do Trabalho

1. **Estatística Descritiva**: Cálculo de média, desvio padrão, moda e frequência das 3 classes (*Setosa*, *Versicolor*, *Virginica*) em geral e por categoria.
2. **Análise de Relações**: Avaliação do comprimento e largura da sépala e da pétala por espécie.
3. **Distribuição Multidimensional**: Visualização das 4 dimensões com pairplot e boxplots.
4. **Principal Component Analysis (PCA)**: Redução de dimensionalidade 4D $\rightarrow$ 2D e avaliação da variância explicada acumulada (~95.8%).
5. **Classificação K-NN**: Divisão estratificada em 80% treino e 20% teste, avaliação com múltiplos valores de $K$, matriz de confusão e fronteiras de decisão.

---

## 🚀 Como Executar

```bash
cd T4
pip install -r requirements.txt

# Executar todas as análises e gerar os gráficos:
python main.py
```

Todos os gráficos gerados e os relatórios numéricos são salvos automaticamente na pasta `output/`.

---

## 📁 Estrutura de Diretórios

```text
T4/
├── data/                             # Cópia local do dataset Iris em CSV
│   └── iris.csv
├── output/                           # Gráficos de alta resolução e relatórios TXT
│   ├── relacao_sepala.png
│   ├── relacao_petala.png
│   ├── distribuicao_4d_pairplot.png
│   ├── distribuicao_boxplots.png
│   ├── pca_2d.png
│   ├── knn_fronteira_decisao.png
│   ├── matriz_confusao.png
│   ├── estatisticas_descritivas.txt
│   └── resultados_classificacao.txt
├── main.py                           # Script Python com todo o pipeline
├── relatorio.tex                     # Relatório sucinto em formato artigo (LaTeX)
├── requirements.txt                  # Bibliotecas requeridas
└── README.md                         # Documentação do projeto
```

---

## 🔗 Repositório Git

- [https://github.com/JPeeeeee/visaoComputacional](https://github.com/JPeeeeee/visaoComputacional)
