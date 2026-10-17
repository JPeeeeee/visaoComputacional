# Trabalhos Práticos de Visão Computacional

Repositório contendo os trabalhos práticos desenvolvidos na disciplina de **Visão Computacional**.

---

## 📂 Estrutura do Repositório

### 🔹 [Trabalho 1 - Segmentação de Textura](./T1/)
- **Descrição**: Segmentação de imagens e texturas utilizando banco de filtros matemáticos manuais 3x3 em múltiplas escalas e algoritmo K-Means.
- **Tecnologias**: Python, NumPy, OpenCV, Scikit-Learn.

### 🔹 [Trabalho 2 - Calibração de Câmera](./T2/)
- **Descrição**: Estimação da matriz intrínseca $K$, coeficientes de distorção radial/tangencial e projeção 3D $\rightarrow$ 2D de pontos e cubo virtual em realidade aumentada (AR) utilizando tabuleiro de xadrez (*chessboard*).
- **Tecnologias**: Python, OpenCV (`calibrateCamera`, `projectPoints`, `undistort`).

### 🔹 [Trabalho 3 - Transformação Perspectiva e Retificação Ortogonal](./T3/)
- **Descrição**: Sistema de retificação de perspectiva e homografia planar 2D para correção geométrica frontal de obras de arte em museus, murais artísticos e fachadas arquitetônicas.
- **Destaques**: Interface gráfica interativa para seleção de vértices com mouse, ordenação determinística de cantos, estimativa euclidiana de aspecto e artigo científico em formato LaTeX (`relatorio.tex`).
- **Tecnologias**: Python, OpenCV (`getPerspectiveTransform`, `warpPerspective`), NumPy.

### 🔹 [Trabalho 4 - Análise Exploratória, PCA e Classificação K-NN no Iris Dataset](./T4/)
- **Descrição**: Análise estatística descritiva completa, correlação entre dimensões de sépala e pétala, redução de dimensionalidade por PCA (95.8% de variância em 2D) e classificação supervisionada com K-NN (divisão 80/20) atingindo até 96.67% de acurácia.
- **Destaques**: Gráficos de alta resolução de distribuição 4D (pairplot, boxplots), projeção PCA, matriz de confusão e fronteiras de decisão, acompanhado de relatório sucinto em LaTeX.
- **Tecnologias**: Python, Scikit-Learn, Pandas, NumPy, SciPy, Matplotlib, Seaborn.

### 🔹 [Trabalho 5 - Depuração e Otimização de MLP from Scratch](./T5/)
- **Descrição**: Estudo detalhado, depuração de erros matemáticos e otimização de uma Rede Neural Multi-Layer Perceptron (MLP) construída do zero (*from scratch*) em NumPy para o dataset Iris.
- **Destaques**: Diagnóstico de 5 falhas no código original (incluindo inferência sem ativação e erro de atualização de bias/pesos), implementação corrigida com Softmax/Cross-Entropy, normalização Z-score e partição balanceada alcançando 96.67% de acurácia nas 3 classes, além de estudo aprofundado de convergência e overfitting. Acompanha notebook interativo (`.ipynb`) e artigo sucinto em LaTeX.
- **Tecnologias**: Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, Jupyter Notebook.

---

## 🚀 Como Executar os Projetos

### Trabalho 3 (Transformação Perspectiva)
```bash
cd T3
pip install -r requirements.txt
python main.py
```

### Trabalho 4 (Iris Dataset - PCA & K-NN)
```bash
cd T4
pip install -r requirements.txt
python main.py
```

### Trabalho 5 (MLP from Scratch - Depuração & Análise)
```bash
cd T5
pip install -r requirements.txt
python main.py

# Para executar o Jupyter Notebook:
jupyter notebook NN_scratch_iris_resolvido.ipynb
```
