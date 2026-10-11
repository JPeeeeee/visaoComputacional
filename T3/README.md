# Trabalho 3: Retificação de Perspectiva em Obras de Arte e Fachadas Arquitetônicas

Este repositório contém a implementação do Trabalho 3 da disciplina de **Visão Computacional**. O foco é a formulação matemática e a aplicação computacional de **Transformações Perspectivas (Homografia Planar 2D)** utilizando a biblioteca **OpenCV** e **Python**.

---

## 🎯 Aplicação Desenvolvida

A aplicação foi concebida para **correção e retificação ortogonal de planos inclinados** no espaço tridimensional capturados sob perspectiva oblíqua:
1. **Obras de Arte e Quadros em Galerias de Arte / Museus**: restauração frontal de quadros fotografados lateralmente devido a restrições espaciais ou reflexos de luz.
2. **Murais e Grafites Urbanos**: retificação de murais artísticos em paredes com ângulo de fuga acentuado (benchmark *Oxford VGG Graffiti*).
3. **Fachadas Arquitetônicas**: recuperação do paralelismo ortogonal de fachadas prediais com convergência vertical e horizontal de retas.

---

## 📐 Fundamentação Teórica

A relação entre um plano tridimensional no mundo real e sua projeção na imagem é descrita por uma homografia planar projetiva $H \in \mathbb{R}^{3 \times 3}$:

$$
s \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

Como a matriz possui 8 graus de liberdade (com fator de escala arbitrário $h_{33} = 1$), 4 correspondências de pontos não colineares $(x_i, y_i) \leftrightarrow (x'_i, y'_i)$ são estritamente necessárias e suficientes para determinar $H$ de maneira única via algoritmo DLT (*Direct Linear Transformation*), implementado internamente pela função `cv2.getPerspectiveTransform`. A interpolação de mapeamento inverso de pixels é realizada via `cv2.warpPerspective` com interpolação bicúbica (`cv2.INTER_CUBIC`).

---

## 🚀 Como Executar

### 1. Instalar as dependências
```bash
pip install -r requirements.txt
```

### 2. Modo Demonstração / Batch (Gera todas as figuras e métricas automaticamente)
```bash
python main.py
# ou
python main.py --demo
```
Os resultados visuais (imagens originais, com vértices marcados, retificadas e painéis lado a lado) serão salvos na pasta `output/`.

### 3. Modo Interativo (Seleção de Pontos via Interface Gráfica)
Permite abrir qualquer imagem e selecionar interativamente os 4 cantos do plano a ser retificado:
```bash
python main.py --interativo --imagem images/museu_quadro_perspectiva.jpg
```
- **Botão Esquerdo do Mouse**: Clique nos 4 cantos (em qualquer ordem ou sentido horário).
- **Tecla `c` ou `ENTER`**: Computa a homografia e exibe a imagem frontal retificada.
- **Tecla `r`**: Reseta os pontos para selecionar novamente.
- **Tecla `s`**: Salva a imagem retificada e o painel em `output/`.
- **Tecla `q` ou `ESC`**: Fecha a janela.

---

## 📁 Estrutura de Diretórios

```text
T3/
├── images/                   # Imagens de entrada dos cenários
│   ├── museu_quadro_perspectiva.jpg
│   ├── graf3.png
│   └── building.jpg
├── output/                   # Imagens geradas e relatórios numéricos
│   ├── cenario1_museu_comparacao.jpg
│   ├── cenario2_grafite_comparacao.jpg
│   ├── cenario3_fachada_comparacao.jpg
│   └── parametros_experimentos.txt
├── main.py                   # Código principal com modos interativo e batch
├── relatorio.tex             # Artigo científico completo em LaTeX
├── requirements.txt          # Dependências do projeto
└── README.md                 # Documentação detalhada
```

---

## 🔗 Repositório Git

- **Link oficial do projeto**: [https://github.com/JPeeeeee/visaoComputacional](https://github.com/JPeeeeee/visaoComputacional)
