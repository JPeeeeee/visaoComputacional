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

---

## 🚀 Como Executar o Trabalho 3

```bash
cd T3
pip install -r requirements.txt

# Executar suíte de demonstração e gerar figuras para o relatório:
python main.py

# Ou executar no modo interativo com seleção por cliques de mouse:
python main.py --interativo
```
