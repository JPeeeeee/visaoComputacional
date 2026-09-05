# Relatório: Segmentação de Imagem por Textura usando Banco de Filtros e K-Means

**Disciplina:** Visão Computacional  

---

## 1. Introdução

A segmentação de textura é uma tarefa clássica e fundamental em Visão Computacional, cujo objetivo é dividir uma imagem em regiões consistentes sem se basear apenas nas intensidades (níveis de cinza) ou cores puras, mas sim na variação e no padrão espacial dos pixels vizinhos. 

Neste trabalho, foi proposto a criação de um vetor de características para descrever diferentes padrões texturais (como texturas oriundas de áreas médicas, paisagens, agrícolas, etc.) baseados em um banco de filtros aplicado em múltiplas escalas. O método é estritamente não-supervisionado, valendo-se do uso do algoritmo K-Means.

## 2. Metodologia

O processamento das imagens seguiu o seguinte pipeline:

### 2.1 Escalas
Ao invés de dimensionar os filtros em diversos tamanhos, optou-se por aplicar a mesma configuração de filtros em 3 escalas de imagem distintas. Foram utilizadas Pirâmides Gaussianas onde, a partir da imagem original ($512 \times 512$), realizaram-se duas reduções (para metades e quartos) após a aplicação de desfoque de prevenção de aliasing. 

### 2.2 Banco de Filtros
Para gerar vetores representativos perante textura, foram escolhidos 8 filtros, de forma a capturar orientações direcionais e padrões circulares isotrópicos:
- **7 Filtros Gabor**: Utilizados para cobrir as bordas e estrias ao longo das orientações $0, 22.5, 45, 67.5, 90, 112.5$ e $135$ graus.
- **1 Filtro Circular (LoG/DoG)**: Implementado como Diferença de Gaussianas (Laplacian of Gaussian aproximado), essencial para identificar pequenas formações em "bolha" ou padrão isométrico independentes de orientação direcional.

Essa composição cumpre os requisitos propostos de cobrir orientações de, no mínimo, horizontal, vertical, $45^{\circ}$, $135^{\circ}$ e um padrão circular.

### 2.3 Cálculo do Vetor de Características (24 Dimensões)
Aplicando-se os $8$ filtros nas $3$ escalas, obtêm-se $8 \times 3 = 24$ matrizes de resposta per pixel, o que cumpre precisamente a exigência de um vetor em $24$ dimensões.
O valor cru do filtro indica se o padrão existe, mas a textura caracteriza a região, não apenas o pixel. Por esse motivo, calculou-se a energia absoluta do filtro seguida de uma média em janela deslizante (densamente) de dimensões de $31 \times 31$ pixels. 
Em seguida, as respostas das subescalas sofreram "upsampling" retornando para o grid da imagem original e permitindo criar um tensor dimensional que serviu de representação de cada localização da imagem. 

### 2.4 Agrupamento
Com os vetores computados (padronizados ao possuírem média nula e variância unitária, impedindo que escalas grosseiras dominassem escalas finas), aplicou-se o algoritmo particional *K-Means* valendo-se de métricas da distância euclidiana. Os agrupamentos resultaram em categorizações similares entre os vizinhos, efetivamente segmentando a imagem.

## 3. Resultados

Foi executada a experimentação validando as imagens de entrada do grupo. A visualização de resultados mapeou as diferentes classes geradas pelo K-Means a uma tabela de cores pseudo-aleatórias (Vermelho, Verde, Azul, Amarelo, etc.), renderizadas posteriormente como sobreposição de forma visual transparente em cima da versão nativa em níveis de cinza.
Mesmo de forma inicial, sem inteligência artificial profunda (Deep Learning), percebe-se a eficácia de extração orientada pelas ondas Gabor para texturas listradas ou axadrezadas, aliada com as frequências do Laplaciano para superfícies rugosas, aglomerando homogeneamente a maior parcela de uma mesma formação fotográfica.

## 4. Conclusões

O trabalho forneceu a intuição prática de extração e agrupamento clássico em visão. A abordagem com Pirâmide Gaussiana combinada a janela deslizante gera densidade o bastante para identificar limites das categorias. Possíveis evoluções incluiriam refinamento e otimização dos parâmetros Gabor adaptando as larguras de banda aos tipos exatos de imagem fotografados pelo grupo.

---
**Código Fonte:** [Link para o repositório Git no GitHub](https://github.com/seu-usuario/seu-repositorio) *(Altere aqui com o link final)*
