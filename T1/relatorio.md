# Segmentação de Textura em Imagens Baseada em Filtros Espaciais e K-Means

**Autores:** [Seu Nome Aqui], [Nome do Colega]  
**Disciplina:** Visão Computacional  

---

## Resumo
A segmentação de imagens por textura é um processo fundamental em Visão Computacional. Este artigo descreve a implementação de um método clássico para segmentar 32 imagens (resolução 512x512 em tons de cinza) utilizando matrizes de convolução de 3x3 e o algoritmo de K-Means. A metodologia envolve a aplicação de 8 máscaras em 3 escalas diferentes, gerando um vetor de 24 características (dimensões) para cada região. Os resultados mostram o agrupamento efetivo das texturas por distância Euclidiana, viabilizando a classificação da imagem sem a necessidade de inteligência artificial.

## 1. Introdução
Uma das formas de compreender uma imagem digital é pela análise da sua textura, ou seja, o padrão de variação visual em regiões adjacentes. Ao contrário de uma segmentação puramente focada na cor ou brilho, a segmentação por textura busca unir trechos que apresentem o mesmo padrão espacial, como rugosidade, linhas horizontais ou verticais.

O presente trabalho visa explorar abordagens clássicas, processando matrizes de convolução para construir mapas de textura perante diferentes direções. O conjunto de dados consiste em fotografias autorais processadas para escala de cinza e redimensionadas uniformemente, em cumprimento com as métricas exigidas pela disciplina.

## 2. Metodologia

O processo divide-se em extração de características em múltiplas resoluções e, em seguida, na categorização dos dados gerados.

### 2.1 Banco de Filtros
Para abranger padrões amplos de variações espaciais, foram projetadas e definidas explicitamente oito matrizes (máscaras) $3\times3$ com coeficientes específicos para destacar características fundamentais:
- Máscaras para detecção direcional: Horizontal ($0^\circ$), Vertical ($90^\circ$), Diagonais ($45^\circ$ e $135^\circ$).
- Máscaras estruturais: Dois tipos de variação circular baseadas em aproximações laplacianas e arranjos centro-rodeio (ponto).
- Máscara rugosa: para extrair flutuações diagonais compostas.

Essas operações de convolução substituem de forma mais simplificada algoritmos genéricos complexos (como Gabor), atuando perfeitamente para extrações primárias.

### 2.2 Geração dos Vetores Multiescala
A fim de abranger tanto padrões texturais finos quanto padrões maiores na fotografia, executou-se uma redução em escalas. O processamento aplicou as máscaras na imagem original, e em sucessivas reduções para metade de sua resolução e, posteriormente, em um quarto (3 escalas distintas). 

Após calcular a magnitude bruta do filtro na imagem, cada região precisou ser avaliada pelo seu comportamento local, não por pixels isolados. Aplicou-se um filtro de média de vizinhança na magnitude (borramento), propagando os resultados nas regiões adjacentes. 
Como se aplicaram 8 filtros sobre 3 escalas de imagem, construiu-se para cada localização espacial (após normalização de tamanho) um vetor descritor com exatas **24 dimensões**.

### 2.3 Agrupamento (Clustering)
Com o banco de características montado (matriz na forma de $N \times 24$, onde $N$ equivale aos pixels de $512\times512$), utilizou-se o algoritmo particional **K-Means**. Ele iterou sobre os vetores calculando a distância matemática (Distância Euclidiana $24$-Dimensional), acomodando cada porção da imagem em seu grupo de maior semelhança.
Definiu-se a variável $K = 4$ para agrupar as quatro texturas dominantes de cada imagem processada.

## 3. Resultados
O pipeline demonstrou sucesso quando aplicado sobre as 32 imagens de paisagens propostas, bem como na imagem adicional de validação. O sistema devolveu saídas codificadas em um padrão de cores (Azul, Verde, Vermelho e Amarelo), separando com relativa clareza seções como folhagem, superfícies pavimentadas e variações difusas de céu. Por se tratar de um algoritmo sem IA (puramente estatístico e convolucional), pequenos ruídos internos e transições suaves podem pertencer ao mesmo grupo devido à limitação rígida da distância euclidiana para padrões naturais caóticos.

## 4. Conclusão
Este trabalho atesta a capacidade de técnicas clássicas de visão computacional em promoverem soluções coesas e explicáveis frente a tarefas de categorização. O vetor estruturado em 24 dimensões atendeu perfeitamente ao requerimento teórico, garantindo embasamento robusto na descrição espacial e de múltiplas resoluções, tornando o projeto uma base firme para futuras abordagens supervisionadas.

---
**Anexo de Código Fonte:** [https://github.com/seu-usuario/seu-repositorio](https://github.com/seu-usuario/seu-repositorio)
