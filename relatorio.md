# Trabalho 1 - Segmentação de Imagens por Textura

**Disciplina:** Visão Computacional  
**Alunos:** [Seu Nome Aqui] e [Colega]

---

## 1. Introdução
O objetivo deste trabalho é separar (segmentar) os elementos de uma imagem com base na sua textura, agrupando regiões que possuem padrões parecidos (ex: folhas, céu, asfalto). 

Nossa implementação foi feita na linguagem Python com a biblioteca OpenCV, utilizando apenas processamento clássico de matrizes, sem uso de inteligência artificial.

## 2. Construção dos Filtros de Textura
Para extrair a textura da imagem, definimos manualmente **8 máscaras (matrizes 3x3)** de convolução. Cada matriz foi desenhada para destacar um tipo de padrão:
1. **Horizontal**: destaca linhas deitadas.
2. **Vertical**: destaca linhas em pé.
3. **45 Graus**: destaca linhas na diagonal.
4. **135 Graus**: destaca linhas na outra diagonal.
5. **Circular 1 (Laplaciano comum)**: destaca pontos isolados e texturas redondas em formato de cruz.
6. **Circular 2 (Laplaciano com diagonais)**: variação mais forte do filtro circular.
7. **Ponto (Centro)**: destaca ruídos e variações centrais.
8. **Rugosa**: máscara mista para texturas diagonais complexas.

## 3. Escalas e Vetor de 24 Dimensões
A imagem original foi forçada para o tamanho de **512x512 pixels**. 
Depois, criamos **3 escalas** da imagem:
- Escala 1: Original (512x512)
- Escala 2: Metade (256x256)
- Escala 3: Um quarto (128x128)

Em cada uma das 3 escalas, aplicamos as 8 máscaras usando a operação de convolução. Em seguida, pegamos o valor absoluto do resultado e aplicamos um filtro de média (borramento) para espalhar a textura na vizinhança daquela região. 

Como temos 8 filtros aplicados em 3 escalas diferentes, o resultado final gera **24 características para cada pixel**. Ou seja, cada pontinho da imagem original agora é descrito por um vetor de **24 dimensões**.

## 4. Agrupamento (K-Means)
Tendo os vetores de 24 dimensões, organizamos todos os pixels em uma grande tabela e utilizamos o algoritmo **K-Means**. Ele calcula a distância Euclidiana entre os vetores e agrupa os pixels semelhantes em 4 grupos (categorias).

Por fim, cada grupo recebeu uma cor (Azul, Verde, Vermelho e Amarelo), e a imagem foi salva novamente, mostrando as regiões da imagem agrupadas apenas por suas texturas.

## 5. Conclusão
O programa atendeu a todos os requisitos do enunciado. O uso de matrizes simples 3x3 provou-se eficaz para extrair informações espaciais, e as 3 escalas ajudaram a capturar texturas grossas e finas. O K-Means conseguiu separar bem as texturas predominantes na nossa imagem de teste (paisagem).

---
**Link do Git:** [https://github.com/seu-usuario/seu-repositorio](https://github.com/seu-usuario/seu-repositorio)
