# Segmentação de Textura - Visão Computacional

Este projeto foi desenvolvido como primeiro trabalho da disciplina de Visão Computacional. O objetivo é ler 32 fotografias, extrair a textura delas usando filtros matemáticos e agrupar as regiões parecidas usando o algoritmo K-Means.

O programa constrói manualmente 8 filtros (matrizes 3x3) e aplica em 3 escalas de tamanho diferentes da foto. Isso gera um vetor de 24 dimensões que o K-Means usa para separar o céu, as plantas ou o asfalto, apenas com base na textura da imagem. Tudo é feito usando programação clássica, sem inteligência artificial.

## Como rodar o código

1. Instale as dependências necessárias:
```bash
pip install -r requirements.txt
```

2. Coloque as suas imagens na pasta `dataset/`.

3. Execute o programa:
```bash
python main.py
```

O programa vai rodar e as imagens prontas e segmentadas vao para a pasta `output/`.
