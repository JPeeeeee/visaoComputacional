# Segmentação de Textura - Visão Computacional

Este projeto realiza a segmentação de regiões de imagens baseando-se em suas texturas. Foi desenvolvido em Python e OpenCV como um trabalho de disciplina de Visão Computacional.

## Requisitos
- Python 3.x
- Opcionalmente um ambiente virtual (`venv`)

Instale as dependências:
```bash
pip install -r requirements.txt
```

## Como usar
1. Coloque as suas imagens recortadas em tamanho 512x512, preferencialmente em tons de cinza, dentro da pasta `dataset/`.
2. Execute o script principal:
```bash
python main.py
```
3. Os resultados segmentados e uma versão sobreposta com a original serão salvos na pasta `output/`.

Se você quiser testar rapidamente sem imagens reais, pode gerar duas imagens sintéticas:
```bash
python gerar_dataset_teste.py
python main.py
```

## Metodologia
- O script usa uma pirâmide Gaussiana (3 escalas) da imagem original.
- Em cada escala, aplica-se um banco de 8 filtros de textura (7 direcionais de Gabor cobrindo as orientações de 0 a 135 graus, mais 1 Laplaciano do Gaussiano para padrão circular).
- Computa-se a magnitude e em seguida uma janela de média. 
- O resultado são 24 dimensões geradas para cada localização de pixel, que são agrupadas utilizando K-Means, separando a imagem nas diferentes texturas encontradas.
