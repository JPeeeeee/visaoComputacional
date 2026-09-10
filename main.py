import cv2
import numpy as np
import os
import glob

# ==============================================================================
# 1. DEFINIÇÃO DOS FILTROS (MÁSCARAS 3x3)
# ==============================================================================
# Filtros pedidos: horizontal, vertical, 45, 135 e circular, além de extras.
f_horiz = np.array([[-1, -1, -1], [ 0,  0,  0], [ 1,  1,  1]], dtype=np.float32)
f_vert = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
f_45 = np.array([[ 0,  1,  1], [-1,  0,  1], [-1, -1,  0]], dtype=np.float32)
f_135 = np.array([[ 1,  1,  0], [ 1,  0, -1], [ 0, -1, -1]], dtype=np.float32)
f_circ1 = np.array([[ 0, -1,  0], [-1,  4, -1], [ 0, -1,  0]], dtype=np.float32) # Laplaciano (Circular)
f_circ2 = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]], dtype=np.float32)
f_ponto = np.array([[ 1, -2,  1], [-2,  4, -2], [ 1, -2,  1]], dtype=np.float32)
f_rugosa = np.array([[ 1,  0, -1], [ 0,  0,  0], [-1,  0,  1]], dtype=np.float32)

filtros = [f_horiz, f_vert, f_45, f_135, f_circ1, f_circ2, f_ponto, f_rugosa]

# ==============================================================================
# 2. PROCESSAMENTO DAS IMAGENS
# ==============================================================================
os.makedirs("output", exist_ok=True)

# Busca por até 32 ou mais imagens
imagens = glob.glob("dataset/*.png") + glob.glob("dataset/*.jpg") + glob.glob("dataset/*.jpeg")

if len(imagens) == 0:
    print("Nenhuma imagem encontrada na pasta dataset/")
    exit()

tamanho_janela = (15, 15)

# O laço percorre todo o conjunto de imagens que você vai analisar
for caminho_imagem in imagens:
    print(f"Processando: {caminho_imagem}")
    
    # 2.1 Lê a imagem e transforma de RGB para cinza
    img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # 2.2 Garante recortes de 512x512
    img = cv2.resize(img, (512, 512))

    # 2.3 Pirâmide de 3 escalas
    escala1 = img
    escala2 = cv2.resize(escala1, (256, 256))
    escala3 = cv2.resize(escala2, (128, 128))
    lista_escalas = [escala1, escala2, escala3]

    imagens_textura = []

    for imagem_escala in lista_escalas:
        for matriz_filtro in filtros:
            # Aplica máscara (filtro espacial clássico)
            filtrada = cv2.filter2D(imagem_escala, cv2.CV_32F, matriz_filtro)
            
            # Valor absoluto
            filtrada_abs = np.abs(filtrada)
            
            # Média da janela local
            media_janela = cv2.blur(filtrada_abs, tamanho_janela)
            
            # Volta para a dimensão 512x512
            media_512 = cv2.resize(media_janela, (512, 512))
            
            imagens_textura.append(media_512)

    # ==============================================================================
    # 3. CONSTRUÇÃO DO VETOR E K-MEANS (DISTÂNCIA EUCLIDIANA)
    # ==============================================================================
    # Os vetores descritores de textura tem exatamente 24 dimensões
    vetor_24_dimensoes = np.zeros((512 * 512, 24), dtype=np.float32)

    for i in range(24):
        vetor_24_dimensoes[:, i] = imagens_textura[i].flatten()

    # K-Means com distância Euclidiana (padrão)
    numero_de_grupos = 4
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    erro, labels, centros = cv2.kmeans(vetor_24_dimensoes, numero_de_grupos, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)

    mapa_grupos = labels.reshape((512, 512))

    # ==============================================================================
    # 4. CATEGORIZAÇÃO E SALVAMENTO
    # ==============================================================================
    # Mostra a imagem colorida conforme os grupos definidos
    resultado_colorido = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Azul, Verde, Vermelho, Amarelo
    cores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] 

    for grupo in range(numero_de_grupos):
        resultado_colorido[mapa_grupos == grupo] = cores[grupo]

    nome_saida = "seg_" + os.path.basename(caminho_imagem)
    cv2.imwrite(os.path.join("output", nome_saida), resultado_colorido)

print("Processamento concluído para todas as imagens! Verifique a pasta output/")
