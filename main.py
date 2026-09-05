import cv2
import numpy as np
import os
import glob

def build_filters():
    """
    Constrói um banco com 8 filtros (7 direcionais e 1 circular).
    Serão aplicados em 3 escalas para gerar 24 dimensões.
    """
    filters = []
    ksize = 31 # Tamanho do kernel
    
    # 7 Filtros Gabor cobrindo as orientações de 0 a 135 graus
    # (horizontal, vertical, 45, 135 estão incluídos/aproximados aqui)
    orientations = [0, 22.5, 45, 67.5, 90, 112.5, 135]
    for theta in orientations:
        theta_rad = theta * np.pi / 180.0
        # Parâmetros Gabor: ksize, sigma, theta, lambda, gamma, psi, ktype
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta_rad, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        # Normalização do kernel para consistência
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    
    # 1 Filtro circular (LoG - Laplaciano do Gaussiano)
    # Aproximado por Diferença de Gaussianas (DoG)
    g1 = cv2.getGaussianKernel(ksize, 2.0)
    g1 = g1 * g1.T
    g2 = cv2.getGaussianKernel(ksize, 4.0)
    g2 = g2 * g2.T
    log_kern = g1 - g2
    filters.append(log_kern)
    
    return filters

def process_image(img_path, filters, output_dir, k_clusters=4):
    """
    Processa uma imagem, aplicando os filtros nas 3 escalas,
    computando o K-Means nas 24 dimensões geradas, e salva a saída.
    """
    # Carregar imagem já convertendo para escala de cinza
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro ao ler a imagem: {img_path}")
        return
    
    original_h, original_w = img.shape
    
    # Construção da Pirâmide de 3 Escalas
    # Escala 0: Original
    # Escala 1: Metade
    # Escala 2: Um quarto
    scales = [img]
    for i in range(2):
        # Filtro Gaussiano antes de reduzir para evitar aliasing
        blurred = cv2.GaussianBlur(scales[-1], (5, 5), 0)
        # Reduzir à metade em cada dimensão
        downsampled = cv2.resize(blurred, (scales[-1].shape[1] // 2, scales[-1].shape[0] // 2))
        scales.append(downsampled)
        
    features = []
    window_size = 31 # Tamanho da janela para calcular a média local da textura
    
    # Aplicação dos filtros em todas as escalas
    for scale_idx, scale_img in enumerate(scales):
        for filter_idx, kern in enumerate(filters):
            # Convolução
            fimg = cv2.filter2D(scale_img, cv2.CV_32F, kern)
            
            # Extrair a energia da textura (magnitude)
            fimg = np.abs(fimg)
            
            # O vetor descreve cada região pela MÉDIA da janela
            # Aplicar filtro de média local densamente (janela deslizante)
            local_mean = cv2.boxFilter(fimg, cv2.CV_32F, (window_size, window_size))
            
            # Para combinarmos as características, precisamos voltar tudo para o tamanho original
            if scale_idx > 0:
                local_mean = cv2.resize(local_mean, (original_w, original_h))
                
            features.append(local_mean)
            
    # Ao final temos 24 features de tamanho original. Vamos empilhar.
    feature_stack = np.stack(features, axis=2) # Shape: (H, W, 24)
    
    # Reshape para N pixels x 24 dimensões
    Z = feature_stack.reshape((-1, 24))
    Z = np.float32(Z)
    
    # Normalização das features (Importante para K-means com escalas diferentes)
    Z = (Z - np.mean(Z, axis=0)) / (np.std(Z, axis=0) + 1e-8)
    
    # Configuração do K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # Clusterização baseada na distância Euclidiana N-dimensional (24D)
    ret, label, center = cv2.kmeans(Z, k_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Recriar a imagem a partir dos rótulos
    label_img = label.reshape((original_h, original_w))
    
    # Paleta de cores para visualização (BGR)
    colors = [
        [255, 0, 0],   # Azul (Classe 0)
        [0, 255, 0],   # Verde (Classe 1)
        [0, 0, 255],   # Vermelho (Classe 2)
        [0, 255, 255], # Amarelo (Classe 3)
        [255, 0, 255], # Magenta
        [255, 255, 0], # Ciano
        [128, 0, 0],
        [0, 128, 0],
    ]
    
    # Imagem segmentada final
    segmented = np.zeros((original_h, original_w, 3), dtype=np.uint8)
    for i in range(k_clusters):
        segmented[label_img == i] = colors[i % len(colors)]
        
    filename = os.path.basename(img_path)
    
    # Salvar segmentação crua
    cv2.imwrite(os.path.join(output_dir, "seg_" + filename), segmented)
    
    # Salvar sobreposição visual (blend com a original)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_color, 0.6, segmented, 0.4, 0)
    cv2.imwrite(os.path.join(output_dir, "overlay_" + filename), overlay)

def main():
    dataset_dir = "dataset"
    output_dir = "output"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    filters = build_filters()
    print(f"Filtros gerados: {len(filters)}. Aplicados em 3 escalas gerarão {len(filters)*3} dimensões.")
    
    # Busca por imagens na pasta do dataset
    image_paths = glob.glob(os.path.join(dataset_dir, "*.jpg")) + \
                  glob.glob(os.path.join(dataset_dir, "*.png"))
                  
    if not image_paths:
        print(f"Nenhuma imagem encontrada na pasta '{dataset_dir}'.")
        print("Adicione suas imagens em formato quadrado (512x512) ou rode 'python gerar_dataset_teste.py'.")
        return
        
    for img_path in image_paths:
        print(f"Processando {img_path}...")
        # Usando k_clusters=4 como exemplo. Para testar mais grupos, basta alterar aqui.
        process_image(img_path, filters, output_dir, k_clusters=4)
        print(f"Finalizado: {img_path}")

if __name__ == "__main__":
    main()
