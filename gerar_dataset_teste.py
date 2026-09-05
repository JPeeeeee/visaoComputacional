import cv2
import numpy as np
import os

def gerar_textura(tipo, size=(512, 512)):
    img = np.zeros(size, dtype=np.uint8)
    if tipo == 'horizontal':
        for i in range(0, size[0], 10):
            img[i:i+5, :] = 255
    elif tipo == 'vertical':
        for i in range(0, size[1], 10):
            img[:, i:i+5] = 255
    elif tipo == 'ruido':
        img = np.random.randint(0, 256, size, dtype=np.uint8)
    elif tipo == 'xadrez':
        block = 16
        for i in range(0, size[0], block):
            for j in range(0, size[1], block):
                if (i // block + j // block) % 2 == 0:
                    img[i:i+block, j:j+block] = 255
    
    # Adicionar um pouco de ruído para ficar mais natural
    noise = np.random.normal(0, 15, size).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img_noisy

def gerar_imagem_composta(size=(512, 512)):
    # Criar uma imagem dividida em 4 quadrantes com diferentes texturas
    img = np.zeros(size, dtype=np.uint8)
    h, w = size
    h2, w2 = h // 2, w // 2
    
    img[0:h2, 0:w2] = gerar_textura('horizontal', (h2, w2))
    img[0:h2, w2:w] = gerar_textura('vertical', (h2, w2))
    img[h2:h, 0:w2] = gerar_textura('ruido', (h2, w2))
    img[h2:h, w2:w] = gerar_textura('xadrez', (h2, w2))
    
    return img

if __name__ == "__main__":
    out_dir = "dataset"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Gerando imagens de teste...")
    img1 = gerar_imagem_composta()
    img2 = gerar_textura('xadrez')
    
    cv2.imwrite(os.path.join(out_dir, "teste_01.png"), img1)
    cv2.imwrite(os.path.join(out_dir, "teste_02.png"), img2)
    print(f"Imagens salvas em {out_dir}/")
