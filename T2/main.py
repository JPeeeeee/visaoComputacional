import cv2
import numpy as np
import os
import glob

# ==============================================================================
# CONFIGURACOES
# ==============================================================================
BOARD_W = 9          # cantos internos na horizontal
BOARD_H = 6          # cantos internos na vertical
SQUARE_SIZE = 25.0   # tamanho do quadrado em mm

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# ETAPA 1: DETECTAR CANTOS DO TABULEIRO
# ==============================================================================
# Pontos 3D do tabuleiro (z=0 porque eh plano)
objp = np.zeros((BOARD_H * BOARD_W, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_SIZE

pontos_3d = []
pontos_2d = []
arquivos_validos = []

imagens = sorted(glob.glob(os.path.join(SCRIPT_DIR, "images", "*.jpg")))
print(f"Procurando cantos em {len(imagens)} imagens...")

if len(imagens) == 0:
    print(f"Erro: Nenhuma imagem encontrada em {os.path.join(SCRIPT_DIR, 'images')}")
    exit(1)

for fpath in imagens:
    img = cv2.imread(fpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    encontrou, cantos = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)

    if encontrou:
        cantos = cv2.cornerSubPix(gray, cantos, (11, 11), (-1, -1), CRITERIA)
        pontos_3d.append(objp)
        pontos_2d.append(cantos)
        arquivos_validos.append(fpath)
        print(f"  OK: {os.path.basename(fpath)}")
    else:
        print(f"  FALHOU: {os.path.basename(fpath)}")

tamanho = gray.shape[::-1]
print(f"Total: {len(arquivos_validos)}/{len(imagens)} imagens validas")

# ==============================================================================
# ETAPA 2: CALIBRAR A CAMERA
# ==============================================================================
print("\nCalibrando...")
erro_rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    pontos_3d, pontos_2d, tamanho, None, None
)

print(f"Erro RMS: {erro_rms:.4f} pixels")
print(f"Matriz K:\n{K}")
print(f"Distorcao: {dist.ravel()}")

# Salvar parametros
with open(os.path.join(OUTPUT_DIR, "parametros.txt"), "w") as f:
    f.write(f"Erro RMS: {erro_rms:.6f}\n\nMatriz K:\n{K}\n\nDistorcao:\n{dist}\n")

# ==============================================================================
# ETAPA 3: SALVAR CANTOS DESENHADOS
# ==============================================================================
img_cantos = cv2.imread(arquivos_validos[0])
cv2.drawChessboardCorners(img_cantos, (BOARD_W, BOARD_H), pontos_2d[0], True)
cv2.imwrite(os.path.join(OUTPUT_DIR, "cantos_detectados.jpg"), img_cantos)
print("\nCantos salvos em output/cantos_detectados.jpg")

# ==============================================================================
# ETAPA 4: REMOVER DISTORCAO
# ==============================================================================
img_original = cv2.imread(arquivos_validos[0])
h, w = img_original.shape[:2]

# Metodo simples
img_corrigida = cv2.undistort(img_original, K, dist, None, K)
cv2.imwrite(os.path.join(OUTPUT_DIR, "original.jpg"), img_original)
cv2.imwrite(os.path.join(OUTPUT_DIR, "sem_distorcao.jpg"), img_corrigida)

# Metodo com recorte
nova_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1)
img_corrigida2 = cv2.undistort(img_original, K, dist, None, nova_K)
x, y, rw, rh = roi
img_recortada = img_corrigida2[y:y+rh, x:x+rw]
cv2.imwrite(os.path.join(OUTPUT_DIR, "sem_distorcao_recortada.jpg"), img_recortada)
print("Imagens sem distorcao salvas em output/")

# ==============================================================================
# ETAPA 5: PROJECAO 3D -> 2D (EXPERIMENTO PRINCIPAL)
# ==============================================================================
print("\nExperimento de projecao 3D -> 2D:")

for i in range(len(arquivos_validos)):
    # Projeta os pontos 3D usando os parametros calibrados
    projetados, _ = cv2.projectPoints(pontos_3d[i], rvecs[i], tvecs[i], K, dist)
    projetados = projetados.reshape(-1, 2)
    detectados = pontos_2d[i].reshape(-1, 2)

    # Calcula erro entre projetado e detectado
    erros = np.linalg.norm(projetados - detectados, axis=1)
    nome = os.path.basename(arquivos_validos[i])
    print(f"  {nome}: erro medio = {erros.mean():.3f} px")

    # Salvar imagem com pontos sobrepostos (primeiras 4)
    if i < 4:
        img = cv2.imread(arquivos_validos[i])
        for p_det, p_proj in zip(detectados, projetados):
            cv2.circle(img, tuple(p_det.astype(int)), 5, (0, 255, 0), -1)
            cv2.circle(img, tuple(p_proj.astype(int)), 5, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"projecao_{i}.jpg"), img)

# ==============================================================================
# ETAPA 6: CUBO VIRTUAL (DEMO AR)
# ==============================================================================
img_cubo = cv2.imread(arquivos_validos[0])
rvec = rvecs[0]
tvec = tvecs[0]

# Vertices do cubo (2 quadrados de lado)
s = SQUARE_SIZE * 2
base = np.float32([[0,0,0], [s,0,0], [s,s,0], [0,s,0]])
topo = np.float32([[0,0,-s], [s,0,-s], [s,s,-s], [0,s,-s]])

base_2d, _ = cv2.projectPoints(base, rvec, tvec, K, dist)
topo_2d, _ = cv2.projectPoints(topo, rvec, tvec, K, dist)
base_2d = base_2d.astype(int).reshape(-1, 2)
topo_2d = topo_2d.astype(int).reshape(-1, 2)

for i in range(4):
    cv2.line(img_cubo, tuple(base_2d[i]), tuple(base_2d[(i+1)%4]), (255,100,0), 3)
    cv2.line(img_cubo, tuple(topo_2d[i]), tuple(topo_2d[(i+1)%4]), (0,220,60), 3)
    cv2.line(img_cubo, tuple(base_2d[i]), tuple(topo_2d[i]), (255,255,255), 2)

cv2.imwrite(os.path.join(OUTPUT_DIR, "cubo_ar.jpg"), img_cubo)
print("\nCubo AR salvo em output/cubo_ar.jpg")

print("\nFim.")
