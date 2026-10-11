#!/usr/bin/env python3
"""
Trabalho 3 - Visao Computacional: Transformacao Perspectiva
Aplicacao: Retificacao de Perspectiva em Obras de Arte e Fachadas Arquitetonicas

Autores: Joao Pedro Vieira Santos e Grupo
Repositorio Git: https://github.com/JPeeeeee/visaoComputacional
"""

import cv2
import numpy as np
import os
import argparse
import sys

# ==============================================================================
# FUNCOES MATEMATICAS E GEOMETRICAS DE PERSPECTIVA
# ==============================================================================

def ordenar_pontos(pts):
    """
    Ordena 4 coordenadas bidimensionais de forma consistente:
    [Topo-Esquerdo (TL), Topo-Direito (TR), Fundo-Direito (BR), Fundo-Esquerdo (BL)].

    Algoritmo:
    - O ponto superior esquerdo (TL) minimiza (x + y).
    - O ponto inferior direito (BR) maximiza (x + y).
    - O ponto superior direito (TR) minimiza (y - x) [ou maximiza (x - y)].
    - O ponto inferior esquerdo (BL) maximiza (y - x) [ou minimiza (x - y)].
    """
    pts = np.array(pts, dtype=np.float32)
    soma = pts.sum(axis=1)
    diferenca = np.diff(pts, axis=1)

    tl = pts[np.argmin(soma)]
    br = pts[np.argmax(soma)]
    tr = pts[np.argmin(diferenca)]
    bl = pts[np.argmax(diferenca)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def calcular_dimensoes_destino(pts, proporcao_aspecto=None, largura_fixa=None, altura_fixa=None):
    """
    Calcula a largura (W) e altura (H) ideais para a imagem retificada.
    Preserva a escala natural com base nas maiores distancias euclidianas dos lados opostos
    ou aplica uma proporcao pre-definida (ex: 4:3, 16:9).
    """
    tl, tr, br, bl = pts

    if largura_fixa is not None and altura_fixa is not None:
        return int(largura_fixa), int(altura_fixa)

    # Distancias Euclidianas horizontais (largura superior e inferior)
    largura_topo = np.linalg.norm(tr - tl)
    largura_base = np.linalg.norm(br - bl)
    max_largura = max(int(largura_topo), int(largura_base))

    # Distancias Euclidianas verticais (altura esquerda e direita)
    altura_esquerda = np.linalg.norm(bl - tl)
    altura_direita = np.linalg.norm(br - tr)
    max_altura = max(int(altura_esquerda), int(altura_direita))

    # Se uma proporcao for fornecida (ex: 4/3 ou 16/9)
    if proporcao_aspecto is not None and proporcao_aspecto > 0:
        # Ajusta altura baseada na largura estimada
        max_altura = int(max_largura / proporcao_aspecto)

    # Garante dimensoes minimas razoaveis
    max_largura = max(max_largura, 50)
    max_altura = max(max_altura, 50)

    return max_largura, max_altura


def retificar_perspectiva(imagem, pontos_origem, largura_dst=None, altura_dst=None, proporcao=None):
    """
    Aplica a transformacao perspectiva (homografia planar 2D) sobre a imagem.

    Retorna:
    - imagem_retificada: imagem com vista ortogonal frontal
    - H: matriz de homografia 3x3 obtida
    - pts_ordenados: coordenadas de origem ordenadas [TL, TR, BR, BL]
    - pts_destino: coordenadas de destino no novo plano
    """
    pts_ordenados = ordenar_pontos(pontos_origem)
    w_dst, h_dst = calcular_dimensoes_destino(
        pts_ordenados,
        proporcao_aspecto=proporcao,
        largura_fixa=largura_dst,
        altura_fixa=altura_dst
    )

    pts_destino = np.array([
        [0, 0],
        [w_dst - 1, 0],
        [w_dst - 1, h_dst - 1],
        [0, h_dst - 1]
    ], dtype=np.float32)

    # Calcula a matriz de transformacao projetiva 3x3 (Homografia)
    H = cv2.getPerspectiveTransform(pts_ordenados, pts_destino)

    # Aplica o mapeamento inverso com interpolacao bicubica para maxima qualidade visual
    imagem_retificada = cv2.warpPerspective(
        imagem,
        H,
        (w_dst, h_dst),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return imagem_retificada, H, pts_ordenados, pts_destino


def realcar_detalhes(imagem):
    """
    Filtro de realce sutil para pinturas e fachadas (aumento de contraste local
    via CLAHE no canal de luminancia L do espaco LAB).
    """
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_realcado = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_realcado, cv2.COLOR_LAB2BGR)


def desenhar_anotacoes(imagem, pontos):
    """
    Desenha os vertices selecionados, o poligono delimitador e identificadores visuais.
    """
    img_anotada = imagem.copy()
    pts = np.array(pontos, dtype=np.int32)

    # Linhas do poligono (cor ciano vibrante)
    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv2.line(img_anotada, tuple(pts[i]), tuple(pts[i + 1]), (255, 200, 0), 2, cv2.LINE_AA)
    if len(pts) == 4:
        cv2.line(img_anotada, tuple(pts[3]), tuple(pts[0]), (255, 200, 0), 2, cv2.LINE_AA)

        # Regiao semitransparente interna
        overlay = img_anotada.copy()
        cv2.fillPoly(overlay, [pts], (255, 220, 50))
        cv2.addWeighted(overlay, 0.25, img_anotada, 0.75, 0, img_anotada)

    # Desenha os pontos e seus rotulos
    nomes_vertices = ["1 (TL)", "2 (TR)", "3 (BR)", "4 (BL)"]
    cores = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

    for idx, pt in enumerate(pontos):
        cv2.circle(img_anotada, tuple(pt.astype(int)), 7, cores[idx % 4], -1, cv2.LINE_AA)
        cv2.circle(img_anotada, tuple(pt.astype(int)), 9, (255, 255, 255), 2, cv2.LINE_AA)
        label = nomes_vertices[idx] if idx < 4 else f"P{idx+1}"
        cv2.putText(
            img_anotada,
            label,
            (int(pt[0]) + 12, int(pt[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            img_anotada,
            label,
            (int(pt[0]) + 12, int(pt[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return img_anotada


def criar_painel_comparativo(img_original_marcada, img_retificada, titulo="Transformacao Perspectiva"):
    """
    Cria uma figura composta lado a lado (Antes x Depois) com layout elegante para relatorios.
    """
    h_orig, w_orig = img_original_marcada.shape[:2]
    h_ret, w_ret = img_retificada.shape[:2]

    # Redimensiona mantendo a mesma altura base para alinhamento horizontal
    altura_comum = 520
    escala_orig = altura_comum / h_orig
    escala_ret = altura_comum / h_ret

    orig_redim = cv2.resize(img_original_marcada, (int(w_orig * escala_orig), altura_comum))
    ret_redim = cv2.resize(img_retificada, (int(w_ret * escala_ret), altura_comum))

    # Barra divisoria
    divisor = np.full((altura_comum, 10, 3), (60, 60, 60), dtype=np.uint8)

    painel_corpo = np.hstack([orig_redim, divisor, ret_redim])

    # Barra superior de cabecalho
    largura_total = painel_corpo.shape[1]
    cabecalho = np.full((60, largura_total, 3), (35, 35, 35), dtype=np.uint8)

    cv2.putText(
        cabecalho,
        titulo,
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2,
        cv2.LINE_AA
    )

    # Subtitulos colunares
    subtitulos = np.full((35, largura_total, 3), (50, 50, 50), dtype=np.uint8)
    cv2.putText(
        subtitulos,
        "[A] Imagem Original (Perspectiva Obliqua com Vertices)",
        (25, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 220, 255),
        1,
        cv2.LINE_AA
    )
    pos_x_ret = orig_redim.shape[1] + 25
    cv2.putText(
        subtitulos,
        "[B] Imagem Retificada (Plano Frontal Ortogonal)",
        (pos_x_ret, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 255, 180),
        1,
        cv2.LINE_AA
    )

    painel_final = np.vstack([cabecalho, subtitulos, painel_corpo])
    return painel_final


# ==============================================================================
# MODO INTERATIVO (INTERFACE GRAFICA OPENCV)
# ==============================================================================

class SeletorInterativo:
    def __init__(self, caminho_imagem):
        self.caminho = caminho_imagem
        self.imagem_original = cv2.imread(caminho_imagem)
        if self.imagem_original is None:
            raise FileNotFoundError(f"Nao foi possivel carregar a imagem: {caminho_imagem}")

        self.pontos = []
        self.nome_janela = "Seletor de Perspectiva (Clique 4 vertices) - Pressione [r]=Reset [c]=Retificar [q]=Sair"

    def callback_mouse(self, evento, x, y, flags, param):
        if evento == cv2.EVENT_LBUTTONDOWN:
            if len(self.pontos) < 4:
                self.pontos.append(np.array([x, y], dtype=np.float32))
                self.atualizar_visualizacao()
                if len(self.pontos) == 4:
                    print("\n[INFO] 4 pontos selecionados com sucesso!")
                    print("Pressione 'c' ou ENTER para executar a retificacao.")
                    print("Pressione 'r' para limpar e selecionar novamente.")

    def atualizar_visualizacao(self):
        img_temp = self.imagem_original.copy()
        if len(self.pontos) > 0:
            img_temp = desenhar_anotacoes(img_temp, self.pontos)
        cv2.imshow(self.nome_janela, img_temp)

    def executar(self):
        cv2.namedWindow(self.nome_janela, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.nome_janela, self.callback_mouse)
        self.atualizar_visualizacao()

        print("\n" + "=" * 70)
        print("MODO INTERATIVO ATIVADO")
        print("Instrucoes:")
        print(" 1. Clique nos 4 cantos do quadro/fachada (em qualquer ordem ou sentido horario).")
        print(" 2. Pressione 'c' ou [ENTER] para retificar.")
        print(" 3. Pressione 'r' para resetar a selecao.")
        print(" 4. Pressione 'q' ou [ESC] para sair.")
        print("=" * 70 + "\n")

        while True:
            tecla = cv2.waitKey(20) & 0xFF
            if tecla in [ord('q'), 27]: # q ou ESC
                break
            elif tecla == ord('r'):
                self.pontos = []
                self.atualizar_visualizacao()
                print("[INFO] Selecao resetada. Clique novamente nos 4 cantos.")
            elif tecla in [ord('c'), 13]: # c ou ENTER
                if len(self.pontos) == 4:
                    retificada, H, pts_ord, _ = retificar_perspectiva(self.imagem_original, self.pontos)
                    anotada = desenhar_anotacoes(self.imagem_original, pts_ord)
                    painel = criar_painel_comparativo(anotada, retificada, "Resultado Interativo")

                    janela_res = "Resultado da Retificacao de Perspectiva (pressione 's' p/ salvar)"
                    cv2.namedWindow(janela_res, cv2.WINDOW_NORMAL)
                    cv2.imshow(janela_res, painel)

                    print("\n[RESULTADO] Matriz de Homografia H (3x3):")
                    print(np.round(H, 5))
                    print(f"Dimensoes retificadas: {retificada.shape[1]}x{retificada.shape[0]} px")
                else:
                    print(f"[AVISO] Selecione exatamente 4 cantos antes de retificar (atuais: {len(self.pontos)}).")
            elif tecla == ord('s'):
                if 'retificada' in locals():
                    os.makedirs("output", exist_ok=True)
                    cv2.imwrite("output/resultado_interativo_retificado.jpg", retificada)
                    cv2.imwrite("output/resultado_interativo_painel.jpg", painel)
                    print("[INFO] Imagens salvas com sucesso em output/!")

        cv2.destroyAllWindows()


# ==============================================================================
# MODO DE DEMONSTRACAO / EXPERIMENTOS AUTOMATIZADOS (BATCH)
# ==============================================================================

def executar_experimentos_batch(diretorio_base):
    """
    Executa os testes automatizados nos 3 cenarios representativos:
    1. Obra de Arte em Galeria de Museu (Obliqua)
    2. Mural Artistico Urbano (Oxford Graffiti VGG)
    3. Fachada Arquitetonica de Edificio Historico
    """
    pasta_img = os.path.join(diretorio_base, "images")
    pasta_out = os.path.join(diretorio_base, "output")
    os.makedirs(pasta_out, exist_ok=True)

    cenarios = [
        {
            "id": "cenario1_museu",
            "titulo": "Quadro a Oleo em Galeria de Museu",
            "arquivo": "museu_quadro_perspectiva.jpg",
            "pontos": np.float32([
                [220, 180],  # TL
                [920, 110],  # TR
                [940, 630],  # BR
                [210, 520]   # BL
            ]),
            "proporcao": 4.0 / 3.0,
            "descricao": "Quadro com moldura nobre fotografado a partir de angulo lateral de ~35 graus."
        },
        {
            "id": "cenario2_grafite",
            "titulo": "Mural Artistico Urbano (Oxford VGG Graffiti)",
            "arquivo": "graf3.png",
            "pontos": np.float32([
                [185, 105],  # TL
                [675, 168],  # TR
                [648, 555],  # BR
                [135, 465]   # BL
            ]),
            "proporcao": None, # Dimensoes euclidianas automaticas
            "descricao": "Pintura mural em parede de tijolos sob forte angulo de fuga horizontal."
        },
        {
            "id": "cenario3_fachada",
            "titulo": "Fachada Arquitetonica de Edificio",
            "arquivo": "building.jpg",
            "pontos": np.float32([
                [165, 260],  # TL
                [470, 192],  # TR
                [485, 452],  # BR
                [175, 510]   # BL
            ]),
            "proporcao": None,
            "descricao": "Fachada predial com janelas e colunas submetida a convergencia de retas paralelas."
        }
    ]

    print("\n" + "=" * 78)
    print("INICIANDO EXPERIMENTOS AUTOMATIZADOS DE TRANSFORMACAO PERSPECTIVA")
    print("=" * 78)

    relatorio_parametros = []

    for idx, c in enumerate(cenarios, 1):
        caminho_imagem = os.path.join(pasta_img, c["arquivo"])
        if not os.path.exists(caminho_imagem):
            print(f"[ALERTA] Imagem nao encontrada: {caminho_imagem}. Pulando...")
            continue

        img = cv2.imread(caminho_imagem)
        print(f"\n--- Cenário {idx}: {c['titulo']} ---")
        print(f"Imagem de Entrada: {c['arquivo']} ({img.shape[1]}x{img.shape[0]} px)")
        print(f"Descricao: {c['descricao']}")

        # Executa a retificacao
        retificada, H, pts_ord, pts_dst = retificar_perspectiva(
            img,
            c["pontos"],
            proporcao=c["proporcao"]
        )

        # Versao com realce sutil
        retificada_realcada = realcar_detalhes(retificada)

        # Imagem com os vertices e o quadrilatero delimitador desenhados
        anotada = desenhar_anotacoes(img, pts_ord)

        # Painel comparativo
        painel = criar_painel_comparativo(
            anotada,
            retificada,
            f"Cenario {idx}: {c['titulo']}"
        )

        # Salva resultados
        caminho_orig = os.path.join(pasta_out, f"{c['id']}_original.jpg")
        caminho_marcado = os.path.join(pasta_out, f"{c['id']}_marcado.jpg")
        caminho_retif = os.path.join(pasta_out, f"{c['id']}_retificado.jpg")
        caminho_realcado = os.path.join(pasta_out, f"{c['id']}_realcado.jpg")
        caminho_comp = os.path.join(pasta_out, f"{c['id']}_comparacao.jpg")

        cv2.imwrite(caminho_orig, img)
        cv2.imwrite(caminho_marcado, anotada)
        cv2.imwrite(caminho_retif, retificada)
        cv2.imwrite(caminho_realcado, retificada_realcada)
        cv2.imwrite(caminho_comp, painel)

        w_ret, h_ret = retificada.shape[1], retificada.shape[0]
        print(f"Dimensoes Retificadas: {w_ret}x{h_ret} px")
        print("Matriz de Homografia H (3x3):")
        for linha in H:
            print("  [" + "  ".join(f"{val:12.6f}" for val in linha) + " ]")

        # Registro para relatorio
        relatorio_parametros.append({
            "cenario": c["titulo"],
            "id": c["id"],
            "w_orig": img.shape[1],
            "h_orig": img.shape[0],
            "w_ret": w_ret,
            "h_ret": h_ret,
            "pts_orig": pts_ord,
            "H": H
        })

    # Salva arquivo de parametros tecnicos
    caminho_txt = os.path.join(pasta_out, "parametros_experimentos.txt")
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write("RELATORIO DOS PARAMETROS DE HOMOGRAFIA - TRABALHO 3\n")
        f.write("=" * 60 + "\n\n")
        for item in relatorio_parametros:
            f.write(f"Cenario: {item['cenario']} ({item['id']})\n")
            f.write(f"Dimensoes Originais: {item['w_orig']}x{item['h_orig']}\n")
            f.write(f"Dimensoes Retificadas: {item['w_ret']}x{item['h_ret']}\n")
            f.write("Pontos de Origem (TL, TR, BR, BL):\n")
            f.write(str(item["pts_orig"]) + "\n")
            f.write("Matriz de Homografia H:\n")
            f.write(str(item["H"]) + "\n\n")

    print("\n" + "=" * 78)
    print(f"TODOS OS EXPERIMENTOS CONCLUIDOS! Resultados salvos em: {pasta_out}")
    print("=" * 78 + "\n")


# ==============================================================================
# PONTO DE ENTRADA PRINCIPAL (CLI)
# ==============================================================================

def main():
    diretorio_script = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Retificacao de Perspectiva em Obras de Arte e Fachadas com OpenCV"
    )
    parser.add_argument(
        "--interativo",
        action="store_true",
        help="Executa em modo interativo com interface de selecao por cliques de mouse."
    )
    parser.add_argument(
        "--imagem",
        type=str,
        default=None,
        help="Caminho para uma imagem especifica a ser processada no modo interativo."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Executa o processamento em lote em todos os cenarios de demonstracao."
    )

    args = parser.parse_args()

    # Se chamado com --interativo
    if args.interativo:
        caminho = args.imagem
        if caminho is None:
            caminho = os.path.join(diretorio_script, "images", "museu_quadro_perspectiva.jpg")
        if not os.path.exists(caminho):
            print(f"Erro: Arquivo '{caminho}' nao encontrado.")
            sys.exit(1)
        seletor = SeletorInterativo(caminho)
        seletor.executar()
    else:
        # Modo padrao: executa a suite de demonstracao e gera todas as figuras
        executar_experimentos_batch(diretorio_script)


if __name__ == "__main__":
    main()
