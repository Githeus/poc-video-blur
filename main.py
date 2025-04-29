# Importa a biblioteca OpenCV para processamento de vídeo e imagem
import cv2
# Importa a biblioteca os (não usada neste script, pode ser removida)
import os
# Importa defaultdict, que facilita agrupar itens por chave sem inicialização manual
from collections import defaultdict

# Caminho para o vídeo de entrada que será processado
video_path = "mdc3.mp4"

# Lista simulando a resposta da AWS Rekognition com timestamps (em milissegundos) e coordenadas de rostos detectados
# Cada item é uma tupla: (timestamp, bounding box do rosto)
# As coordenadas estão em proporção (0 a 1), relativas ao tamanho do frame
raw_timestamps = [
    (0, {
        "Width": 0.11710939556360245,
        "Height": 0.11229322105646133,
        "Left": 0.4043998420238495,
        "Top": 0.027280054986476898
    }),
    (500, {
        "Width": 0.11710798740386963,
        "Height": 0.11406604200601578,
        "Left": 0.4143161475658417,
        "Top": 0.03844483569264412
    }),
    (500, {
        "Width": 0.01255035400390625,
        "Height": 0.006393551826477051,
        "Left": 0.06901542097330093,
        "Top": 0.4143158495426178
    }),
    (1000, {
        "Width": 0.012294260784983635,
        "Height": 0.005983471870422363,
        "Left": 0.06913621723651886,
        "Top": 0.4135735034942627
    }),
    (1500, {
        "Width": 0.11675584316253662,
        "Height": 0.11592300981283188,
        "Left": 0.38936853408813477,
        "Top": 0.03752869367599487
    })
]

# Organiza as detecções em um dicionário onde cada timestamp agrupa uma lista de rostos detectados
final_timestamps = defaultdict(list)
for timestamp, bbox in raw_timestamps:
    final_timestamps[timestamp].append(bbox)

# Função que aplica um efeito de pixelização (anonimização) em uma região da imagem
def anonymize_face_pixelate(image, blocks=10):
    # Obtém altura (h) e largura (w) da imagem
    (h, w) = image.shape[:2]
    
    # Define o tamanho de cada bloco de pixelização
    xSteps = max(1, w // blocks)
    ySteps = max(1, h // blocks)

    # Percorre a imagem em blocos e aplica média de cor para criar efeito pixelado
    for y in range(0, h, ySteps):
        for x in range(0, w, xSteps):
            endX = min(x + xSteps, w)
            endY = min(y + ySteps, h)
            roi = image[y:endY, x:endX]  # Região do bloco
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]  # Média das cores no bloco
            image[y:endY, x:endX] = (B, G, R)  # Aplica a média no bloco

    return image

# Função principal que processa o vídeo, aplica a anonimização e gera um novo arquivo
def process_video(video_path, final_timestamps, output_path='mdc3-blur.mp4', delta_seconds=0.5, blocks=10):
    # Abre o vídeo de entrada
    v = cv2.VideoCapture(video_path)
    if not v.isOpened():
        raise IOError(f"Erro ao abrir vídeo: {video_path}")

    # Obtém propriedades do vídeo: dimensões, taxa de quadros (FPS), número total de quadros
    frame_width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = v.get(cv2.CAP_PROP_FPS)
    total_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))

    # Margem extra aplicada nas caixas para garantir cobertura completa
    width_delta = int(0.05 * frame_width)
    height_delta = int(0.05 * frame_height)

    # Define o codec e inicializa o vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    # Dicionário que mapeia frames específicos para as detecções de rosto a serem pixeladas
    frames_with_faces = {}

    # Para cada timestamp com rostos detectados...
    for t, faces in final_timestamps.items():
        # Converte timestamp de milissegundos para segundos
        timestamp_sec = int(t) / 1000

        # Define intervalo de frames a considerar antes e depois do timestamp
        lower_bound = int((timestamp_sec - delta_seconds) * frame_rate)
        upper_bound = int((timestamp_sec + delta_seconds) * frame_rate)

        # Associa os rostos detectados aos frames dentro do intervalo
        for i in range(lower_bound, upper_bound + 1):
            if i < 0 or i >= total_frames:
                continue
            if i not in frames_with_faces:
                frames_with_faces[i] = []
            frames_with_faces[i].extend(faces)

    # Contador de quadros atual
    frame_counter = 0

    # Itera sobre os quadros do vídeo
    while v.isOpened():
        has_frame, frame = v.read()
        if not has_frame:
            break  # Fim do vídeo

        # Se este quadro tiver rostos a serem anonimados...
        if frame_counter in frames_with_faces:
            for f in frames_with_faces[frame_counter]:
                # Converte coordenadas proporcionais para coordenadas absolutas de pixel
                x = int(f['Left'] * frame_width) - width_delta
                y = int(f['Top'] * frame_height) - height_delta
                w = int(f['Width'] * frame_width) + 2 * width_delta
                h = int(f['Height'] * frame_height) + 2 * height_delta

                # Garante que os limites da área não ultrapassem o frame
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame_width, x1 + w)
                y2 = min(frame_height, y1 + h)

                # Recorta a região do rosto, aplica pixelização e substitui na imagem original
                to_blur = frame[y1:y2, x1:x2]
                blurred = anonymize_face_pixelate(to_blur, blocks=blocks)
                frame[y1:y2, x1:x2] = blurred

        # Escreve o frame (modificado ou não) no vídeo de saída
        out.write(frame)
        frame_counter += 1

    # Libera os arquivos de vídeo
    v.release()
    out.release()
    print(f"✅ Vídeo processado salvo em: {output_path}")

# Chamada da função principal com os dados simulados
process_video(video_path, final_timestamps)