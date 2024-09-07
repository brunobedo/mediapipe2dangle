import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_video(video_path, side='d', output_path=None,show=False):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Carregar o vídeo
    cap = cv2.VideoCapture(video_path)

    # Definir codec e criar VideoWriter para salvar o vídeo com ffmpeg
    if output_path: 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Listas para armazenar os ângulos do quadril e joelho
    hip_angles = []
    knee_angles = []
    frame_ids = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Recolor para RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Processar a imagem para detecção de poses
            results = pose.process(image)
            
            # Recolor de volta para BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extrair landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                if side == 'd': 
                    # Coordenadas do quadril, joelho e tornozelo direito
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                elif side == 'e':
                    # Coordenadas do quadril, joelho e tornozelo esquerdo
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                else:
                    raise ValueError("O valor de 'side' precisa ser 'r' (direito) ou 'l' (esquerdo).")
                
                # Cálculo dos ângulos
                angle_knee = calculate_angle(hip, knee, ankle)  # Ângulo do joelho
                angle_hip = calculate_angle(shoulder, hip, knee)  # Ângulo do quadril
                angle_hip = 180-angle_hip
                angle_knee = 180-angle_knee

                # Armazenar ângulos
                knee_angles.append(angle_knee)
                hip_angles.append(angle_hip)
                frame_ids.append(frame_id)

                # Exibir os ângulos calculados no vídeo
                cv2.putText(image, f'Joelho: {round(angle_knee, 2)}', 
                            tuple(np.multiply(knee, [frame_width, frame_height]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Quadril: {round(angle_hip, 2)}', 
                            tuple(np.multiply(hip, [frame_width, frame_height]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Renderizar landmarks e conexões no vídeo
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=4, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(203,17,17), thickness=8, circle_radius=2))
                
            except:
                pass
            
            # Escrever o frame no vídeo de saída
            if output_path:
                out.write(image)
            
            # Exibir o vídeo (opcional)
            if show is True:
                cv2.imshow('Mediapipe Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            frame_id += 1

    if output_path: 
        cap.release()
        out.release()
    
    if show is True:
        cv2.destroyAllWindows()

    # Criar um DataFrame com os ângulos do quadril e joelho e os ids dos frames
    df = pd.DataFrame({'Frame': frame_ids, 'Knee_Angle': knee_angles, 'Hip_Angle': hip_angles})

    return df



# Função para calcular o ângulo entre 3 pontos utilizando acos e o produto escalar
def calculate_angle(a, b, c):
    a = np.array(a)  # Ponto A
    b = np.array(b)  # Ponto B (articulação)
    c = np.array(c)  # Ponto C
    
    # Vetores AB e BC
    ba = a - b
    bc = c - b
    
    # Normalizar os vetores
    ba_norm = ba / np.linalg.norm(ba)
    bc_norm = bc / np.linalg.norm(bc)
    
    # Produto escalar entre os vetores normalizados
    cos_angle = np.dot(ba_norm, bc_norm)
    
    # Garantir que o valor esteja dentro do domínio de arccos [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calcular o ângulo em radianos e converter para graus
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    return angle



def plot_angle(df,save=False,output=None, figsize=(5, 3)):
    # Criar uma figura e eixos
    plt.figure(figsize=figsize)
    
    # Plotar os ângulos do joelho
    plt.plot(df['Frame'], df['Knee_Angle'], label='Ângulo do Joelho', color='blue', linewidth=2, marker='o', markersize=3, linestyle='--')
    
    # Plotar os ângulos do quadril
    plt.plot(df['Frame'], df['Hip_Angle'], label='Ângulo do Quadril', color='red', linewidth=2, marker='s', markersize=3, linestyle='-')

    # Adicionar título e rótulos aos eixos
    plt.title('Ângulos do Joelho e Quadril ao longo dos frames', fontsize=10, fontweight='bold')
    plt.xlabel('Frame', fontsize=8)
    plt.ylabel('Ângulo (graus)', fontsize=8)
    
    # Adicionar uma grade
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adicionar uma legenda
    plt.legend(loc='upper right', fontsize=8)
    
    # Adicionar anotação especial no ponto máximo do ângulo do joelho
    max_knee_angle = df['Knee_Angle'].max()
    max_knee_frame = df[df['Knee_Angle'] == max_knee_angle]['Frame'].values[0]
    plt.annotate(f'Máximo Joelho: {round(max_knee_angle, 2)}°',
                    xy=(max_knee_frame, max_knee_angle),
                    xytext=(max_knee_frame + 5, max_knee_angle + 5),
                    arrowprops=dict(facecolor='blue', shrink=0.05, width=0.5, headwidth=5, headlength=5, alpha=0.6),
                    fontsize=8, color='blue')
    
    # Adicionar anotação especial no ponto máximo do ângulo do quadril
    max_hip_angle = df['Hip_Angle'].max()
    max_hip_frame = df[df['Hip_Angle'] == max_hip_angle]['Frame'].values[0]
    plt.annotate(f'Máximo Quadril: {round(max_hip_angle, 2)}°',
                xy=(max_hip_frame, max_hip_angle),  # Ponto de ancoragem da seta (valores numéricos)
                xytext=(max_hip_frame + 5, max_hip_angle + 5),  # Posição do texto da anotação
                arrowprops=dict(facecolor='red', shrink=0.05, width=0.5, headwidth=5, headlength=5, alpha=0.6),
                fontsize=8, color='red')  # Fonte e cor da anotação
    
    plt.ylim(0, max([max_knee_angle,max_hip_angle])+10)
    
    if save is True:
        if output:
            plt.savefig(output, dpi=300, bbox_inches='tight')
        else: 
            raise ValueError("Inseira o output name.")
        
    # Mostrar o gráfico
    plt.tight_layout()
    plt.show()
