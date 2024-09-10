import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import maintools as tools
import os
import argparse
import sys


def update_angles(angle_hip, angle_knee):
    # Construa a string formatada
    output = f'Joint Angles | Hip: {round(angle_hip, 2)} - Knee: {round(angle_knee, 2)}'
    sys.stdout.write(f'\r{output}')
    sys.stdout.flush()



def plot_to_image(frame_ids, knee_angles, hip_angles):
    # Criar o gráfico em um buffer de memória
    fig, ax = plt.subplots(figsize=(10, 5))  # Definir o tamanho do gráfico
    
    # Plotar os ângulos com aumento da grossura da linha
    ax.plot(frame_ids, knee_angles, label="Ângulo do Joelho", color="k", linewidth=5)
    ax.plot(frame_ids, hip_angles, label="Ângulo do Quadril", color="red", linewidth=5)

    # Definir o fundo da figura como preto com alpha 0.5
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.5)
    
    # Definir o fundo dos eixos (área do gráfico) como preto
    ax.set_facecolor('white')
    # Definir limites dos eixos
    ax.set_xlim(0, max(frame_ids) + 10)
    ax.set_ylim(0, 180)
    
    # Definir rótulos e aumentar o tamanho da fonte
    ax.set_xlabel("Frame", fontsize=20)
    ax.set_ylabel("Ângulo (graus)", fontsize=20)
    ax.legend(loc='upper right', fontsize=20)  # Aumentar o tamanho da fonte da legenda
    
    # Adicionar título ao gráfico com fonte maior
    ax.set_title("Ângulos do Joelho e Quadril", fontsize=22)

    # Salvar o gráfico em um buffer de memória com fundo transparente
    buf = BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    buf.seek(0)
    
    # Abrir a imagem com PIL e converter para numpy array
    img_pil = Image.open(buf)
    img_cv = np.array(img_pil)
    
    # Fechar o buffer e a figura
    buf.close()
    plt.close(fig)
    
    # Converter a imagem RGB para BGR (OpenCV usa BGR)
    img_bgr = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    return img_bgr



def process_video_graph(video_path, side='d', output_path=None, show=False, save=True):
    print(' ')
    print('Calculating Jointe angles:')
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Carregar o vídeo
    cap = cv2.VideoCapture(video_path)

    # Definir codec e criar VideoWriter para salvar o vídeo com ffmpeg
    if show is True:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Listas para armazenar os ângulos do quadril e joelho
    hip_angles = []
    knee_angles = []
    frame_ids = []

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
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
                    raise ValueError("O valor de 'side' precisa ser 'd' (direito) ou 'e' (esquerdo).")
                
                # Cálculo dos ângulos
                angle_knee = tools.calculate_angle(hip, knee, ankle)  # Ângulo do joelho
                angle_hip = tools.calculate_angle(shoulder, hip, knee)  # Ângulo do quadril
                angle_hip = 180 - angle_hip
                angle_knee = 180 - angle_knee

                # Print joint angles
                update_angles(angle_hip, angle_knee)

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
                                            mp_drawing.DrawingSpec(color=(203, 17, 17), thickness=8, circle_radius=2))
                
            except:
                pass
            
            # Gerar o gráfico e adicioná-lo ao frame
            graph_img = plot_to_image(frame_ids, knee_angles, hip_angles)
            graph_height, graph_width, _ = graph_img.shape

            # Redimensionar o gráfico, se necessário
            scale_factor = 0.3  # Definir o fator de escala
            graph_img = cv2.resize(graph_img, (int(graph_width * scale_factor), int(graph_height * scale_factor)))

            # Inserir o gráfico no canto superior esquerdo do frame de vídeo
            x_offset, y_offset = 10, 10  # Posição do gráfico no frame
            image[y_offset:y_offset + graph_img.shape[0], x_offset:x_offset + graph_img.shape[1]] = graph_img

            # Escrever o frame no vídeo de saída
            if show == True:
                if output_path:
                    out.write(image)
            
            # Exibir o vídeo (opcional)
            if show == True:
                cv2.imshow('Mediapipe Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            frame_id += 1

    if show == True:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Criar um DataFrame com os ângulos do quadril e joelho e os ids dos frames
    df = pd.DataFrame({'Frame': frame_ids, 'Knee_Angle': knee_angles, 'Hip_Angle': hip_angles})

    return df



def run_markerless(video_path, side='d', save=True, show=False):
    print('-----------------------------------')
    print('Calculating Hip and Knee Flexion:')
    print(' ')
    # Implemente a função conforme necessário
    print(f'Video Path: {video_path}')
    print(f'Side: {side}')
    print(f'Save: {save}')
    print(f'Show: {show}')

    # Get the main folder
    main_folder = video_path.split('.')[0]
    
    # Get parent folder
    parent_dir_of_last_folder = os.path.dirname(main_folder)

    # Get the base name without extension
    main_name = os.path.splitext(os.path.basename(video_path))[0]

    # Extract the folder name from the video path
    res_folder = f'{parent_dir_of_last_folder}/results/{main_name}'

    # Ensure that the directory exists
    os.makedirs(res_folder, exist_ok=True)
    
    print(f"Saving results to: {res_folder}")
    
    # Define the output paths
    output_vid = os.path.join(res_folder, f'{main_name}_markerless_{side}.mp4')
    output_fig = os.path.join(res_folder, f'{main_name}_markerless_{side}.jpg')
    output_df = os.path.join(res_folder, f'{main_name}_markerless_{side}.csv')
    
    # Process the video and save results
    df_angles = process_video_graph(video_path, side=side, output_path=output_vid, show=show, save=save)
    print(df_angles)
    if save == True:
        print(output_df) 
        df_angles.to_csv(output_df, index=False)
    
    # Plot and save the angles
    if show is True:
        tools.plot_angle(df_angles, save=False, output=output_fig)
        if save is True:
            tools.plot_angle(df_angles, save=True, output=output_fig)



def str_to_bool(value):
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processa vídeos e calcula ângulos de joelho e quadril.")
    parser.add_argument('--side', type=str, default='d', help="Lado do corpo (d para direito, e para esquerdo)")
    parser.add_argument('--videopath', type=str, required=True, help="Caminho do vídeo")
    parser.add_argument('--save', type=str_to_bool, default=True, help="Salvar em CSV (True ou False)")
    parser.add_argument('--show', type=str_to_bool, default=False, help="Mostrar o vídeo (True ou False)")

    args = parser.parse_args()
    
    run_markerless(video_path=args.videopath, side=args.side, save=args.save, show=args.show)