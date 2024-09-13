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
    # Build the formatted string
    output = f'Joint Angles | Hip: {round(angle_hip, 2)} - Knee: {round(angle_knee, 2)}'
    sys.stdout.write(f'\r{output}')
    sys.stdout.flush()



def plot_to_image(frame_ids, knee_angles, hip_angles):
    # Create the plot in a memory buffer
    fig, ax = plt.subplots(figsize=(10, 5))  # Set the plot size
    
    # Plot the angles with thicker lines
    ax.plot(frame_ids, knee_angles, label="Knee Angle", color="k", linewidth=5)
    ax.plot(frame_ids, hip_angles, label="Hip Angle", color="red", linewidth=5)

    # Set the figure background to white with alpha 0.5
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0.5)
    
    # Set the axes (plot area) background to white
    ax.set_facecolor('white')
    # Set axis limits
    ax.set_xlim(0, max(frame_ids) + 10)
    ax.set_ylim(0, 180)
    
    # Set labels and increase font size
    ax.set_xlabel("Frame", fontsize=20)
    ax.set_ylabel("Angle (degrees)", fontsize=20)
    ax.legend(loc='upper right', fontsize=20)  # Increase the font size of the legend
    
    # Add title to the plot with a larger font
    ax.set_title("Knee and Hip Angles", fontsize=22)

    # Save the plot in a memory buffer with a transparent background
    buf = BytesIO()
    plt.savefig(buf, format="png", transparent=True)
    buf.seek(0)
    
    # Open the image with PIL and convert it to a numpy array
    img_pil = Image.open(buf)
    img_cv = np.array(img_pil)
    
    # Close the buffer and the figure
    buf.close()
    plt.close(fig)
    
    # Convert the RGB image to BGR (OpenCV uses BGR)
    img_bgr = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    return img_bgr



def process_video_graph(video_path, side='r', min_confidence=0.5, scale_factor=0.5, output_path=None, show=False, save=True):
    print(' ')
    print('Calculating Joint angles:')
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Define codec and create VideoWriter to save the video with ffmpeg
    if show is True:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Lists to store hip and knee angles
    hip_angles = []
    knee_angles = []
    frame_ids = []

    with mp_pose.Pose(min_detection_confidence=min_confidence, min_tracking_confidence=min_confidence) as pose:
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Recolor to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process the image for pose detection
            results = pose.process(image)
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                if side == 'r': 
                    # Right side hip, knee, ankle coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                elif side == 'l':
                    # Left side hip, knee, ankle coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                else:
                    raise ValueError("The 'side' value must be 'r' (right) or 'l' (left).")
                
                # Angle calculations
                angle_knee = tools.calculate_angle(hip, knee, ankle)  # Knee angle
                angle_hip = tools.calculate_angle(shoulder, hip, knee)  # Hip angle
                angle_hip = 180 - angle_hip
                angle_knee = 180 - angle_knee

                # Print joint angles
                update_angles(angle_hip, angle_knee)

                # Store angles
                knee_angles.append(angle_knee)
                hip_angles.append(angle_hip)
                frame_ids.append(frame_id)

                # Display the calculated angles on the video
                cv2.putText(image, f'Knee: {round(angle_knee, 2)}', 
                            tuple(np.multiply(knee, [frame_width, frame_height]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(image, f'Hip: {round(angle_hip, 2)}', 
                            tuple(np.multiply(hip, [frame_width, frame_height]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Render landmarks and connections on the video
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=4, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(203, 17, 17), thickness=8, circle_radius=2))
                
            except:
                pass
            
            # Generate the graph and add it to the video frame
            graph_img = plot_to_image(frame_ids, knee_angles, hip_angles)
            graph_height, graph_width, _ = graph_img.shape

            # Resize the graph if necessary
            scale_factor = scale_factor  # Define the scale factor
            graph_img = cv2.resize(graph_img, (int(graph_width * scale_factor), int(graph_height * scale_factor)))

            # Insert the graph in the top left corner of the video frame
            x_offset, y_offset = 10, 10  # Graph position in the frame
            image[y_offset:y_offset + graph_img.shape[0], x_offset:x_offset + graph_img.shape[1]] = graph_img

            # Write the frame to the output video
            if show == True:
                if output_path:
                    out.write(image)
            
            # Display the video (optional)
            if show == True:
                cv2.imshow('Mediapipe Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            frame_id += 1

    if show == True:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Create a DataFrame with hip and knee angles and frame ids
    df = pd.DataFrame({'Frame': frame_ids, 'Knee_Angle': knee_angles, 'Hip_Angle': hip_angles})

    return df



def run_markerless(video_path, side='r', save=True, show=False, min_confidence=0.5, scale_factor=0.5):
    print('-----------------------------------')
    print('Calculating Hip and Knee Flexion:')
    print(' ')
    # Implement the function as needed
    print(f'Video Path: {video_path}')
    print(f'Side: {side}')
    print(f'Save: {save}')
    print(f'Show: {show}')
    print(f'min_confidence: {min_confidence}')
    print(f'scale_factor: {scale_factor}')

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
    df_angles = process_video_graph(video_path, side=side, output_path=output_vid, show=show, save=save, min_confidence=min_confidence, scale_factor=scale_factor)
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
    parser = argparse.ArgumentParser(description="Processes videos and calculates knee and hip angles.")
    parser.add_argument('--side', type=str, default='r', help="Body side (d for right, e for left)")
    parser.add_argument('--videopath', type=str, required=True, help="Path to the video")
    parser.add_argument('--save', type=str_to_bool, default=True, help="Save as CSV (True or False)")
    parser.add_argument('--show', type=str_to_bool, default=False, help="Show the video (True or False)")
    parser.add_argument('--min_confidence', type=float, default=0.9, help="Value between 0 and 1 for mediapipe tracking")
    parser.add_argument('--scale_factor', type=float, default=0.5, help="Value between 0 and 1 for graph size in the video")

    args = parser.parse_args()
    
    run_markerless(video_path=args.videopath, side=args.side, save=args.save, show=args.show, min_confidence=args.min_confidence, scale_factor=args.scale_factor)