import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def process_video(video_path, side='d', output_path=None, show=False):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Define codec and create VideoWriter to save the video with ffmpeg
    if output_path: 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Lists to store hip and knee angles
    hip_angles = []
    knee_angles = []
    frame_ids = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                    # Right side hip, knee, and ankle coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                elif side == 'l':
                    # Left side hip, knee, and ankle coordinates
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                else:
                    raise ValueError("The 'side' value must be 'r' (right) or 'l' (left).")
                
                # Angle calculations
                angle_knee = calculate_angle(hip, knee, ankle)  # Knee angle
                angle_hip = calculate_angle(shoulder, hip, knee)  # Hip angle
                angle_hip = 180 - angle_hip
                angle_knee = 180 - angle_knee

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
            
            # Write the frame to the output video
            if output_path:
                out.write(image)
            
            # Display the video (optional)
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

    # Create a DataFrame with hip and knee angles and frame ids
    df = pd.DataFrame({'Frame': frame_ids, 'Knee_Angle': knee_angles, 'Hip_Angle': hip_angles})

    return df



# Function to calculate the angle between 3 points using acos and the dot product
def calculate_angle(a, b, c):
    a = np.array(a)  # Point A
    b = np.array(b)  # Point B (joint)
    c = np.array(c)  # Point C
    
    # Vectors AB and BC
    ba = a - b
    bc = c - b
    
    # Normalize the vectors
    ba_norm = ba / np.linalg.norm(ba)
    bc_norm = bc / np.linalg.norm(bc)
    
    # Dot product between the normalized vectors
    cos_angle = np.dot(ba_norm, bc_norm)
    
    # Ensure the value is within the arccos domain [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate the angle in radians and convert to degrees
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    return angle



def plot_angle(df, save=False, output=None, figsize=(5, 3)):
    # Create a figure and axes
    plt.figure(figsize=figsize)
    
    # Plot knee angles
    plt.plot(df['Frame'], df['Knee_Angle'], label='Knee Angle', color='blue', linewidth=2, marker='o', markersize=3, linestyle='--')
    
    # Plot hip angles
    plt.plot(df['Frame'], df['Hip_Angle'], label='Hip Angle', color='red', linewidth=2, marker='s', markersize=3, linestyle='-')

    # Add title and axis labels
    plt.title('Knee and Hip Angles over Frames', fontsize=10, fontweight='bold')
    plt.xlabel('Frame', fontsize=8)
    plt.ylabel('Angle (degrees)', fontsize=8)
    
    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add a legend
    plt.legend(loc='upper right', fontsize=8)
    
    # Add special annotation at the maximum knee angle point
    max_knee_angle = df['Knee_Angle'].max()
    max_knee_frame = df[df['Knee_Angle'] == max_knee_angle]['Frame'].values[0]
    plt.annotate(f'Max Knee: {round(max_knee_angle, 2)}°',
                    xy=(max_knee_frame, max_knee_angle),
                    xytext=(max_knee_frame + 5, max_knee_angle + 5),
                    arrowprops=dict(facecolor='blue', shrink=0.05, width=0.5, headwidth=5, headlength=5, alpha=0.6),
                    fontsize=8, color='blue')
    
    # Add special annotation at the maximum hip angle point
    max_hip_angle = df['Hip_Angle'].max()
    max_hip_frame = df[df['Hip_Angle'] == max_hip_angle]['Frame'].values[0]
    plt.annotate(f'Max Hip: {round(max_hip_angle, 2)}°',
                xy=(max_hip_frame, max_hip_angle),  # Anchor point for the arrow (numerical values)
                xytext=(max_hip_frame + 5, max_hip_angle + 5),  # Position of the annotation text
                arrowprops=dict(facecolor='red', shrink=0.05, width=0.5, headwidth=5, headlength=5, alpha=0.6),
                fontsize=8, color='red')  # Font and color of the annotation
    
    plt.ylim(0, max([max_knee_angle, max_hip_angle]) + 10)
    
    if save is True:
        if output:
            plt.savefig(output, dpi=300, bbox_inches='tight')
        else: 
            raise ValueError("Please provide an output name.")
        
    # Show the plot
    plt.tight_layout()
