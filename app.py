import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import numpy as np
from scipy.signal import find_peaks
import subprocess
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# App Title
st.title("Jump Analysis with Visual Overlays and Jump Classification")
st.write("Upload a video to see pose keypoints, connections, and jump analysis.")

# Video Upload Section
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov"])

if uploaded_file:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.success("Video uploaded successfully!")

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Prepare to save frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Temporary directory for frames
    frames_dir = tempfile.mkdtemp()

    # Variables for jump classification and counting
    hip_y_positions = []
    frame_count = 0

    # Process video frame by frame and collect hip positions
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw pose landmarks and connections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Track Y-coordinate of the left hip
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            hip_y_positions.append(left_hip.y)

        # Save the frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    # Convert hip_y_positions to numpy array
    hip_y_positions = np.array(hip_y_positions)
    
    # Perform jump detection
    peaks, properties = find_peaks(-hip_y_positions, prominence=0.01)
    
    # Create and display the jump detection graph
    st.write("### Jump Detection Graph")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Set threshold and separate big and small jumps
    big_jump_threshold = 0.07
    displacements = properties['prominences']
    big_jump_indices = peaks[displacements >= big_jump_threshold]
    small_jump_indices = peaks[displacements < big_jump_threshold]
    
    # Plot the hip positions and detected jumps
    ax.plot(-hip_y_positions, label='Hip Y-Position', color='blue', alpha=0.7)
    
    # Plot big jumps with red X markers
    if len(big_jump_indices) > 0:
        ax.plot(big_jump_indices, -hip_y_positions[big_jump_indices], "rx", 
                label="Big Jumps", markersize=10, markeredgewidth=2)
    
    # Plot small jumps with yellow circle markers
    if len(small_jump_indices) > 0:
        ax.plot(small_jump_indices, -hip_y_positions[small_jump_indices], "yo", 
                label="Small Jumps (Pogos)", markersize=8, markeredgewidth=2)
    
    # Customize the graph
    ax.set_title('Hip Y-Position with Detected Jumps\n(Higher points represent jumps)')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Hip Height (inverted Y-Coordinate)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Display the graph in Streamlit
    st.pyplot(fig)

    # Create frame-by-frame counters
    jumps_by_frame = {frame: {'big': 0, 'small': 0, 'total': 0} for frame in range(frame_count)}

    # Determine which frames contain jumps and their types
    current_big_jumps = 0
    current_small_jumps = 0
    
    for peak_idx, prominence in zip(peaks, displacements):
        if prominence >= big_jump_threshold:
            current_big_jumps += 1
        else:
            current_small_jumps += 1
        
        # Update all frames after this jump with the new counts
        for frame in range(peak_idx, frame_count):
            jumps_by_frame[frame] = {
                'big': current_big_jumps,
                'small': current_small_jumps,
                'total': current_big_jumps + current_small_jumps
            }

    # Display final counts in Streamlit
    st.write(f"Final Jump Counts:")
    st.write(f"Total Jumps: {current_big_jumps + current_small_jumps}")
    st.write(f"Big Jumps: {current_big_jumps}")
    st.write(f"Small Jumps: {current_small_jumps}")

    # Process saved frames again to add text overlays with running counts
    for i in range(frame_count):
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        frame = cv2.imread(frame_path)
        
        if frame is not None:
            # Get current counts for this frame
            current_counts = jumps_by_frame[i]
            
            # Add text overlays with adjusted positioning and style
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            color = (0, 255, 0)  # Green
            
            def put_text_with_background(img, text, position):
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(img, 
                            (position[0] - 10, position[1] - text_height - 10),
                            (position[0] + text_width + 10, position[1] + 10),
                            (0, 0, 0),
                            -1)
                cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

            # Add text with backgrounds showing running counts
            put_text_with_background(frame, f'Big Jumps: {current_counts["big"]}', (50, 50))
            put_text_with_background(frame, f'Pogos: {current_counts["small"]}', (50, 100))
            put_text_with_background(frame, f'Total Jumps: {current_counts["total"]}', (50, 150))

            # Save the frame with overlays
            cv2.imwrite(frame_path, frame)

    # Use FFmpeg to compile frames into video
    output_video_path = "output_with_overlays.mp4"
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        output_video_path
    ]
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Display the processed video
    st.write("### Processed Video with Visual Overlays and Jump Classification")
    st.video(output_video_path)

    # Cleanup
    os.remove(video_path)
    for frame_file in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, frame_file))
    os.rmdir(frames_dir)

else:
    st.warning("Please upload a video.")
