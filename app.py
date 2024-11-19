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

def detect_lateral_bounds(positions, reference_positions, prominence=0.01, distance=10):
    """
    Detect lateral bounds using both foot movement and reference point validation
    This looks for side-to-side movement patterns
    """
    # Find peaks for both left and right movement
    right_peaks, right_properties = find_peaks(positions, prominence=prominence, distance=distance)  # Moving right
    left_peaks, left_properties = find_peaks(-positions, prominence=prominence, distance=distance)   # Moving left
    
    # Validate bounds using reference movement (e.g., hip)
    validated_right = []
    validated_left = []
    
    for peak in right_peaks:
        if np.abs(reference_positions[peak] - np.mean(reference_positions)) > prominence * 0.5:
            validated_right.append(peak)
            
    for peak in left_peaks:
        if np.abs(reference_positions[peak] - np.mean(reference_positions)) > prominence * 0.5:
            validated_left.append(peak)
    
    return np.array(validated_right), np.array(validated_left)

def calculate_flight_time(positions, peaks, fps):
    """
    Calculate flight time for lateral bounds by finding takeoff and landing points
    """
    flight_times = []
    takeoff_indices = []
    landing_indices = []
    
    for peak in peaks:
        # Look before peak for takeoff (when movement starts)
        takeoff_idx = peak
        baseline = positions[max(peak-20, 0):peak].mean()  # Get baseline position
        for i in range(peak, max(peak-20, 0), -1):
            if abs(positions[i] - baseline) < 0.01:  # Threshold for takeoff detection
                takeoff_idx = i
                break
                
        # Look after peak for landing (when movement stabilizes)
        landing_idx = peak
        for i in range(peak, min(peak+20, len(positions)-1)):
            if abs(positions[i] - positions[i-1]) < 0.005:  # Threshold for landing detection
                landing_idx = i
                break
        
        flight_time = (landing_idx - takeoff_idx) / fps
        
        flight_times.append(flight_time)
        takeoff_indices.append(takeoff_idx)
        landing_indices.append(landing_idx)
    
    return np.array(flight_times), np.array(takeoff_indices), np.array(landing_indices)

# App Title
st.title("Lateral Bounds Analysis")
st.write("Upload a video to analyze lateral bound jumps.")

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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate video duration in minutes
    video_duration_minutes = total_frames / (fps * 60)

    # Debug information
    st.write("Video properties:")
    st.write(f"Width: {frame_width}, Height: {frame_height}")
    st.write(f"FPS: {fps}, Total Frames: {total_frames}")
    st.write(f"Duration: {video_duration_minutes:.2f} minutes")

    # Temporary directory for frames
    frames_dir = tempfile.mkdtemp()

    # Variables for tracking
    ankle_x_positions = []  # Track x-coordinate instead of y
    hip_x_positions = []    # Reference point
    frame_count = 0

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Track x-coordinates for lateral movement
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Calculate midpoints for tracking
            mid_ankle_x = (left_ankle.x + right_ankle.x) / 2
            mid_hip_x = (left_hip.x + right_hip.x) / 2
            
            ankle_x_positions.append(mid_ankle_x)
            hip_x_positions.append(mid_hip_x)

        # Save frame
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    # Convert positions to numpy arrays
    ankle_x_positions = np.array(ankle_x_positions)
    hip_x_positions = np.array(hip_x_positions)

    try:
        # Detect bounds in both directions
        right_peaks, left_peaks = detect_lateral_bounds(ankle_x_positions, hip_x_positions, prominence=0.02, distance=15)

        # Initialize empty arrays
        right_flight_times = np.array([])
        left_flight_times = np.array([])
        right_takeoffs = np.array([])
        right_landings = np.array([])
        left_takeoffs = np.array([])
        left_landings = np.array([])

        # Calculate flight times if peaks exist
        if len(right_peaks) > 0:
            right_flight_times, right_takeoffs, right_landings = calculate_flight_time(ankle_x_positions, right_peaks, fps)
        if len(left_peaks) > 0:
            left_flight_times, left_takeoffs, left_landings = calculate_flight_time(ankle_x_positions, left_peaks, fps)

        # Debug information
        st.write("Detection Results:")
        st.write(f"Right peaks detected: {len(right_peaks)}")
        st.write(f"Left peaks detected: {len(left_peaks)}")
        
        # Create and display the analysis graphs
        st.write("### Movement Analysis Graphs")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot lateral movement
        ax1.plot(ankle_x_positions, label='Lateral Position', color='blue', alpha=0.7)
        ax1.plot(hip_x_positions, label='Hip Reference', color='green', alpha=0.5)
        
        # Plot bound phases
        if len(right_takeoffs) > 0:
            for takeoff, peak, landing in zip(right_takeoffs, right_peaks, right_landings):
                ax1.axvspan(takeoff, landing, alpha=0.2, color='red', label='Right Bound' if takeoff == right_takeoffs[0] else "")
        if len(left_takeoffs) > 0:
            for takeoff, peak, landing in zip(left_takeoffs, left_peaks, left_landings):
                ax1.axvspan(takeoff, landing, alpha=0.2, color='blue', label='Left Bound' if takeoff == left_takeoffs[0] else "")
        
        if len(right_peaks) > 0:
            ax1.plot(right_peaks, ankle_x_positions[right_peaks], "rx", label="Right Peaks")
        if len(left_peaks) > 0:
            ax1.plot(left_peaks, ankle_x_positions[left_peaks], "bx", label="Left Peaks")
        
        ax1.set_title('Lateral Movement Tracking\n(Shaded areas show bound phases)')
        ax1.set_ylabel('Position (normalized)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot flight times
        ax2.set_title('Flight Times')
        if len(right_flight_times) > 0:
            valid_peaks = right_peaks[:len(right_flight_times)]
            ax2.plot(valid_peaks, right_flight_times, 'ro-', label='Right Flight Time', alpha=0.7)
        if len(left_flight_times) > 0:
            valid_peaks = left_peaks[:len(left_flight_times)]
            ax2.plot(valid_peaks, left_flight_times, 'bo-', label='Left Flight Time', alpha=0.7)
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Flight Time (seconds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

        # Calculate and display statistics
        total_bounds = len(right_peaks) + len(left_peaks)
        bounds_per_minute = total_bounds / video_duration_minutes

        # Display statistics
        st.write("### Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Bound Counts:")
            st.write(f"Total Bounds: {total_bounds}")
            st.write(f"Right Bounds: {len(right_peaks)}")
            st.write(f"Left Bounds: {len(left_peaks)}")
            st.write(f"Bounds per Minute: {bounds_per_minute:.1f}")
        
        with col2:
            st.write("Flight Times:")
            if len(right_flight_times) > 0:
                st.write(f"Right Avg Flight Time: {np.mean(right_flight_times):.3f} seconds")
                st.write(f"Right Max Flight Time: {np.max(right_flight_times):.3f} seconds")
            if len(left_flight_times) > 0:
                st.write(f"Left Avg Flight Time: {np.mean(left_flight_times):.3f} seconds")
                st.write(f"Left Max Flight Time: {np.max(left_flight_times):.3f} seconds")

        # Create frame-by-frame data
        bounds_by_frame = {frame: {
            'count': 0, 
            'flight_time': 0,
            'direction': ''
        } for frame in range(frame_count)}

        # Update running counters
        def get_current_metrics(frame_idx, right_p, left_p, right_times, left_times):
            right_count = len([p for p in right_p if p <= frame_idx])
            left_count = len([p for p in left_p if p <= frame_idx])
            
            if right_count > 0 and frame_idx >= right_p[0]:
                time = right_times[min(right_count - 1, len(right_times) - 1)] if len(right_times) > 0 else 0
                direction = 'right'
            elif left_count > 0 and frame_idx >= left_p[0]:
                time = left_times[min(left_count - 1, len(left_times) - 1)] if len(left_times) > 0 else 0
                direction = 'left'
            else:
                time = 0
                direction = ''
                
            return right_count + left_count, time, direction

        # Update metrics for each frame
        for i in range(frame_count):
            total_count, flight_time, direction = get_current_metrics(
                i, right_peaks, left_peaks, right_flight_times, left_flight_times
            )
            
            bounds_by_frame[i] = {
                'count': total_count,
                'flight_time': flight_time,
                'direction': direction
            }

        # Process frames with overlays
        for i in range(frame_count):
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                current_data = bounds_by_frame[i]
                
                def put_text_with_background(img, text, position):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3
                    color = (0, 255, 0)
                    
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    cv2.rectangle(img, 
                                (position[0] - 10, position[1] - text_height - 10),
                                (position[0] + text_width + 10, position[1] + 10),
                                (0, 0, 0),
                                -1)
                    cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

                # Add text overlays
                put_text_with_background(frame, f'Total Bounds: {current_data["count"]}', (50, 50))
                current_bpm = (current_data["count"] / (i/fps/60)) if i > 0 else 0
                put_text_with_background(frame, f'Bounds/min: {current_bpm:.1f}', (50, 100))
                if current_data["direction"]:
                    direction_text = f'{current_data["direction"].title()} Bound'
                    put_text_with_background(frame, direction_text, (50, 150))
                    put_text_with_background(frame, f'Flight Time: {current_data["flight_time"]:.3f}s', (50, 200))

                cv2.imwrite(frame_path, frame)

        # Use FFmpeg to compile video
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
        st.write("### Processed Video with Visual Overlays")
        st.video(output_video_path)

    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")

    finally:
        # Cleanup
        os.remove(video_path)
        for frame_file in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, frame_file))
        os.rmdir(frames_dir)

else:
    st.warning("Please upload a video.")
