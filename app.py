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

def calculate_bound_time(positions, peaks, fps):
    """
    Calculate time between takeoff and landing for lateral bounds
    """
    bound_times = []
    takeoff_indices = []
    landing_indices = []
    
    for peak in peaks:
        # Look before peak for takeoff
        takeoff_idx = peak
        for i in range(peak, max(peak-20, 0), -1):
            if abs(positions[i] - positions[i-1]) > 0.01:  # Threshold for lateral movement
                takeoff_idx = i
                break
                
        # Look after peak for landing
        landing_idx = peak
        for i in range(peak, min(peak+20, len(positions)-1)):
            if abs(positions[i] - positions[i-1]) < 0.005:  # Threshold for stabilization
                landing_idx = i
                break
        
        bound_time = (landing_idx - takeoff_idx) / fps
        
        bound_times.append(bound_time)
        takeoff_indices.append(takeoff_idx)
        landing_indices.append(landing_idx)
    
    return np.array(bound_times), np.array(takeoff_indices), np.array(landing_indices)

# App Title
st.title("Lateral Bounds Analysis")
st.write("Upload a video to analyze lateral bound jumps.")

# Video Upload Section
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov"])

# Optional height input for distance calculation
person_height = st.number_input("Enter your height in inches (default: 68 inches = 5'8\")", 
                              min_value=48, 
                              max_value=84, 
                              value=68)

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

        # Initialize empty arrays for times and distances
        right_bound_times = np.array([])
        left_bound_times = np.array([])
        right_distances_inches = np.array([])
        left_distances_inches = np.array([])
        right_takeoffs = np.array([])
        right_landings = np.array([])
        left_takeoffs = np.array([])
        left_landings = np.array([])

        # Calculate bound times if peaks exist
        if len(right_peaks) > 0:
            right_bound_times, right_takeoffs, right_landings = calculate_bound_time(ankle_x_positions, right_peaks, fps)
        if len(left_peaks) > 0:
            left_bound_times, left_takeoffs, left_landings = calculate_bound_time(ankle_x_positions, left_peaks, fps)
        
        # Calculate distances if we have at least 2 peaks
        pixels_per_inch = (frame_width * 0.3) / person_height
        if len(right_peaks) > 1:
            # Calculate distances between consecutive peaks
            right_distances = np.abs(np.diff(ankle_x_positions[right_peaks])) * frame_width
            right_distances_inches = right_distances / pixels_per_inch
            # Now right_distances_inches will have length len(right_peaks) - 1
        else:
            right_distances_inches = np.array([])
        
        if len(left_peaks) > 1:
            # Calculate distances between consecutive peaks
            left_distances = np.abs(np.diff(ankle_x_positions[left_peaks])) * frame_width
            left_distances_inches = left_distances / pixels_per_inch
            # Now left_distances_inches will have length len(left_peaks) - 1
        else:
            left_distances_inches = np.array([])
            
        # Debug information
        st.write("Detection Results:")
        st.write(f"Right peaks detected: {len(right_peaks)}")
        st.write(f"Left peaks detected: {len(left_peaks)}")
        
        # Create and display the analysis graphs
        st.write("### Movement Analysis Graphs")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
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

        # Plot bound times
        ax2.set_title('Bound Times')
        if len(right_bound_times) > 0:
            # Only plot if we have peaks and matching times
            valid_peaks = right_peaks[:len(right_bound_times)]  # Match lengths
            ax2.plot(valid_peaks, right_bound_times, 'ro-', label='Right Bound Time', alpha=0.7)
        if len(left_bound_times) > 0:
            # Only plot if we have peaks and matching times
            valid_peaks = left_peaks[:len(left_bound_times)]  # Match lengths
            ax2.plot(valid_peaks, left_bound_times, 'bo-', label='Left Bound Time', alpha=0.7)
        ax2.set_ylabel('Time (seconds)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot bound distances
        ax3.set_title('Bound Distances')
        if len(right_distances_inches) > 0:
            # Only plot if we have peaks and matching distances
            valid_peaks = right_peaks[:len(right_distances_inches)]  # Match lengths
            ax3.plot(valid_peaks, right_distances_inches, 'ro-', label='Right Distance', alpha=0.7)
        if len(left_distances_inches) > 0:
            # Only plot if we have peaks and matching distances
            valid_peaks = left_peaks[:len(left_distances_inches)]  # Match lengths
            ax3.plot(valid_peaks, left_distances_inches, 'bo-', label='Left Distance', alpha=0.7)
        ax3.set_xlabel('Frame Index')
        ax3.set_ylabel('Distance (inches)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
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
            st.write("Bound Metrics:")
            if len(right_bound_times) > 0:
                st.write(f"Right Avg Time: {np.mean(right_bound_times):.3f} seconds")
                if len(right_distances_inches) > 0:
                    st.write(f"Right Avg Distance: {np.mean(right_distances_inches):.1f} inches")
            if len(left_bound_times) > 0:
                st.write(f"Left Avg Time: {np.mean(left_bound_times):.3f} seconds")
                if len(left_distances_inches) > 0:
                    st.write(f"Left Avg Distance: {np.mean(left_distances_inches):.1f} inches")

        # Create frame-by-frame data
        bounds_by_frame = {frame: {
            'count': 0, 
            'current_bound_time': 0,
            'current_distance': 0,
            'direction': ''
        } for frame in range(frame_count)}

        # Update running counters
        def get_current_metrics(frame_idx, right_p, left_p, right_times, left_times, right_dist, left_dist):
            right_count = len([p for p in right_p if p <= frame_idx])
            left_count = len([p for p in left_p if p <= frame_idx])
            
            if right_count > 0 and frame_idx >= right_p[0]:
                time = right_times[min(right_count - 1, len(right_times) - 1)] if len(right_times) > 0 else 0
                dist = right_dist[min(right_count - 1, len(right_dist) - 1)] if len(right_dist) > 0 else 0
                direction = 'right'
            elif left_count > 0 and frame_idx >= left_p[0]:
                time = left_times[min(left_count - 1, len(left_times) - 1)] if len(left_times) > 0 else 0
                dist = left_dist[min(left_count - 1, len(left_dist) - 1)] if len(left_dist) > 0 else 0
                direction = 'left'
            else:
                time = 0
                dist = 0
                direction = ''
                
            return right_count + left_count, time, dist, direction

        # Update metrics for each frame
        for i in range(frame_count):
            total_count, bound_time, distance, direction = get_current_metrics(
                i, right_peaks, left_peaks, right_bound_times, left_bound_times,
                right_distances_inches, left_distances_inches
            )
            
            bounds_by_frame[i] = {
                'count': total_count,
                'current_bound_time': bound_time,
                'current_distance': distance,
                'direction': direction
            }

        # Process frames with overlays
        for i in range(frame_count):
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
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
                    put_text_with_background(frame, f'Time: {current_data["current_bound_time"]:.3f}s', (50, 200))
                    put_text_with_background(frame, f'Distance: {current_data["current_distance"]:.1f}"', (50, 250))

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

