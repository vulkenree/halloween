"""
Core module for person detection and stick figure rendering.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import Optional


class PoseDetection:
    """Handles person pose detection using MediaPipe Pose."""
    
    def __init__(self):
        """Initialize MediaPipe Pose model."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 0=fast, 1=balanced, 2=accurate
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_pose(self, frame: np.ndarray):
        """
        Detect pose landmarks in frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            MediaPipe pose results object with landmarks, or None if no pose detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        return results if results.pose_landmarks else None
    
    def close(self):
        """Clean up resources."""
        self.pose.close()


class StickFigureRenderer:
    """Renders white stick figures on black background from pose landmarks."""
    
    def __init__(self, width: int, height: int):
        """
        Initialize stick figure renderer.
        
        Args:
            width: Output frame width
            height: Output frame height
        """
        self.width = width
        self.height = height
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.white_color = (255, 255, 255)  # White in BGR
    
    def render_stick_figure(self, pose_results: Optional) -> np.ndarray:
        """
        Render stick figure on black background.
        
        Args:
            pose_results: MediaPipe pose results object with landmarks, or None
            
        Returns:
            Frame with black background and white stick figure
        """
        # Create black background
        output_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if pose_results and pose_results.pose_landmarks:
            # Draw stick figure using MediaPipe drawing utilities
            self.mp_drawing.draw_landmarks(
                output_frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=self.white_color,
                    thickness=5,
                    circle_radius=3
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=self.white_color,
                    thickness=5
                )
            )
        
        return output_frame


def process_video(input_path: str, output_path: str, rotate: bool = True) -> None:
    """
    Main video processing pipeline - detects poses and renders stick figures.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save output video
        rotate: If True, rotate output video 90 degrees clockwise (default: True)
    """
    # Initialize components
    pose_detector = PoseDetection()
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate input video length in seconds
    input_video_length = total_frames / fps if fps > 0 else 0
    
    # Initialize stick figure renderer
    renderer = StickFigureRenderer(width, height)
    
    # Determine output dimensions (swap if rotating)
    output_width = height if rotate else width
    output_height = width if rotate else height
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    print(f"Input video length: {input_video_length:.2f} seconds")
    if rotate:
        print(f"Output will be rotated 90° clockwise: {output_width}x{output_height}")
    print("Rendering stick figures on black background...")
    
    # Start timing
    start_time = time.time()
    
    frame_count = 0
    poses_detected = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect pose
        pose_results = pose_detector.detect_pose(frame)
        
        if pose_results:
            poses_detected += 1
        
        # Render stick figure on black background
        output_frame = renderer.render_stick_figure(pose_results)
        
        # Rotate frame if needed
        if rotate:
            output_frame = cv2.rotate(output_frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Write output frame
        out.write(output_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) - Poses detected: {poses_detected}")
    
    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Cleanup
    cap.release()
    out.release()
    pose_detector.close()
    
    # Calculate processing metrics
    fps_processing = total_frames / processing_time if processing_time > 0 else 0
    
    print(f"Processing complete! Output saved to: {output_path}")
    print(f"Total poses detected: {poses_detected} out of {total_frames} frames")
    print(f"Input video length: {input_video_length:.2f} seconds")
    print(f"Time to generate output video: {processing_time:.2f} seconds")
    print(f"Processing speed: {fps_processing:.2f} frames/second")
