"""
Core module for person detection and stick figure rendering.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List


class PoseDetection:
    """Handles person pose detection using MediaPipe Pose."""
    
    def __init__(self):
        """Initialize MediaPipe Pose model."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=fast, 1=balanced, 2=accurate
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
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
        self.black_color = (0, 0, 0)  # Black in BGR
    
    def get_head_position(self, pose_results) -> Optional[Tuple[int, int, int]]:
        """
        Get head position from pose landmarks.
        
        Args:
            pose_results: MediaPipe pose results object with landmarks
            
        Returns:
            Tuple of (center_x, center_y, radius) or None if not detected
        """
        if not pose_results or not pose_results.pose_landmarks:
            return None
        
        landmarks = pose_results.pose_landmarks.landmark
        h = self.height
        w = self.width
        
        # Get head landmarks (nose is index 0)
        nose = landmarks[0]
        left_eye = landmarks[2]
        right_eye = landmarks[5]
        
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)
        eye_x = int((left_eye.x + right_eye.x) / 2 * w)
        eye_y = int((left_eye.y + right_eye.y) / 2 * h)
        
        head_x = (nose_x + eye_x) // 2
        head_y = (nose_y + eye_y) // 2
        head_radius = int(max(
            abs(nose_x - head_x), abs(nose_y - head_y),
            abs(eye_x - head_x), abs(eye_y - head_y)
        ) * 1.5)
        
        return (head_x, head_y, head_radius)
    
    def draw_spooky_head(self, frame: np.ndarray, head_pos: Tuple[int, int, int]) -> np.ndarray:
        """Draw a spooky ghost head at the given position."""
        x, y, radius = head_pos
        result = frame.copy()
        
        # Draw main oval head
        cv2.ellipse(result, (x, y - radius // 3), (radius, radius), 0, 0, 360, self.white_color, -1)
        
        # Wavy bottom edge
        wave_points = []
        for i in range(-radius, radius + 1, 5):
            wave_y = y + radius // 2 + int(radius * 0.1 * np.sin(i * 0.3))
            wave_points.append((x + i, wave_y))
        
        pts = np.array([(x - radius, y), (x - radius, y + radius // 2)] + 
                      wave_points + 
                      [(x + radius, y + radius // 2), (x + radius, y)], np.int32)
        cv2.fillPoly(result, [pts], self.white_color)
        
        # Draw spooky eyes
        eye_size = radius // 4
        cv2.circle(result, (x - radius // 3, y - radius // 6), eye_size, self.black_color, -1)
        cv2.circle(result, (x + radius // 3, y - radius // 6), eye_size, self.black_color, -1)
        
        # Small mouth
        cv2.ellipse(result, (x, y + radius // 6), (radius // 3, radius // 6), 0, 0, 180, self.black_color, 2)
        
        return result
    
    def render_stick_figure(self, pose_results, spooky_head: bool = False) -> np.ndarray:
        """
        Render stick figure on black background.
        
        Args:
            pose_results: MediaPipe pose results object with landmarks
            
        Returns:
            Frame with black background and white stick figure
        """
        # Create black background
        output_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if pose_results and pose_results.pose_landmarks:
            if spooky_head:
                # Draw spooky head first
                head_pos = self.get_head_position(pose_results)
                if head_pos:
                    output_frame = self.draw_spooky_head(output_frame, head_pos)
                    head_mask = np.ones((self.height, self.width), dtype=np.uint8) * 255
                    head_x, head_y, head_r = head_pos
                    cv2.circle(head_mask, (head_x, head_y), head_r * 2, 0, -1)
                
                # Draw body parts (exclude head region)
                temp_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.mp_drawing.draw_landmarks(
                    temp_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.white_color,
                        thickness=2,
                        circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.white_color,
                        thickness=2
                    )
                )
                
                # Apply mask to exclude head
                if head_pos:
                    head_mask_3ch = cv2.cvtColor(head_mask, cv2.COLOR_GRAY2BGR) / 255.0
                    temp_frame = (temp_frame.astype(np.float32) * head_mask_3ch).astype(np.uint8)
                
                output_frame = cv2.bitwise_or(output_frame, temp_frame)
            else:
                # Normal stick figure rendering
                self.mp_drawing.draw_landmarks(
                    output_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.white_color,
                        thickness=2,
                        circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.white_color,
                        thickness=2
                    )
                )
        
        return output_frame


def process_video(
    input_path: str,
    output_path: str,
    spooky_head: bool = False
) -> None:
    """
    Main video processing pipeline - detects poses and renders stick figures.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save output video
        spooky_head: If True, replace head with spooky ghost head
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
    
    # Initialize stick figure renderer
    renderer = StickFigureRenderer(width, height)
    
    # Initialize video writer with rotated dimensions (90 deg clockwise swaps width/height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rotated_width = height  # After 90 deg clockwise rotation
    rotated_height = width
    out = cv2.VideoWriter(output_path, fourcc, fps, (rotated_width, rotated_height))
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    print(f"Output will be rotated 90° clockwise: {rotated_width}x{rotated_height}")
    print("Rendering stick figures on black background...")
    
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
        output_frame = renderer.render_stick_figure(pose_results, spooky_head=spooky_head)
        
        # Rotate frame 90 degrees clockwise
        rotated_frame = cv2.rotate(output_frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Write output frame
        out.write(rotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) - Poses detected: {poses_detected}")
    
    # Cleanup
    cap.release()
    out.release()
    pose_detector.close()
    
    print(f"Processing complete! Output saved to: {output_path}")
    print(f"Total poses detected: {poses_detected} out of {total_frames} frames")


def process_stick_figure_to_ghost(
    input_path: str,
    output_path: str,
    glow_intensity: int = 15,
    ghost_color: Tuple[int, int, int] = (200, 200, 255),  # Light blue/white in BGR
    fill_transparency: float = 0.3,
    ultra_ghostly: bool = False
) -> None:
    """
    Process stick figure video to create spooky ghost effects.
    
    Args:
        input_path: Path to stick figure video (black background, white stick figures)
        output_path: Path to save ghost effect video
        glow_intensity: Intensity of glow effect (blur radius)
        ghost_color: BGR color for ghost (default: light blue/white)
        fill_transparency: Transparency of filled ghost area (0.0-1.0)
        ultra_ghostly: If True, apply enhanced blurring and atmospheric effects
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")
    
    # Get video properties
    fps_value = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps_value) if fps_value is not None and fps_value > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    total_frames_value = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(total_frames_value) if total_frames_value is not None and total_frames_value > 0 else 0
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing stick figure video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    print("Creating ghost effects...")
    
    frame_count = 0
    
    # Ensure ghost_color is set
    if ghost_color is None:
        ghost_color = (200, 200, 255)  # Default light blue/white in BGR
    
    # Adjust parameters for ultra ghostly mode
    if ultra_ghostly:
        glow_intensity = glow_intensity * 2  # Double the glow
        fill_transparency = fill_transparency * 0.7  # More transparent
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create black background
        ghost_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Extract white stick figure (non-black pixels)
        # Convert to grayscale for mask extraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Create mask for white/non-black pixels
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        if np.any(mask):
            # Create colored ghost version
            ghost_overlay = np.zeros((height, width, 3), dtype=np.uint8)
            ghost_overlay[mask > 0] = ghost_color
            
            # Apply glow effect with multiple blur layers for ultra ghostly mode
            if ultra_ghostly:
                # Multiple blur passes for enhanced ghostly feel
                blurred_mask1 = cv2.GaussianBlur(mask.astype(np.float32), 
                                                (glow_intensity * 2 + 1, glow_intensity * 2 + 1), 
                                                0)
                # Second blur pass for more depth
                blurred_mask2 = cv2.GaussianBlur(blurred_mask1, 
                                                (glow_intensity + 1, glow_intensity + 1), 
                                                0)
                # Third subtle blur for atmospheric effect
                blur_kernel = max(5, glow_intensity // 2) * 2 + 1
                blurred_mask3 = cv2.GaussianBlur(blurred_mask2, 
                                                (blur_kernel, blur_kernel), 
                                                0)
                blurred_mask = blurred_mask3 / 255.0
            else:
                # Single blur pass for normal mode
                blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), 
                                              (glow_intensity * 2 + 1, glow_intensity * 2 + 1), 
                                              0)
                blurred_mask = (blurred_mask / 255.0).astype(np.float32)
            
            # Create glow overlay with ghost color
            glow_overlay = np.zeros((height, width, 3), dtype=np.float32)
            for c in range(3):
                glow_overlay[:, :, c] = blurred_mask * ghost_color[c]
            
            # Add glow to ghost frame (stronger for ultra ghostly)
            glow_strength = 0.8 if ultra_ghostly else 0.6
            ghost_frame = ghost_frame.astype(np.float32) + glow_overlay * glow_strength
            
            # Add semi-transparent filled ghost
            fill_overlay = np.zeros((height, width, 3), dtype=np.float32)
            fill_mask = (mask.astype(np.float32) / 255.0)
            for c in range(3):
                fill_overlay[:, :, c] = fill_mask * ghost_color[c] * fill_transparency
            
            ghost_frame = ghost_frame + fill_overlay
            
            # Add original stick figure outline (white, slightly visible)
            white_outline = np.zeros((height, width, 3), dtype=np.float32)
            white_outline[mask > 0] = [255, 255, 255]
            outline_strength = 0.3 if ultra_ghostly else 0.4  # More faded for ultra ghostly
            ghost_frame = ghost_frame + white_outline * outline_strength
            
            # Additional atmospheric blur for ultra ghostly mode
            if ultra_ghostly:
                ghost_frame_float = ghost_frame.astype(np.float32)
                # Apply subtle overall blur to entire ghost frame for ethereal effect
                for c in range(3):
                    ghost_frame_float[:, :, c] = cv2.GaussianBlur(
                        ghost_frame_float[:, :, c], (5, 5), 0
                    )
                ghost_frame = ghost_frame_float
            
            # Clip to valid range and convert to uint8
            ghost_frame = np.clip(ghost_frame, 0, 255).astype(np.uint8)
        
        # Write output frame
        out.write(ghost_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Ghost processing complete! Output saved to: {output_path}")
