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
                        thickness=5,
                        circle_radius=3
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.white_color,
                        thickness=5
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
                        thickness=5,
                        circle_radius=3
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=self.white_color,
                        thickness=5
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


def load_witch_kernel(image_path: str, kernel_size: int = 25) -> Optional[np.ndarray]:
    """
    Load witch image as grayscale and prepare as convolution kernel.
    
    Args:
        image_path: Path to witch image file
        kernel_size: Size of the convolution kernel (will be resized to this)
        
    Returns:
        Normalized convolution kernel or None if loading fails
    """
    # Load as grayscale
    witch_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if witch_img is None:
        return None
    
    # Resize to kernel size
    witch_kernel = cv2.resize(witch_img, (kernel_size, kernel_size))
    
    # Normalize to avoid brightness blowout
    kernel_sum = witch_kernel.sum()
    if kernel_sum > 0:
        witch_kernel = witch_kernel.astype(np.float32) / kernel_sum
    else:
        witch_kernel = witch_kernel.astype(np.float32) / (kernel_size * kernel_size)
    
    return witch_kernel


def load_witch_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load witch image and extract it from background.
    
    Args:
        image_path: Path to witch image file
        
    Returns:
        Tuple of (witch_image, mask) or (None, None) if loading fails
    """
    witch_img = cv2.imread(image_path)
    if witch_img is None:
        return None, None
    
    # Convert to grayscale for thresholding
    gray = cv2.cvtColor(witch_img, cv2.COLOR_BGR2GRAY)
    
    # Try multiple methods to extract witch from background
    # Method 1: Assume white/light background
    _, mask_white = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Method 2: Use color-based segmentation (common backgrounds: white, light gray)
    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(witch_img, cv2.COLOR_BGR2HSV)
    
    # Create mask for non-white/light backgrounds
    # Lower and upper bounds for white/light colors in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_color = cv2.inRange(hsv, lower_white, upper_white)
    mask_color = cv2.bitwise_not(mask_color)  # Invert to get witch pixels
    
    # Combine masks (use the better one)
    # Use the mask that has reasonable amount of foreground
    mask_white_area = np.sum(mask_white > 0)
    mask_color_area = np.sum(mask_color > 0)
    
    if mask_color_area > mask_white_area * 0.8:
        mask = mask_color
    else:
        mask = mask_white
    
    # Clean up mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to image (create RGBA-like effect)
    witch_extracted = cv2.bitwise_and(witch_img, witch_img, mask=mask)
    
    return witch_extracted, mask


def estimate_person_height(pose_results, frame_height: int, frame_width: int) -> Tuple[Optional[float], Optional[str]]:
    """
    Estimate person height from pose landmarks and classify as kid or adult.
    
    Args:
        pose_results: MediaPipe pose results object with landmarks
        frame_height: Height of the video frame
        frame_width: Width of the video frame
        
    Returns:
        Tuple of (height_ratio, classification) where:
        - height_ratio: Normalized height (0.0-1.0) or None if not detectable
        - classification: "kid" or "adult" or None
    """
    if not pose_results or not pose_results.pose_landmarks:
        return None, None
    
    landmarks = pose_results.pose_landmarks.landmark
    
    # Get head position (use nose as top reference, or estimate from eyes)
    nose = landmarks[0]
    left_ankle = landmarks[27]  # Left ankle
    right_ankle = landmarks[28]  # Right ankle
    
    # Calculate head top (estimate slightly above nose)
    head_top_y = nose.y * frame_height - frame_height * 0.05
    
    # Calculate feet position (use lower ankle)
    feet_y = max(left_ankle.y, right_ankle.y) * frame_height
    
    # Calculate height in pixels
    person_height = feet_y - head_top_y
    
    # Normalize by frame height
    height_ratio = person_height / frame_height if frame_height > 0 else 0
    
    # Classify: threshold at 40% of frame height
    classification = "kid" if height_ratio < 0.4 else "adult"
    
    return height_ratio, classification


def apply_witch_image(
    frame: np.ndarray,
    witch_image: np.ndarray,
    witch_mask: np.ndarray,
    head_pos: Tuple[int, int, int],
    scale: float = 1.0
) -> np.ndarray:
    """
    Apply witch image overlay to frame at head position.
    
    Args:
        frame: Frame to draw on
        witch_image: Witch image (BGR, already extracted from background)
        witch_mask: Mask for witch image (non-background pixels)
        head_pos: Tuple of (center_x, center_y, radius)
        scale: Scale factor for sizing (based on person height)
        
    Returns:
        Frame with witch image overlaid
    """
    result = frame.copy()
    x, y, base_radius = head_pos
    
    # Calculate target size for witch image
    target_size = int(base_radius * scale * 3)  # Make it larger than head to cover more area
    
    # Ensure minimum size
    target_size = max(50, target_size)
    
    # Get original witch image dimensions
    orig_h, orig_w = witch_image.shape[:2]
    
    # Calculate scaling factor
    scale_factor = target_size / max(orig_h, orig_w)
    new_w = int(orig_w * scale_factor)
    new_h = int(orig_h * scale_factor)
    
    # Resize witch image and mask
    witch_resized = cv2.resize(witch_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(witch_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate position (center witch image at head position)
    start_x = x - new_w // 2
    start_y = y - new_h // 2
    end_x = start_x + new_w
    end_y = start_y + new_h
    
    # Ensure coordinates are within frame bounds
    frame_h, frame_w = result.shape[:2]
    
    # Calculate crop if needed
    crop_x_start = 0
    crop_y_start = 0
    crop_x_end = new_w
    crop_y_end = new_h
    
    if start_x < 0:
        crop_x_start = -start_x
        start_x = 0
    if start_y < 0:
        crop_y_start = -start_y
        start_y = 0
    if end_x > frame_w:
        crop_x_end = new_w - (end_x - frame_w)
        end_x = frame_w
    if end_y > frame_h:
        crop_y_end = new_h - (end_y - frame_h)
        end_y = frame_h
    
    # Crop witch image and mask if needed
    witch_cropped = witch_resized[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    mask_cropped = mask_resized[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    
    # Only overlay if we have valid dimensions
    if witch_cropped.shape[0] > 0 and witch_cropped.shape[1] > 0:
        # Convert mask to 3-channel for blending
        mask_3ch = cv2.cvtColor(mask_cropped, cv2.COLOR_GRAY2BGR)
        mask_normalized = (mask_3ch.astype(np.float32) / 255.0)
        
        # Extract ROI from result frame
        roi = result[start_y:end_y, start_x:end_x]
        
        # Blend witch image with frame using mask
        witch_float = witch_cropped.astype(np.float32)
        roi_float = roi.astype(np.float32)
        
        # Overlay with alpha blending
        blended = roi_float * (1 - mask_normalized) + witch_float * mask_normalized
        
        # Put blended result back
        result[start_y:end_y, start_x:end_x] = np.clip(blended, 0, 255).astype(np.uint8)
    
    return result


def draw_witch_head(frame: np.ndarray, head_pos: Tuple[int, int, int], scale: float = 1.0) -> np.ndarray:
    """
    Draw a witch head with pointed hat at the given position.
    
    Args:
        frame: Frame to draw on
        head_pos: Tuple of (center_x, center_y, radius)
        scale: Scale factor for sizing (based on person height)
        
    Returns:
        Frame with witch head drawn
    """
    x, y, base_radius = head_pos
    result = frame.copy()
    
    # Scale the radius based on person size
    radius = int(base_radius * scale)
    
    # Draw face (circle/ellipse)
    cv2.ellipse(result, (x, y), (radius, int(radius * 1.1)), 0, 0, 360, (255, 255, 255), -1)
    
    # Draw pointed witch hat
    hat_height = max(10, int(radius * 1.5))
    hat_base_width = max(10, int(radius * 1.8))
    hat_brim_height = max(2, radius // 4)
    
    # Hat base (brim)
    cv2.ellipse(result, (x, y - radius // 3), (hat_base_width, hat_brim_height), 0, 0, 360, (100, 50, 0), -1)  # Brown
    
    # Hat point (triangle/cone)
    hat_top_y = y - radius // 3 - hat_height
    hat_points = np.array([
        [x, hat_top_y],  # Top point
        [x - hat_base_width, y - radius // 3],  # Left base
        [x + hat_base_width, y - radius // 3]   # Right base
    ], np.int32)
    cv2.fillPoly(result, [hat_points], (100, 50, 0))  # Brown hat
    
    # Draw eyes
    eye_size = max(3, radius // 6)
    eye_offset_x = radius // 3
    cv2.circle(result, (x - eye_offset_x, y - radius // 8), eye_size, (0, 0, 0), -1)  # Black eyes
    cv2.circle(result, (x + eye_offset_x, y - radius // 8), eye_size, (0, 0, 0), -1)
    
    # Draw mouth (small curve)
    mouth_y = y + radius // 4
    mouth_w = max(5, radius // 3)
    mouth_h = max(3, radius // 6)
    cv2.ellipse(result, (x, mouth_y), (mouth_w, mouth_h), 0, 0, 180, (0, 0, 0), 2)
    
    return result


def draw_skateboard(frame: np.ndarray, feet_x: int, feet_y: int, person_height_ratio: float, frame_width: int) -> np.ndarray:
    """
    Draw a skateboard at the person's feet.
    
    Args:
        frame: Frame to draw on
        feet_x: X position of feet (center)
        feet_y: Y position of feet
        person_height_ratio: Normalized person height for sizing
        frame_width: Frame width for proportional sizing
        
    Returns:
        Frame with skateboard drawn
    """
    result = frame.copy()
    
    # Size skateboard based on person height
    board_length = max(20, int(frame_width * 0.15 * person_height_ratio))
    board_width = max(5, int(board_length * 0.3))
    
    # Position skateboard horizontally at feet level
    board_x = feet_x
    board_y = feet_y + board_width // 2
    
    # Ensure board_y is within frame bounds
    board_y = min(max(board_y, board_width), result.shape[0] - board_width)
    
    # Draw skateboard (horizontal oval/rectangle)
    ellipse_w = max(1, board_length // 2)
    ellipse_h = max(1, board_width // 2)
    cv2.ellipse(result, (board_x, board_y), (ellipse_w, ellipse_h), 0, 0, 360, (100, 100, 100), -1)  # Gray
    cv2.ellipse(result, (board_x, board_y), (ellipse_w, ellipse_h), 0, 0, 360, (255, 255, 255), 2)  # White outline
    
    # Draw wheels (small circles)
    wheel_size = max(5, board_width // 3)
    wheel_offset = board_length // 3
    cv2.circle(result, (board_x - wheel_offset, board_y), wheel_size, (50, 50, 50), -1)  # Dark gray wheels
    cv2.circle(result, (board_x + wheel_offset, board_y), wheel_size, (50, 50, 50), -1)
    
    return result


def apply_witch_convolution(
    stick_figure: np.ndarray,
    witch_kernel: np.ndarray
) -> np.ndarray:
    """
    Apply witch kernel convolution to stick figure.
    
    Args:
        stick_figure: Stick figure frame (black background, white stick figure)
        witch_kernel: Normalized witch convolution kernel
        
    Returns:
        Convolved frame with witch texture applied
    """
    # Convert stick figure to grayscale if needed
    if len(stick_figure.shape) == 3:
        stick_gray = cv2.cvtColor(stick_figure, cv2.COLOR_BGR2GRAY)
    else:
        stick_gray = stick_figure.copy()
    
    # Apply convolution
    convolved = cv2.filter2D(stick_gray, -1, witch_kernel)
    
    # Convert back to BGR if needed
    if len(stick_figure.shape) == 3:
        convolved_bgr = cv2.cvtColor(convolved, cv2.COLOR_GRAY2BGR)
        return convolved_bgr
    else:
        return convolved


def get_scaled_kernel_size(base_size: int, height_ratio: Optional[float]) -> int:
    """
    Calculate kernel size based on person height.
    
    Args:
        base_size: Base kernel size (e.g., 25)
        height_ratio: Normalized person height ratio or None
        
    Returns:
        Scaled kernel size (always odd)
    """
    if height_ratio is None:
        height_ratio = 0.5  # Default
    
    # Scale based on height: kernel_size = base * (1 + height_ratio * 2)
    kernel_size = int(base_size * (1 + height_ratio * 2))
    
    # Ensure kernel size is odd (required for convolution)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Ensure minimum size
    kernel_size = max(5, kernel_size)
    
    return kernel_size


def draw_broom(frame: np.ndarray, person_x: int, person_y_top: int, person_y_bottom: int, person_height_ratio: float) -> np.ndarray:
    """
    Draw a broomstick next to the person.
    
    Args:
        frame: Frame to draw on
        person_x: X position of person (center)
        person_y_top: Y position of person top (head)
        person_y_bottom: Y position of person bottom (feet)
        person_height_ratio: Normalized person height for sizing
        
    Returns:
        Frame with broom drawn
    """
    result = frame.copy()
    
    # Size broom based on person height
    broom_length = int((person_y_bottom - person_y_top) * 1.2)
    broom_width = max(3, int(broom_length * 0.08))
    
    # Position broom to the right of person
    broom_x = person_x + int(broom_width * 3)
    broom_top_y = person_y_top
    
    # Draw broom handle (vertical line/rectangle)
    cv2.rectangle(result, 
                 (broom_x - broom_width // 2, broom_top_y),
                 (broom_x + broom_width // 2, broom_top_y + broom_length),
                 (101, 67, 33), -1)  # Brown handle
    cv2.rectangle(result,
                 (broom_x - broom_width // 2, broom_top_y),
                 (broom_x + broom_width // 2, broom_top_y + broom_length),
                 (255, 255, 255), 2)  # White outline
    
    # Draw broom bristles at bottom (fan shape)
    bristle_width = broom_width * 4
    bristle_height = int(broom_length * 0.2)
    bristle_bottom_y = broom_top_y + broom_length
    
    # Bristles as multiple lines or triangle
    bristle_points = np.array([
        [broom_x, bristle_bottom_y],  # Bottom center
        [broom_x - bristle_width // 2, bristle_bottom_y - bristle_height],  # Left
        [broom_x + bristle_width // 2, bristle_bottom_y - bristle_height]   # Right
    ], np.int32)
    cv2.fillPoly(result, [bristle_points], (139, 69, 19))  # Brown bristles
    
    return result


def process_ghost_video(input_path: str, output_path: str) -> None:
    """
    Process input video directly to ghost output (detect pose → stick figure → ghost).
    
    Args:
        input_path: Path to input video file
        output_path: Path to save ghost output video
    """
    # Initialize components
    pose_detector = PoseDetection()
    
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
    
    # Initialize renderer and ghost parameters
    renderer = StickFigureRenderer(width, height)
    ghost_color = (200, 200, 255)  # Light blue/white in BGR
    glow_intensity = 15
    fill_transparency = 0.3
    
    # Initialize video writer (rotated 90° clockwise)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rotated_width = height
    rotated_height = width
    out = cv2.VideoWriter(output_path, fourcc, fps, (rotated_width, rotated_height))
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    print("Creating ghost effects...")
    
    frame_count = 0
    poses_detected = 0
    previous_stick_frame = None  # Frame buffer for trailing effect
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect pose
        pose_results = pose_detector.detect_pose(frame)
        
        if pose_results:
            poses_detected += 1
        
        # Render stick figure on black background
        stick_frame = renderer.render_stick_figure(pose_results, spooky_head=False)
        
        # Rotate stick figure frame (for output orientation)
        stick_frame_rotated = cv2.rotate(stick_frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Create trailing stick figure from previous frame (80% scale, blurred, glowing)
        trailing_frame = None
        if previous_stick_frame is not None:
            # Scale previous frame's stick figure to 80%
            prev_rotated = cv2.rotate(previous_stick_frame, cv2.ROTATE_90_CLOCKWISE)
            h, w = prev_rotated.shape[:2]
            new_h = int(h * 0.8)
            new_w = int(w * 0.8)
            
            # Resize to 80%
            trailing_scaled = cv2.resize(prev_rotated, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Center it on the frame (or position it appropriately)
            trailing_frame = np.zeros((rotated_height, rotated_width, 3), dtype=np.uint8)
            y_offset = (rotated_height - new_h) // 2
            x_offset = (rotated_width - new_w) // 2
            trailing_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = trailing_scaled
        
        # Update frame buffer for next iteration
        previous_stick_frame = stick_frame.copy()
        
        # Create ghost frame (start with trailing figure if available)
        ghost_frame = np.zeros((rotated_height, rotated_width, 3), dtype=np.uint8)
        
        # Process trailing stick figure first (behind main figure)
        if trailing_frame is not None:
            # Convert trailing frame to grayscale and create mask
            trailing_gray = cv2.cvtColor(trailing_frame, cv2.COLOR_BGR2GRAY)
            _, trailing_mask = cv2.threshold(trailing_gray, 1, 255, cv2.THRESH_BINARY)
            
            if np.any(trailing_mask):
                # Apply heavy blur for trailing effect (2x the main glow intensity)
                trailing_blur_intensity = glow_intensity * 2
                trailing_blurred_mask = cv2.GaussianBlur(trailing_mask.astype(np.float32), 
                                                        (trailing_blur_intensity * 2 + 1, trailing_blur_intensity * 2 + 1), 
                                                        0) / 255.0
                
                # Apply additional blur passes for smoother effect
                trailing_blurred_mask = cv2.GaussianBlur(trailing_blurred_mask, 
                                                        (trailing_blur_intensity + 1, trailing_blur_intensity + 1), 
                                                        0)
                
                # Create strong glow overlay for trailing figure
                trailing_glow_overlay = np.zeros((rotated_height, rotated_width, 3), dtype=np.float32)
                for c in range(3):
                    trailing_glow_overlay[:, :, c] = trailing_blurred_mask * ghost_color[c]
                
                # Add trailing glow (stronger than main, more transparent)
                ghost_frame = ghost_frame.astype(np.float32) + trailing_glow_overlay * 0.8
                
                # Add semi-transparent filled trailing ghost
                trailing_fill_mask = (trailing_mask.astype(np.float32) / 255.0)
                trailing_fill_overlay = np.zeros((rotated_height, rotated_width, 3), dtype=np.float32)
                for c in range(3):
                    trailing_fill_overlay[:, :, c] = trailing_fill_mask * ghost_color[c] * (fill_transparency * 0.6)
                ghost_frame = ghost_frame + trailing_fill_overlay
                
                # Add faded outline for trailing figure
                trailing_outline = np.zeros((rotated_height, rotated_width, 3), dtype=np.float32)
                trailing_outline[trailing_mask > 0] = [255, 255, 255]
                ghost_frame = ghost_frame + trailing_outline * 0.2
        
        # Apply ghost effects to main rotated frame
        gray = cv2.cvtColor(stick_frame_rotated, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        if np.any(mask):
            # Apply blur for glow (main figure)
            blurred_mask = cv2.GaussianBlur(mask.astype(np.float32), 
                                          (glow_intensity * 2 + 1, glow_intensity * 2 + 1), 
                                          0) / 255.0
            
            # Create glow overlay
            glow_overlay = np.zeros((rotated_height, rotated_width, 3), dtype=np.float32)
            for c in range(3):
                glow_overlay[:, :, c] = blurred_mask * ghost_color[c]
            
            # Add main ghost glow on top of trailing figure
            ghost_frame = ghost_frame.astype(np.float32) + glow_overlay * 0.6
            
            # Add filled ghost
            fill_mask = (mask.astype(np.float32) / 255.0)
            fill_overlay = np.zeros((rotated_height, rotated_width, 3), dtype=np.float32)
            for c in range(3):
                fill_overlay[:, :, c] = fill_mask * ghost_color[c] * fill_transparency
            ghost_frame = ghost_frame + fill_overlay
            
            # Add outline
            white_outline = np.zeros((rotated_height, rotated_width, 3), dtype=np.float32)
            white_outline[mask > 0] = [255, 255, 255]
            ghost_frame = ghost_frame + white_outline * 0.4
        
        # Ensure ghost_frame is uint8 before writing
        ghost_frame = np.clip(ghost_frame, 0, 255).astype(np.uint8)
        
        out.write(ghost_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    cap.release()
    out.release()
    pose_detector.close()
    
    print(f"Ghost processing complete! Output saved to: {output_path}")
    print(f"Total poses detected: {poses_detected} out of {total_frames} frames")


def process_witch_video(input_path: str, output_path: str, witch_image_path: str = "test_videos/stock_witch.jpg") -> None:
    """
    Process input video to witch mode output (stick figure with witch convolution and accessories).
    
    Args:
        input_path: Path to input video file
        output_path: Path to save witch output video
        witch_image_path: Path to witch image file
    """
    # Initialize components
    pose_detector = PoseDetection()
    
    # Load witch image as kernel (base size, will be resized per frame based on person height)
    print(f"Loading witch image for convolution kernel from: {witch_image_path}")
    base_kernel_size = 25
    witch_kernel_base = load_witch_kernel(witch_image_path, base_kernel_size)
    if witch_kernel_base is None:
        raise ValueError(f"Could not load witch image from: {witch_image_path}")
    print(f"Witch kernel loaded: {base_kernel_size}x{base_kernel_size}")
    
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
    
    # Initialize renderer
    renderer = StickFigureRenderer(width, height)
    
    # Initialize video writer (rotated 90° clockwise)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rotated_width = height
    rotated_height = width
    out = cv2.VideoWriter(output_path, fourcc, fps, (rotated_width, rotated_height))
    
    print(f"Processing video: {input_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    print("Creating witch effects...")
    
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
        
        # Render stick figure first
        stick_frame = renderer.render_stick_figure(pose_results, spooky_head=False)
        
        # Estimate height and classify for accessories and kernel scaling
        height_ratio = None
        classification = None
        feet_x = width // 2
        feet_y = height - 50
        person_top_y = 50
        person_bottom_y = height - 50
        person_center_x = width // 2
        
        if pose_results and pose_results.pose_landmarks:
            # Estimate height and classify
            height_ratio, classification = estimate_person_height(pose_results, height, width)
            
            # Get landmarks for positioning accessories
            landmarks = pose_results.pose_landmarks.landmark
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            feet_x = int(((left_ankle.x + right_ankle.x) / 2) * width)
            feet_y = int(max(left_ankle.y, right_ankle.y) * height)
            
            nose = landmarks[0]
            person_top_y = int((nose.y * height) - height * 0.05)
            person_bottom_y = int(max(left_ankle.y, right_ankle.y) * height)
            person_center_x = int(nose.x * width)
        
        # Calculate dynamic kernel size based on person height
        kernel_size = get_scaled_kernel_size(base_kernel_size, height_ratio)
        
        # Resize and re-normalize kernel for this frame
        witch_kernel = load_witch_kernel(witch_image_path, kernel_size)
        if witch_kernel is None:
            # Fallback to base kernel if resize fails
            witch_kernel = witch_kernel_base
        
        # Apply convolution to stick figure
        output_frame = apply_witch_convolution(stick_frame, witch_kernel)
        
        # Draw accessories based on classification
        if classification == "kid" and height_ratio:
            # Draw skateboard for kids
            output_frame = draw_skateboard(output_frame, feet_x, feet_y, height_ratio, width)
        elif classification == "adult":
            # Draw broom for adults
            output_frame = draw_broom(output_frame, person_center_x, person_top_y, person_bottom_y, height_ratio or 0.5)
        
        # Rotate frame 90 degrees clockwise
        rotated_frame = cv2.rotate(output_frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Write output frame
        out.write(rotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    cap.release()
    out.release()
    pose_detector.close()
    
    print(f"Witch processing complete! Output saved to: {output_path}")
    print(f"Total poses detected: {poses_detected} out of {total_frames} frames")
