"""
Simplified hand tracking using OpenCV and basic computer vision.
This is a fallback implementation when MediaPipe is not available.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class HandLandmarks:
    """Container for simplified hand landmark data."""
    landmarks: np.ndarray  # Shape: (21, 2) - x, y pixel coordinates (simulated)
    confidence: float
    handedness: str  # 'Left' or 'Right'
    
    def get_landmark(self, index: int) -> np.ndarray:
        """Get a single landmark as (x, y) array."""
        return self.landmarks[index]
    
    def get_mcp_centroid(self) -> np.ndarray:
        """Get the centroid of MCP joints (palm center approximation)."""
        # Use indices 5, 9, 13, 17 (MCP joints)
        mcp_indices = [5, 9, 13, 17]
        mcp_points = self.landmarks[mcp_indices]
        return np.mean(mcp_points, axis=0)
    
    def get_wrist(self) -> np.ndarray:
        """Get wrist landmark."""
        return self.landmarks[0]


class SimpleHandTracker:
    """
    Simplified hand tracker using OpenCV contour detection.
    This generates approximate hand landmarks for testing purposes.
    """
    
    def __init__(self, min_detection_confidence: float = 0.7):
        self.min_confidence = min_detection_confidence
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        
    def _generate_fake_landmarks(self, contour: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Generate 21 fake landmarks based on hand contour."""
        # Get bounding box and moments
        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        
        if M["m00"] == 0:
            return None
            
        # Center of mass
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Generate 21 landmarks in a hand-like pattern
        landmarks = np.zeros((21, 2))
        
        # Wrist (0)
        landmarks[0] = [cx, y + h * 0.9]
        
        # Thumb (1-4)
        thumb_base_x = x + w * 0.2
        thumb_base_y = y + h * 0.7
        for i in range(4):
            landmarks[1 + i] = [
                thumb_base_x - i * w * 0.05,
                thumb_base_y - i * h * 0.15
            ]
        
        # Index finger (5-8)
        index_base_x = x + w * 0.3
        index_base_y = y + h * 0.6
        for i in range(4):
            landmarks[5 + i] = [
                index_base_x + i * w * 0.02,
                index_base_y - i * h * 0.2
            ]
        
        # Middle finger (9-12)
        middle_base_x = x + w * 0.5
        middle_base_y = y + h * 0.6
        for i in range(4):
            landmarks[9 + i] = [
                middle_base_x,
                middle_base_y - i * h * 0.25
            ]
        
        # Ring finger (13-16)
        ring_base_x = x + w * 0.7
        ring_base_y = y + h * 0.6
        for i in range(4):
            landmarks[13 + i] = [
                ring_base_x - i * w * 0.02,
                ring_base_y - i * h * 0.2
            ]
        
        # Pinky (17-20)
        pinky_base_x = x + w * 0.85
        pinky_base_y = y + h * 0.65
        for i in range(4):
            landmarks[17 + i] = [
                pinky_base_x - i * w * 0.03,
                pinky_base_y - i * h * 0.15
            ]
        
        return landmarks
    
    def process_frame(
        self, 
        frame: np.ndarray,
        target_hand: str = 'Right'
    ) -> Optional[HandLandmarks]:
        """
        Process frame using simple contour detection.
        This is a basic implementation for testing.
        """
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range (this is very basic)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (assume it's the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Filter by area
        area = cv2.contourArea(largest_contour)
        if area < 5000:  # Minimum hand area
            return None
        
        # Generate fake landmarks
        landmarks = self._generate_fake_landmarks(largest_contour, frame.shape[:2])
        if landmarks is None:
            return None
        
        # Calculate confidence based on contour properties
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        confidence = min(0.9, solidity * 1.2)  # Simple confidence metric
        
        if confidence < self.min_confidence:
            return None
        
        return HandLandmarks(
            landmarks=landmarks,
            confidence=confidence,
            handedness=target_hand,
        )
    
    def draw_landmarks(
        self, 
        frame: np.ndarray, 
        hand_landmarks: HandLandmarks,
        draw_connections: bool = True,
    ) -> np.ndarray:
        """Draw hand landmarks on frame."""
        output = frame.copy()
        landmarks = hand_landmarks.landmarks
        
        # Simple connections (not anatomically correct, but visual)
        if draw_connections:
            connections = [
                # Thumb
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Index
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Middle  
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Ring
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Pinky
                (0, 17), (17, 18), (18, 19), (19, 20),
                # Palm connections
                (5, 9), (9, 13), (13, 17),
            ]
            
            for start_idx, end_idx in connections:
                start = tuple(landmarks[start_idx].astype(int))
                end = tuple(landmarks[end_idx].astype(int))
                cv2.line(output, start, end, (0, 255, 0), 2)
        
        # Draw landmark points
        for i, (x, y) in enumerate(landmarks):
            color = (255, 0, 0) if i == 0 else (0, 0, 255)  # Wrist is blue, others red
            cv2.circle(output, (int(x), int(y)), 4, color, -1)
        
        return output
    
    def close(self):
        """Release resources."""
        pass


class VideoCapture:
    """Wrapper for OpenCV video capture with convenient methods."""
    
    def __init__(
        self,
        source: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Get actual values (may differ from requested)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video source."""
        return self.cap.read()
    
    def is_opened(self) -> bool:
        """Check if video source is opened."""
        return self.cap.isOpened()
    
    def release(self):
        """Release video capture resources."""
        self.cap.release()


def main():
    """Demo: Run simple hand tracking on webcam feed."""
    print("Starting simple hand tracking demo...")
    print("Press 'q' to quit")
    print("NOTE: This is a basic implementation for testing")
    
    # Initialize video capture and hand tracker
    video = VideoCapture(source=0, width=640, height=480)
    tracker = SimpleHandTracker(min_detection_confidence=0.5)
    
    if not video.is_opened():
        print("Error: Could not open webcam")
        return
    
    print(f"Camera: {video.width}x{video.height} @ {video.fps}fps")
    
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame for hand landmarks
            hand = tracker.process_frame(frame, target_hand='Right')
            
            if hand is not None:
                # Draw landmarks
                frame = tracker.draw_landmarks(frame, hand)
                
                # Display info
                cv2.putText(
                    frame,
                    f"Hand detected (conf: {hand.confidence:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                
                # Show wrist and palm center
                wrist = hand.get_wrist()
                palm_center = hand.get_mcp_centroid()
                cv2.circle(frame, tuple(wrist.astype(int)), 8, (255, 0, 0), -1)
                cv2.circle(frame, tuple(palm_center.astype(int)), 8, (0, 0, 255), -1)
            else:
                cv2.putText(
                    frame,
                    "No hand detected - show your hand to camera",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            
            cv2.imshow('Simple Hand Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        video.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
