"""
Hand landmark detection using MediaPipe.
Provides real-time 21-point hand landmark tracking for piano posture analysis.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Path to the hand landmarker model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task')


# MediaPipe hand landmark indices
class LandmarkIndex:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Finger landmark groups: (MCP, PIP, DIP, TIP)
FINGER_LANDMARKS = {
    'thumb': (LandmarkIndex.THUMB_CMC, LandmarkIndex.THUMB_MCP, LandmarkIndex.THUMB_IP, LandmarkIndex.THUMB_TIP),
    'index': (LandmarkIndex.INDEX_MCP, LandmarkIndex.INDEX_PIP, LandmarkIndex.INDEX_DIP, LandmarkIndex.INDEX_TIP),
    'middle': (LandmarkIndex.MIDDLE_MCP, LandmarkIndex.MIDDLE_PIP, LandmarkIndex.MIDDLE_DIP, LandmarkIndex.MIDDLE_TIP),
    'ring': (LandmarkIndex.RING_MCP, LandmarkIndex.RING_PIP, LandmarkIndex.RING_DIP, LandmarkIndex.RING_TIP),
    'pinky': (LandmarkIndex.PINKY_MCP, LandmarkIndex.PINKY_PIP, LandmarkIndex.PINKY_DIP, LandmarkIndex.PINKY_TIP),
}

# Colors (BGR) used to render landmarks by anatomical joint role.
# Note: the thumb has CMC + IP joints (no PIP/DIP) -- they get their own colors.
LANDMARK_ROLE_COLORS = {
    'WRIST': (255, 255, 255),  # white
    'CMC':   (0, 165, 255),    # orange (thumb base)
    'MCP':   (0, 0, 255),      # red
    'PIP':   (0, 255, 255),    # yellow
    'DIP':   (255, 255, 0),    # cyan
    'IP':    (0, 255, 0),      # green (thumb-only)
    'TIP':   (255, 0, 255),    # magenta
}

# Anatomical role for every MediaPipe hand landmark index (0-20).
LANDMARK_ROLES = {
    0: 'WRIST',
    1: 'CMC',  2: 'MCP',  3: 'IP',   4: 'TIP',   # thumb
    5: 'MCP',  6: 'PIP',  7: 'DIP',  8: 'TIP',   # index
    9: 'MCP', 10: 'PIP', 11: 'DIP', 12: 'TIP',   # middle
    13: 'MCP',14: 'PIP', 15: 'DIP', 16: 'TIP',   # ring
    17: 'MCP',18: 'PIP', 19: 'DIP', 20: 'TIP',   # pinky
}

# MCP indices for palm centroid calculation
MCP_INDICES = [
    LandmarkIndex.INDEX_MCP,
    LandmarkIndex.MIDDLE_MCP,
    LandmarkIndex.RING_MCP,
    LandmarkIndex.PINKY_MCP,
]


@dataclass
class HandLandmarks:
    """Container for hand landmark data.

    - ``landmarks``: (21, 2) pixel coordinates in the image, used for drawing
      and for any 2D image-space measurement (e.g. hand-arch vs. a calibrated
      keyboard pixel line).
    - ``world_landmarks``: (21, 3) metric coordinates in meters with the origin
      approximately at the hand's geometric center. Use these for any angle or
      3D geometric computation -- unlike the image-space ``z`` channel, these
      are on a real metric scale and far more stable under self-occlusion.
    """
    landmarks: np.ndarray  # Shape: (21, 2) - x, y pixel coordinates
    world_landmarks: np.ndarray  # Shape: (21, 3) - x, y, z in meters
    confidence: float
    handedness: str  # 'Left' or 'Right'
    
    def get_landmark(self, index: int) -> np.ndarray:
        """Get a single pixel-space landmark as (x, y) array."""
        return self.landmarks[index]
    
    def get_mcp_centroid(self) -> np.ndarray:
        """Get the pixel-space centroid of MCP joints (palm center)."""
        mcp_points = self.landmarks[MCP_INDICES]
        return np.mean(mcp_points, axis=0)
    
    def get_wrist(self) -> np.ndarray:
        """Get pixel-space wrist landmark."""
        return self.landmarks[LandmarkIndex.WRIST]
    
    def get_world_landmark(self, index: int) -> np.ndarray:
        """Get a single world-space landmark as (x, y, z) array in meters."""
        return self.world_landmarks[index]
    
    def get_world_mcp_centroid(self) -> np.ndarray:
        """Get the world-space centroid of MCP joints (palm center)."""
        mcp_points = self.world_landmarks[MCP_INDICES]
        return np.mean(mcp_points, axis=0)
    
    def get_world_wrist(self) -> np.ndarray:
        """Get world-space wrist landmark."""
        return self.world_landmarks[LandmarkIndex.WRIST]


class HandTracker:
    """Real-time hand landmark tracker using MediaPipe."""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.max_num_hands = max_num_hands
        
        # Create HandLandmarker with task-based API
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._frame_timestamp = 0
    
    def process_frame(
        self, 
        frame: np.ndarray,
        target_hand: str = 'Right'
    ) -> Optional[HandLandmarks]:
        """
        Process a single frame and extract hand landmarks.
        
        Args:
            frame: BGR image from OpenCV
            target_hand: 'Right' or 'Left' - which hand to track
            
        Returns:
            HandLandmarks if target hand detected with sufficient confidence, else None
        """
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process with timestamp (required for VIDEO mode, must be monotonically increasing)
        self._frame_timestamp += 33  # ~30fps
        results = self.detector.detect_for_video(mp_image, self._frame_timestamp)
        
        if not results.hand_landmarks or not results.handedness:
            return None
        
        # World landmarks are aligned 1:1 with hand_landmarks by index.
        world_list = getattr(results, 'hand_world_landmarks', None) or []
        
        # Find the target hand
        for i, (hand_landmarks, handedness) in enumerate(
            zip(results.hand_landmarks, results.handedness)
        ):
            hand_label = handedness[0].category_name
            confidence = handedness[0].score
            
            # Note: MediaPipe returns mirrored labels for front-facing camera
            # 'Right' in MediaPipe means the hand appears on the right side of the image
            if hand_label == target_hand:
                # Convert normalized landmarks to pixel coordinates
                landmarks = np.array([
                    [lm.x * w, lm.y * h] 
                    for lm in hand_landmarks
                ])
                
                # Metric 3D skeleton centred at the hand (meters). Fall back to
                # zeros if the model happens not to emit them, so downstream code
                # still receives a well-shaped array.
                if i < len(world_list):
                    world_landmarks = np.array([
                        [lm.x, lm.y, lm.z] for lm in world_list[i]
                    ])
                else:
                    world_landmarks = np.zeros((21, 3), dtype=np.float64)
                
                return HandLandmarks(
                    landmarks=landmarks,
                    world_landmarks=world_landmarks,
                    confidence=confidence,
                    handedness=hand_label,
                )
        
        return None
    
    def draw_landmarks(
        self, 
        frame: np.ndarray, 
        hand_landmarks: HandLandmarks,
        draw_connections: bool = True,
    ) -> np.ndarray:
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: BGR image to draw on
            hand_landmarks: HandLandmarks object
            draw_connections: Whether to draw connections between landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        output = frame.copy()
        landmarks = hand_landmarks.landmarks
        
        # Draw connections
        if draw_connections:
            # Define hand connections manually
            connections = [
                # Thumb
                (0, 1), (1, 2), (2, 3), (3, 4),
                # Index finger
                (0, 5), (5, 6), (6, 7), (7, 8),
                # Middle finger
                (0, 9), (9, 10), (10, 11), (11, 12),
                # Ring finger
                (0, 13), (13, 14), (14, 15), (15, 16),
                # Pinky
                (0, 17), (17, 18), (18, 19), (19, 20),
                # Palm
                (5, 9), (9, 13), (13, 17),
            ]
            for start_idx, end_idx in connections:
                start = tuple(landmarks[start_idx].astype(int))
                end = tuple(landmarks[end_idx].astype(int))
                cv2.line(output, start, end, (0, 255, 0), 2)
        
        # Draw landmark points colored by anatomical joint role.
        for i, (x, y) in enumerate(landmarks):
            role = LANDMARK_ROLES.get(i, 'WRIST')
            color = LANDMARK_ROLE_COLORS[role]
            cv2.circle(output, (int(x), int(y)), 5, color, -1)
            cv2.circle(output, (int(x), int(y)), 5, (0, 0, 0), 1)

        return output
    
    def close(self):
        """Release MediaPipe resources."""
        self.detector.close()


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
    """Demo: Run hand tracking on webcam feed."""
    print("Starting hand tracking demo...")
    print("Press 'q' to quit")
    
    # Initialize video capture and hand tracker
    video = VideoCapture(source=0, width=640, height=480)
    tracker = HandTracker(min_detection_confidence=0.7)
    
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
                    f"Right hand detected (conf: {hand.confidence:.2f})",
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
                    "No right hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            
            cv2.imshow('Piano Posture - Hand Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        video.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
