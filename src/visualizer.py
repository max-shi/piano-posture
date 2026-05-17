"""
Visualization module for piano posture feedback.
Draws overlays showing posture metrics and scores on video frames.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from hand_tracker import HandLandmarks, FINGER_LANDMARKS, LandmarkIndex
from posture_analyzer import (
    PostureMetrics, 
    FingerPosture, 
    WristPosition, 
    HandArchStatus,
)


# Color definitions (BGR format)
COLORS = {
    'good': (0, 255, 0),        # Green
    'warning': (0, 165, 255),   # Orange
    'bad': (0, 0, 255),         # Red
    'neutral': (255, 255, 0),   # Cyan
    'text': (255, 255, 255),    # White
    'background': (40, 40, 40), # Dark gray
}

# Posture to color mapping
FINGER_POSTURE_COLORS = {
    FingerPosture.CURVED: COLORS['good'],
    FingerPosture.NEUTRAL: COLORS['neutral'],
    FingerPosture.COLLAPSED: COLORS['warning'],
    FingerPosture.FLAT: COLORS['bad'],
}

WRIST_POSITION_COLORS = {
    WristPosition.NEUTRAL: COLORS['good'],
    WristPosition.HIGH: COLORS['warning'],
    WristPosition.LOW: COLORS['bad'],
}

ARCH_STATUS_COLORS = {
    HandArchStatus.GOOD: COLORS['good'],
    HandArchStatus.EXCESSIVE: COLORS['warning'],
    HandArchStatus.COLLAPSED: COLORS['bad'],
}


class PostureVisualizer:
    """Draws posture feedback overlays on video frames."""
    
    def __init__(
        self,
        show_landmarks: bool = True,
        show_angles: bool = True,
        show_score: bool = True,
        show_panel: bool = True,
    ):
        self.show_landmarks = show_landmarks
        self.show_angles = show_angles
        self.show_score = show_score
        self.show_panel = show_panel
    
    def _draw_score_bar(
        self, 
        frame: np.ndarray, 
        score: float,
        x: int, 
        y: int, 
        width: int = 200, 
        height: int = 20,
    ) -> np.ndarray:
        """Draw a score bar with gradient coloring."""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), COLORS['background'], -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), COLORS['text'], 1)
        
        # Fill based on score
        fill_width = int((score / 100) * width)
        
        # Color gradient: red -> yellow -> green
        if score < 50:
            color = COLORS['bad']
        elif score < 75:
            color = COLORS['warning']
        else:
            color = COLORS['good']
        
        cv2.rectangle(frame, (x + 2, y + 2), (x + fill_width - 2, y + height - 2), color, -1)
        
        # Score text
        cv2.putText(
            frame,
            f"{score:.0f}",
            (x + width + 10, y + height - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS['text'],
            2,
        )
        
        return frame
    
    def _draw_finger_indicators(
        self,
        frame: np.ndarray,
        landmarks: HandLandmarks,
        metrics: PostureMetrics,
    ) -> np.ndarray:
        """Draw colored circles at joints indicating posture quality with angles."""
        for finger_name, (mcp_idx, pip_idx, dip_idx, tip_idx) in FINGER_LANDMARKS.items():
            finger_metrics = metrics.fingers[finger_name]
            color = FINGER_POSTURE_COLORS[finger_metrics.posture]
            
            # Draw at PIP joint (where PIP angle is measured)
            pip_pos = landmarks.get_landmark(pip_idx).astype(int)
            cv2.circle(frame, tuple(pip_pos), 8, color, -1)
            cv2.circle(frame, tuple(pip_pos), 8, (0, 0, 0), 2)
            
            # Draw at DIP joint (where DIP angle is measured)
            dip_pos = landmarks.get_landmark(dip_idx).astype(int)
            cv2.circle(frame, tuple(dip_pos), 6, (255, 165, 0), -1)  # Orange for DIP
            cv2.circle(frame, tuple(dip_pos), 6, (0, 0, 0), 1)
            
            # Draw angle text if enabled
            if self.show_angles:
                # PIP angle (larger, at PIP joint)
                pip_text = f"{finger_metrics.pip_angle:.0f}"
                pip_text_pos = (pip_pos[0] - 12, pip_pos[1] - 12)
                cv2.putText(
                    frame,
                    pip_text,
                    pip_text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    COLORS['text'],
                    1,
                )
                
                # DIP angle (smaller, at DIP joint)
                dip_text = f"{finger_metrics.dip_angle:.0f}"
                dip_text_pos = (dip_pos[0] + 8, dip_pos[1] - 5)
                cv2.putText(
                    frame,
                    dip_text,
                    dip_text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 200, 100),  # Light orange
                    1,
                )
        
        return frame
    
    def _draw_wrist_indicator(
        self,
        frame: np.ndarray,
        landmarks: HandLandmarks,
        metrics: PostureMetrics,
        debug_info: dict = None,
    ) -> np.ndarray:
        """Draw wrist position indicator with angle visualization."""
        wrist = landmarks.get_wrist().astype(int)
        mcp_centroid = landmarks.get_mcp_centroid().astype(int)
        
        color = WRIST_POSITION_COLORS[metrics.wrist_position]
        
        # Draw line from wrist to MCP centroid
        cv2.line(frame, tuple(wrist), tuple(mcp_centroid), color, 3)
        
        # Draw wrist marker
        cv2.circle(frame, tuple(wrist), 12, color, -1)
        cv2.circle(frame, tuple(wrist), 12, (0, 0, 0), 2)
        
        # Draw angle debug info
        if debug_info and self.show_angles:
            raw_angle = debug_info.get('raw_angle', 0)
            deviation = debug_info.get('deviation', 0)
            
            # Draw horizontal reference line from wrist
            line_len = 60
            h_end = (wrist[0] + line_len, wrist[1])
            cv2.line(frame, tuple(wrist), h_end, (100, 100, 100), 1)
            
            # Draw actual angle line
            import math
            angle_rad = math.radians(raw_angle)
            a_end = (int(wrist[0] + line_len * math.cos(angle_rad)),
                     int(wrist[1] - line_len * math.sin(angle_rad)))  # Y inverted
            cv2.line(frame, tuple(wrist), a_end, color, 2)
            
            # Draw angle text
            angle_text = f"{raw_angle:.0f}deg"
            cv2.putText(
                frame,
                angle_text,
                (wrist[0] - 50, wrist[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLORS['text'],
                1,
            )
        
        return frame
    
    def _draw_info_panel(
        self,
        frame: np.ndarray,
        metrics: PostureMetrics,
        running_score: float,
    ) -> np.ndarray:
        """Draw information panel with posture details."""
        h, w = frame.shape[:2]
        panel_width = 220
        panel_x = w - panel_width - 10
        panel_y = 10
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (w - 10, panel_y + 280),
            COLORS['background'],
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(
            frame,
            "POSTURE ANALYSIS",
            (panel_x + 10, panel_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLORS['text'],
            2,
        )
        
        # Score bar
        cv2.putText(
            frame,
            "Score:",
            (panel_x + 10, panel_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS['text'],
            1,
        )
        self._draw_score_bar(frame, running_score, panel_x + 10, panel_y + 65, 150, 15)
        
        # Finger postures
        cv2.putText(
            frame,
            "Fingers:",
            (panel_x + 10, panel_y + 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLORS['text'],
            1,
        )
        
        y_offset = panel_y + 125
        for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            finger = metrics.fingers[finger_name]
            color = FINGER_POSTURE_COLORS[finger.posture]
            
            # Finger name and status
            cv2.circle(frame, (panel_x + 20, y_offset - 5), 6, color, -1)
            cv2.putText(
                frame,
                f"{finger_name.capitalize()}: {finger.posture.value}",
                (panel_x + 35, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLORS['text'],
                1,
            )
            y_offset += 20
        
        # Wrist position
        y_offset += 10
        wrist_color = WRIST_POSITION_COLORS[metrics.wrist_position]
        cv2.putText(
            frame,
            f"Wrist: {metrics.wrist_position.value}",
            (panel_x + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            wrist_color,
            1,
        )
        
        # Hand arch
        y_offset += 25
        arch_color = ARCH_STATUS_COLORS[metrics.hand_arch_status]
        cv2.putText(
            frame,
            f"Arch: {metrics.hand_arch_status.value}",
            (panel_x + 10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            arch_color,
            1,
        )
        
        return frame
    
    def draw(
        self,
        frame: np.ndarray,
        landmarks: Optional[HandLandmarks],
        metrics: Optional[PostureMetrics],
        running_score: float = 0.0,
        debug_info: dict = None,
    ) -> np.ndarray:
        """
        Draw all posture visualizations on frame.
        
        Args:
            frame: BGR image to draw on
            landmarks: Hand landmarks (or None if no hand detected)
            metrics: Posture metrics (or None if no analysis)
            running_score: Running average score
            debug_info: Debug info from analyzer (for wrist angle viz)
            
        Returns:
            Frame with visualizations
        """
        output = frame.copy()
        
        if landmarks is None or metrics is None:
            # No hand detected - show message
            cv2.putText(
                output,
                "No hand detected - Position right hand in view",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLORS['warning'],
                2,
            )
            return output
        
        # Draw finger indicators
        if self.show_landmarks:
            output = self._draw_finger_indicators(output, landmarks, metrics)
            output = self._draw_wrist_indicator(output, landmarks, metrics, debug_info)
        
        # Draw info panel
        if self.show_panel:
            output = self._draw_info_panel(output, metrics, running_score)
        
        # Draw simple score at top if panel disabled
        if self.show_score and not self.show_panel:
            cv2.putText(
                output,
                f"Score: {running_score:.0f}/100",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLORS['good'] if running_score >= 75 else COLORS['warning'],
                2,
            )
        
        return output
