"""
Piano Posture Analyzer - Main Application.

Real-time piano hand posture analysis using MediaPipe hand tracking.
Draws per-frame finger/wrist feedback, maintains a running posture score,
and optionally logs metrics + screenshots to a CSV.

Usage:
    python main.py [--camera ID] [--width W] [--height H] [--fps FPS]

Controls:
    q     - Quit
    r     - Reset running score history
    s     - Save screenshot
    h     - Toggle tracked hand (Left/Right)
    m     - Toggle mirror mode
    SPACE - Log current stats to CSV + screenshot
    a     - Start a 30-second recording session (video + stats + screenshots)
    1     - Toggle posture overlay (UI)
    2     - Toggle landmark skeleton
    3     - Toggle joint-role color legend
    4     - Reset view toggles to defaults
    0     - Toggle landmark index numbers (0-20)
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np

from hand_tracker import HandTracker, VideoCapture, LANDMARK_ROLE_COLORS
from posture_analyzer import PostureAnalyzer, PostureScorer
from visualizer import PostureVisualizer


class PianoPostureApp:
    """Main application for piano posture analysis."""

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.video = VideoCapture(source=camera_id, width=width, height=height, fps=fps)
        self.tracker = HandTracker(min_detection_confidence=0.7)
        self.analyzer = PostureAnalyzer()
        self.scorer = PostureScorer(window_size=30)
        self.visualizer = PostureVisualizer()

        self.running = True
        self.target_hand = 'Left'  # Start with Left because image is mirrored
        self.mirror_mode = True
        self._frame_count = 0

        # Display toggles
        self.show_overlay = True    # toggled by '1' (posture UI/panel/indicators)
        self.show_landmarks = True  # toggled by '2' (raw landmark skeleton)
        self.show_legend = False    # toggled by '3' (joint-role color legend)
        self.show_indices = False   # toggled by '0' (landmark index numbers 0-20)

        # Results directory
        self.results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        # Stats log file
        self.stats_file = os.path.join(self.results_dir, 'stats_log.csv')
        self._init_stats_file()

        # Recording session state
        self.record_duration = 10.0
        self.recording = False
        self._rec_start: float = 0.0
        self._rec_dir: str = ''
        self._rec_writer = None
        self._rec_stats_path: str = ''
        self._rec_stats_fh = None
        self._rec_metrics: list = []
        self._rec_screenshot_marks: set = set()

    # -- stats logging -----------------------------------------------------

    def _init_stats_file(self) -> None:
        """Initialize the stats CSV file with headers if it doesn't exist."""
        if os.path.exists(self.stats_file):
            return
        with open(self.stats_file, 'w') as f:
            f.write('timestamp,hand,mirror,')
            f.write('thumb_pip,thumb_dip,index_pip,index_dip,middle_pip,middle_dip,')
            f.write('ring_pip,ring_dip,pinky_pip,pinky_dip,')
            f.write('wrist_angle,hand_arch,score,screenshot\n')

    def _log_stats(self, metrics, frame=None) -> None:
        """Log current stats to CSV and save a screenshot."""
        if metrics is None:
            print("No hand detected - cannot log stats")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

        screenshot_name = ''
        if frame is not None:
            screenshots_dir = os.path.join(self.results_dir, 'screenshots')
            os.makedirs(screenshots_dir, exist_ok=True)
            screenshot_name = f"snapshot_{ts_file}.png"
            cv2.imwrite(os.path.join(screenshots_dir, screenshot_name), frame)

        row = [timestamp, self.target_hand, str(self.mirror_mode)]
        for finger_name in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            fm = metrics.fingers[finger_name]
            row.append(f"{fm.pip_angle:.1f}")
            row.append(f"{fm.dip_angle:.1f}")

        debug = self.analyzer.get_debug_info()
        wrist_angle = debug['raw_angle'] if debug else 0
        row.append(f"{wrist_angle:.1f}")
        row.append(f"{metrics.hand_arch:.3f}")
        row.append(f"{metrics.score:.1f}")
        row.append(screenshot_name)

        with open(self.stats_file, 'a') as f:
            f.write(','.join(row) + '\n')

        print(f"Stats logged: Score={metrics.score:.0f} | Screenshot: {screenshot_name}")

    # -- debug / screenshots ----------------------------------------------

    def _print_debug_line(self, metrics) -> None:
        """One-line periodic debug print summarising per-frame state."""
        pip_angles = [f"{m.name[0].upper()}:{m.pip_angle:.0f}" for m in metrics.fingers.values()]
        dip_angles = [f"{m.name[0].upper()}:{m.dip_angle:.0f}" for m in metrics.fingers.values()]
        # Per-finger MCP angle + one-letter posture flag (G/C/X).
        status_flag = {'good': 'G', 'collapsed': 'C', 'excessive': 'X'}
        mcp_angles = []
        for m in metrics.fingers.values():
            if m.mcp_angle is None or m.mcp_posture is None:
                continue
            flag = status_flag.get(m.mcp_posture.value, '?')
            mcp_angles.append(f"{m.name[0].upper()}:{m.mcp_angle:.0f}{flag}")
        debug = self.analyzer.get_debug_info()
        wrist_angle = debug['raw_angle'] if debug else 0
        closed_tag = " | CLOSED-HAND" if metrics.closed_hand else ""
        print(
            f"PIP: {', '.join(pip_angles)} | "
            f"MCP: {', '.join(mcp_angles)} | "
            f"DIP: {', '.join(dip_angles)} | "
            f"Wrist: {wrist_angle:.0f}° | "
            f"Arch(meanMCP): {metrics.hand_arch:.0f}° "
            f"({metrics.hand_arch_status.value}) | "
            f"Open: {metrics.openness:.2f}{closed_tag} | "
            f"Score: {metrics.score:.0f}"
        )

    def _save_screenshot(self, frame) -> None:
        """Save current displayed frame as a screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f"screenshot_{timestamp}.png")
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")

    # -- recording session -------------------------------------------------

    def _start_recording(self) -> None:
        if self.recording:
            print("Already recording.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._rec_dir = os.path.join(self.results_dir, f"recording_{ts}")
        os.makedirs(self._rec_dir, exist_ok=True)
        os.makedirs(os.path.join(self._rec_dir, 'screenshots'), exist_ok=True)

        video_path = os.path.join(self._rec_dir, 'video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.video.fps if self.video.fps > 0 else 30
        self._rec_writer = cv2.VideoWriter(
            video_path, fourcc, fps, (self.video.width, self.video.height)
        )

        self._rec_stats_path = os.path.join(self._rec_dir, 'stats.csv')
        self._rec_stats_fh = open(self._rec_stats_path, 'w')
        self._rec_stats_fh.write(
            't_sec,score,thumb_pip,thumb_dip,index_pip,index_dip,'
            'middle_pip,middle_dip,ring_pip,ring_dip,pinky_pip,pinky_dip,'
            'wrist_angle,hand_arch,openness,closed_hand\n'
        )

        self._rec_metrics = []
        self._rec_screenshot_marks = set()
        self._rec_start = time.time()
        self.recording = True
        print(f"\n[REC] Started 30s recording -> {self._rec_dir}")

    def _update_recording(self, annotated_frame, clean_frame, metrics) -> None:
        """Called every frame while recording. Writes video, stats, screenshots."""
        if not self.recording:
            return

        elapsed = time.time() - self._rec_start

        # Write video (use clean annotated frame without countdown overlay).
        if self._rec_writer is not None:
            # Ensure frame matches writer size.
            h, w = clean_frame.shape[:2]
            if (w, h) != (self.video.width, self.video.height):
                clean_frame = cv2.resize(
                    clean_frame, (self.video.width, self.video.height)
                )
            self._rec_writer.write(clean_frame)

        # Per-frame stats row.
        if metrics is not None and self._rec_stats_fh is not None:
            debug = self.analyzer.get_debug_info()
            wrist_angle = debug['raw_angle'] if debug else 0.0
            row = [f"{elapsed:.3f}", f"{metrics.score:.2f}"]
            for fn in ('thumb', 'index', 'middle', 'ring', 'pinky'):
                fm = metrics.fingers[fn]
                row.append(f"{fm.pip_angle:.2f}")
                row.append(f"{fm.dip_angle:.2f}")
            row.append(f"{wrist_angle:.2f}")
            row.append(f"{metrics.hand_arch:.3f}")
            row.append(f"{metrics.openness:.3f}")
            row.append(str(metrics.closed_hand))
            self._rec_stats_fh.write(','.join(row) + '\n')
            self._rec_metrics.append(metrics)

        # Screenshot every 5 seconds (0, 5, 10, ..., 30).
        mark = int(elapsed // 5) * 5
        if 0 <= mark <= int(self.record_duration) and mark not in self._rec_screenshot_marks:
            self._rec_screenshot_marks.add(mark)
            shot_path = os.path.join(
                self._rec_dir, 'screenshots', f"t{mark:02d}s.png"
            )
            cv2.imwrite(shot_path, annotated_frame)

        if elapsed >= self.record_duration:
            self._stop_recording()

    def _stop_recording(self) -> None:
        if not self.recording:
            return
        self.recording = False

        if self._rec_writer is not None:
            self._rec_writer.release()
            self._rec_writer = None
        if self._rec_stats_fh is not None:
            self._rec_stats_fh.close()
            self._rec_stats_fh = None

        self._write_recording_summary()
        print(f"[REC] Done. Output: {self._rec_dir}\n")

    def _write_recording_summary(self) -> None:
        summary_path = os.path.join(self._rec_dir, 'summary.txt')
        metrics_list = self._rec_metrics
        duration = time.time() - self._rec_start

        lines = []
        lines.append("Piano Posture Recording Summary")
        lines.append("=" * 40)
        lines.append(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Duration  : {duration:.2f} s")
        lines.append(f"Hand      : {self.target_hand}")
        lines.append(f"Mirror    : {self.mirror_mode}")
        lines.append(f"Frames w/ detection: {len(metrics_list)}")
        lines.append("")

        if not metrics_list:
            lines.append("No hand detected during recording.")
        else:
            scores = np.array([m.score for m in metrics_list])
            lines.append("Score:")
            lines.append(f"  mean={scores.mean():.1f}  min={scores.min():.1f}  max={scores.max():.1f}  std={scores.std():.1f}")
            lines.append("")

            lines.append("Per-finger PIP angle (deg) mean / std:")
            for fn in ('thumb', 'index', 'middle', 'ring', 'pinky'):
                vals = np.array([m.fingers[fn].pip_angle for m in metrics_list])
                lines.append(f"  {fn:<7s} mean={vals.mean():6.1f}  std={vals.std():5.1f}")
            lines.append("")

            lines.append("Per-finger DIP angle (deg) mean / std:")
            for fn in ('thumb', 'index', 'middle', 'ring', 'pinky'):
                vals = np.array([m.fingers[fn].dip_angle for m in metrics_list])
                lines.append(f"  {fn:<7s} mean={vals.mean():6.1f}  std={vals.std():5.1f}")
            lines.append("")

            lines.append("Per-finger PIP posture distribution:")
            for fn in ('thumb', 'index', 'middle', 'ring', 'pinky'):
                counts = Counter(
                    m.fingers[fn].posture.value for m in metrics_list
                )
                total = sum(counts.values())
                dist = ', '.join(
                    f"{k}={v/total*100:.1f}%" for k, v in counts.most_common()
                )
                lines.append(f"  {fn:<7s} {dist}")
            lines.append("")

            arch = np.array([m.hand_arch for m in metrics_list])
            lines.append("Hand arch (mean MCP angle, deg):")
            lines.append(f"  mean={arch.mean():.1f}  min={arch.min():.1f}  max={arch.max():.1f}")
            arch_dist = Counter(m.hand_arch_status.value for m in metrics_list)
            total = sum(arch_dist.values())
            lines.append(
                "  distribution: "
                + ', '.join(f"{k}={v/total*100:.1f}%" for k, v in arch_dist.most_common())
            )
            lines.append("")

            openness = np.array([m.openness for m in metrics_list])
            closed_pct = np.mean([1.0 if m.closed_hand else 0.0 for m in metrics_list]) * 100
            lines.append("Openness:")
            lines.append(f"  mean={openness.mean():.2f}  closed-hand frames={closed_pct:.1f}%")

        with open(summary_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    def _draw_landmark_indices(self, frame, landmarks):
        """Overlay the 21 MediaPipe landmark index numbers next to each joint."""
        pts = landmarks.landmarks
        for i, (x, y) in enumerate(pts):
            px, py = int(x), int(y)
            text = str(i)
            # Black outline + white text for readability on any background.
            cv2.putText(frame, text, (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(frame, text, (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
        caption = ("Fig. 2. The 21 MediaPipe Hand landmarks are overlaid on the "
                   "detected hand,")
        caption2 = "with indices matching the joints for each finger."
        h, w = frame.shape[:2]
        y1 = h - 30
        y2 = h - 12
        for txt, y in ((caption, y1), (caption2, y2)):
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _draw_joint_legend(self, frame):
        """Draw a small legend mapping joint role -> dot color (bottom-left)."""
        labels = ['WRIST', 'CMC', 'MCP', 'PIP', 'DIP', 'IP', 'TIP']
        items = [(lbl, LANDMARK_ROLE_COLORS[lbl]) for lbl in labels]
        h, w = frame.shape[:2]
        pad = 8
        line_h = 20
        box_w = 130
        box_h = pad * 2 + line_h * len(items)
        x0 = 10
        y0 = h - box_h - 10
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)
        for i, (label, color) in enumerate(items):
            cy = y0 + pad + line_h * i + line_h // 2
            cv2.circle(frame, (x0 + 14, cy), 6, color, -1)
            cv2.circle(frame, (x0 + 14, cy), 6, (0, 0, 0), 1)
            cv2.putText(frame, label, (x0 + 30, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def _draw_countdown_overlay(self, frame):
        """Draw a countdown timer + REC indicator on top of the frame."""
        remaining = max(0.0, self.record_duration - (time.time() - self._rec_start))
        text = f"REC  {remaining:4.1f}s"
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x = w - tw - 20
        y = 40
        cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + 10), (0, 0, 0), -1)
        cv2.circle(frame, (x - 25, y - th // 2), 8, (0, 0, 255), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return frame

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        if not self.video.is_opened():
            print("Error: Could not open camera")
            return

        print(f"Camera initialized: {self.video.width}x{self.video.height} @ {self.video.fps}fps")
        print("\nControls:")
        print("  r     - Reset score history")
        print("  s     - Save screenshot")
        print("  h     - Toggle hand (Left/Right)")
        print("  m     - Toggle mirror mode")
        print("  SPACE - Log stats to file")
        print("  a     - Start 30s recording session")
        print("  1     - Toggle posture overlay (UI)")
        print("  2     - Toggle landmark skeleton")
        print("  3     - Toggle joint-role color legend")
        print("  4     - Reset view toggles to defaults")
        print("  0     - Toggle landmark index numbers (0-20)")
        print("  q     - Quit")
        print(f"\nCurrently tracking: {self.target_hand} hand (mirror: {self.mirror_mode})")
        print(f"Stats file: {self.stats_file}")
        print("Starting posture analysis...")

        window_name = 'Piano Posture Analyzer'
        cv2.namedWindow(window_name)

        try:
            while self.running:
                ret, frame = self.video.read()
                if not ret:
                    print("Error: Could not read frame")
                    break

                if self.mirror_mode:
                    frame = cv2.flip(frame, 1)

                landmarks = self.tracker.process_frame(frame, target_hand=self.target_hand)

                metrics = None
                running_score = self.scorer.get_average_score()

                if landmarks is not None:
                    metrics = self.analyzer.analyze(landmarks)
                    running_score = self.scorer.update(metrics)

                    self._frame_count += 1
                    if self._frame_count % 30 == 0:
                        self._print_debug_line(metrics)

                    if self.show_landmarks:
                        frame = self.tracker.draw_landmarks(frame, landmarks)

                if self.show_overlay:
                    debug_info = self.analyzer.get_debug_info() if metrics else None
                    frame = self.visualizer.draw(frame, landmarks, metrics, running_score, debug_info)

                if self.show_legend:
                    frame = self._draw_joint_legend(frame)

                if self.show_indices and landmarks is not None:
                    frame = self._draw_landmark_indices(frame, landmarks)

                # Frame used for video + periodic screenshots (no countdown overlay).
                clean_frame = frame.copy() if self.recording else frame

                display_frame = frame
                if self.recording:
                    display_frame = self._draw_countdown_overlay(frame.copy())
                    self._update_recording(display_frame, clean_frame, metrics)

                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('r'):
                    self.scorer.reset()
                    print("Score history reset")
                elif key == ord('s'):
                    self._save_screenshot(display_frame)
                elif key == ord('h'):
                    self.target_hand = 'Right' if self.target_hand == 'Left' else 'Left'
                    self.scorer.reset()
                    print(f"Now tracking: {self.target_hand} hand")
                elif key == ord('m'):
                    self.mirror_mode = not self.mirror_mode
                    print(f"Mirror mode: {self.mirror_mode}")
                elif key == ord(' '):
                    self._log_stats(metrics, display_frame)
                elif key == ord('a'):
                    self._start_recording()
                elif key == ord('1'):
                    self.show_overlay = not self.show_overlay
                    print(f"Posture overlay: {'ON' if self.show_overlay else 'OFF'}")
                elif key == ord('2'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmark skeleton: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('3'):
                    self.show_legend = not self.show_legend
                    print(f"Joint legend: {'ON' if self.show_legend else 'OFF'}")
                elif key == ord('4'):
                    self.show_overlay = True
                    self.show_landmarks = True
                    self.show_legend = False
                    self.show_indices = False
                    print("View reset to defaults (overlay ON, landmarks ON, legend OFF, indices OFF)")
                elif key == ord('0'):
                    self.show_indices = not self.show_indices
                    print(f"Landmark indices: {'ON' if self.show_indices else 'OFF'}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Release resources."""
        if self.recording:
            self._stop_recording()
        self.video.release()
        self.tracker.close()
        cv2.destroyAllWindows()
        print("\nApplication closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description='Piano Posture Analyzer')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--width', type=int, default=640, help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Frame height (default: 480)')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
    args = parser.parse_args()

    app = PianoPostureApp(
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    app.run()


if __name__ == '__main__':
    main()
