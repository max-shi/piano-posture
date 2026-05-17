"""
Posture analysis module for piano hand posture evaluation.
Extracts geometric features from hand landmarks and classifies posture.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from hand_tracker import HandLandmarks, FINGER_LANDMARKS, MCP_INDICES, LandmarkIndex


# Image-space openness ratio below which we declare the hand 'closed' (a
# fist or near-fist). Computed as mean(tip-to-palm distance) / palm size,
# both in image pixels. A flat open hand reads ~1.6-2.0; a balled fist
# reads ~0.6-0.8. The threshold sits in the gap.
#
# This check lives entirely in 2D image space, so it is immune to the
# 3D depth ambiguity that lets a curled-toward-camera finger masquerade
# as a perfect piano curve in any axis-projected angle.
CLOSED_HAND_OPENNESS = 1.6


class FingerPosture(Enum):
    """Finger posture categories based on PIP joint angle."""
    COLLAPSED = "collapsed"  # Over-curved, angle too acute
    CURVED = "curved"        # Good posture
    NEUTRAL = "neutral"      # Acceptable
    FLAT = "flat"            # Bad posture, too straight


class WristPosition(Enum):
    """Wrist height categories relative to MCP line."""
    LOW = "low"
    NEUTRAL = "neutral"
    HIGH = "high"


class HandArchStatus(Enum):
    """Hand arch categories based on palm height."""
    COLLAPSED = "collapsed"
    GOOD = "good"
    EXCESSIVE = "excessive"


@dataclass
class FingerMetrics:
    """Metrics for a single finger."""
    name: str
    pip_angle: float  # Angle at PIP joint in degrees (MCP-PIP-DIP)
    dip_angle: float  # Angle at DIP joint in degrees (PIP-DIP-TIP)
    mcp_angle: Optional[float]  # Sagittal angle at MCP (WRIST-MCP-PIP); None for thumb
    posture: FingerPosture
    # Per-finger knuckle-dome status derived from mcp_angle. ``None`` when
    # mcp_angle is ``None`` (thumb).
    mcp_posture: Optional["HandArchStatus"] = None


@dataclass
class PostureMetrics:
    """Complete posture metrics for a hand."""
    fingers: Dict[str, FingerMetrics]
    wrist_height: float  # Normalized wrist offset from MCP line
    wrist_position: WristPosition
    hand_arch: float  # Normalized palm height
    hand_arch_status: HandArchStatus
    score: float  # Overall posture score 0-100
    # 2D image-space ratio: mean(tip→palm distance) / palm size. ~1.6–2.0
    # for a flat open hand, ~0.6–0.8 for a fist. See CLOSED_HAND_OPENNESS.
    openness: float = 0.0
    # True when openness is below the closed-hand threshold; in this case
    # all non-thumb finger postures are forced to COLLAPSED regardless of
    # what the angle calculations report.
    closed_hand: bool = False


class PostureAnalyzer:
    """
    Analyzes hand posture from landmarks.
    
    Extracts geometric features and classifies posture based on:
    - PIP joint angles for each finger
    - Wrist height relative to MCP line
    - Hand arch (palm height)
    """
    
    # PIP angle thresholds (degrees) -- sagittal (Y-Z) projection from world
    # landmarks for index/middle/ring/pinky; full-3D angle for the thumb.
    # 180° = fully straight; angle decreases as the finger curls.
    # Fitted from labelled snapshots captured 2026-04-28 in stats_log.csv.
    # In good posture: index ~99°, middle ~114°, ring ~111°, pinky ~140°,
    # thumb ~158°. Over-curled examples bottom out at 7-60°.
    PIP_FLAT_THRESHOLD = 70         # < 70°  = over-curled / collapsed
    PIP_NEUTRAL_LOW = 70            # 70-100° = acceptable (slightly over-curled)
    PIP_NEUTRAL_HIGH = 100
    PIP_CURVED_LOW = 100            # 100-160° = good curved posture
    PIP_CURVED_HIGH = 160
    # > 160° = too flat / hyper-extended
    
    # Wrist pitch thresholds (degrees) -- pitch of the wrist->MCP_centroid
    # vector in the world Y-Z plane. Empirical sign on this camera setup
    # (calibrated to labelled rows in stats_log.csv on 2026-04-28):
    #   ~-12°  = good piano arch (knuckles slightly raised),
    #   ~-19°  = wrist labelled "too high",
    #   -24..-33° = "flat / pancake hand",
    #   ~-3°   = approximately level.
    # Note: more negative = wrist higher / hand flatter on this rig.
    WRIST_IDEAL_ANGLE = -12.0       # Ideal: matches labelled "good" cluster
    WRIST_HIGH_THRESHOLD = 0.0      # Above (less negative) = wrist dropped
    WRIST_LOW_THRESHOLD = -20.0     # Below (more negative) = wrist too high / flat
    
    # Hand arch thresholds (degrees) -- now derived from the *MCP joint
    # angle*: the sagittal angle at each knuckle between (wrist->MCP) and
    # (MCP->PIP). We average the four non-thumb fingers. 180° means the
    # proximal phalanx continues straight off the wrist axis (knuckles
    # totally flat / "on knuckles" piano fault). Lower angles mean the
    # finger descends sharply from the knuckle (good piano dome). Very low
    # (<80°) means the hand is over-clenched into a fist.
    ARCH_COLLAPSED_THRESHOLD = 150.0   # mean MCP > 150° = flat / on knuckles
    ARCH_EXCESSIVE_THRESHOLD = 80.0    # mean MCP < 80°  = over-clenched
    
    # DIP "flat fingertip" thresholds (degrees, sagittal-projected). When
    # the DIP joint is nearly straight (~180°) the distal phalanx is in
    # line with the middle phalanx -- either the player is striking with a
    # flat fingertip, or the tip is occluded and MediaPipe defaulted to an
    # extended pose. Either way, it's a piano-posture fault and should be
    # treated as FLAT regardless of how good the PIP angle looks.
    #
    # Per-finger thresholds because the index naturally extends (170-180°
    # in clean labelled "good" rows) and the pinky almost always reads
    # very flat too. Middle/ring give the cleanest signal.
    DIP_FLAT_THRESHOLDS: Dict[str, Optional[float]] = {
        'thumb':  None,    # full-3D angle, separate mechanics
        'index':  None,    # naturally extends in good posture
        'middle': 150.0,   # tightened from 165° -- catches occluded-tip drift
        'ring':   150.0,   # tightened from 165°
        'pinky':  150.0,   # tightened from 178°
    }
    
    def __init__(self):
        """Initialize the posture analyzer."""
        self._reference_length: Optional[float] = None
        self._debug_wrist: Optional[dict] = None
    
    def get_debug_info(self) -> Optional[dict]:
        """Get debug info for visualization."""
        return self._debug_wrist
    
    def _calculate_angle(self, p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
        """Calculate the angle at ``vertex`` formed by ``p1-vertex-p2``.

        Works for any dimensionality (2D or 3D). Returns degrees in [0, 180].
        """
        v1 = p1 - vertex
        v2 = p2 - vertex
        
        # Handle zero-length vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 180.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _curl_angle(
        self, mcp: np.ndarray, pip: np.ndarray, dip: np.ndarray
    ) -> float:
        """Flexion angle at the middle of three joints, in full 3D.

        We previously projected onto the camera's Y-Z (sagittal) plane on
        the assumption that finger curl happens in Y-Z for a front-facing
        camera. That broke catastrophically in fist / pronated-hand poses:
        a finger curled *toward* the camera has its real flexion plane
        rotated away from Y-Z, and the projected angle reads near-90°
        (looking like a perfect piano curve) regardless of the actual
        curl. Using the full 3D angle is rotation-invariant and
        identical to the projection answer when the finger really is
        Y-Z aligned (the hinge keeps all four joints coplanar).

        Inputs must be 3D points (MediaPipe world landmarks, in meters).
        Returns degrees in [0, 180]; 180° = straight, decreasing as the
        finger curls.
        """
        return self._calculate_angle(mcp, pip, dip)
    
    def _calculate_reference_length(self, landmarks: HandLandmarks) -> float:
        """
        Calculate reference length for normalization.
        Uses distance from wrist to middle finger MCP.
        """
        wrist = landmarks.get_landmark(LandmarkIndex.WRIST)
        middle_mcp = landmarks.get_landmark(LandmarkIndex.MIDDLE_MCP)
        return np.linalg.norm(middle_mcp - wrist)
    
    def _classify_finger_posture(self, pip_angle: float) -> FingerPosture:
        """Classify finger posture based on the sagittal PIP angle.

        Angle convention: 180° = fully straight finger; angle *decreases* as
        the finger curls. So small angles mean over-curled, large angles mean
        flat/hyper-extended.
        """
        if pip_angle < self.PIP_FLAT_THRESHOLD:
            # Over-curled -- fingertip tucked under
            return FingerPosture.COLLAPSED
        elif pip_angle < self.PIP_NEUTRAL_HIGH:
            # Slightly over-curled but acceptable
            return FingerPosture.NEUTRAL
        elif pip_angle < self.PIP_CURVED_HIGH:
            # Ideal piano-curved posture
            return FingerPosture.CURVED
        else:
            # Too straight / hyper-extended
            return FingerPosture.FLAT
    
    def _classify_wrist_position(self, wrist_angle: float) -> WristPosition:
        """Classify wrist position based on angle (degrees)."""
        # Note: angles are negative (e.g., -100°)
        # Higher angle (e.g., -95°) = wrist too high
        # Lower angle (e.g., -115°) = wrist too low
        if wrist_angle > self.WRIST_HIGH_THRESHOLD:
            return WristPosition.HIGH
        elif wrist_angle < self.WRIST_LOW_THRESHOLD:
            return WristPosition.LOW
        else:
            return WristPosition.NEUTRAL
    
    def _classify_hand_arch(self, mean_mcp_angle: float) -> HandArchStatus:
        """Classify hand arch from the mean (non-thumb) MCP sagittal angle.

        High angle (close to 180°) => proximal phalanges in line with the
        wrist axis => knuckles flat / "on knuckles" piano fault.
        Low angle => fingers descend sharply from the knuckle => good dome.
        Very low (<80°) => over-clenched.
        """
        if mean_mcp_angle > self.ARCH_COLLAPSED_THRESHOLD:
            return HandArchStatus.COLLAPSED
        elif mean_mcp_angle < self.ARCH_EXCESSIVE_THRESHOLD:
            return HandArchStatus.EXCESSIVE
        else:
            return HandArchStatus.GOOD
    
    def _calculate_finger_metrics(
        self, 
        landmarks: HandLandmarks, 
        finger_name: str
    ) -> FingerMetrics:
        """Calculate metrics for a single finger from world landmarks.

        For index/middle/ring/pinky the PIP and DIP angles are projected
        onto the camera Y-Z (sagittal) plane: this matches the axis along
        which those fingers actually flex and is far more robust than a 3D
        dot-product under monocular depth ambiguity.

        The thumb's flexion axis is nearly orthogonal to the other fingers'
        sagittal plane, so projecting it onto Y-Z would destroy the signal.
        We compute its joint angles in full 3D instead.
        """
        mcp_idx, pip_idx, dip_idx, tip_idx = FINGER_LANDMARKS[finger_name]
        
        mcp = landmarks.get_world_landmark(mcp_idx)
        pip = landmarks.get_world_landmark(pip_idx)
        dip = landmarks.get_world_landmark(dip_idx)
        tip = landmarks.get_world_landmark(tip_idx)
        wrist = landmarks.get_world_wrist()
        
        if finger_name == 'thumb':
            pip_angle = self._calculate_angle(mcp, pip, dip)
            dip_angle = self._calculate_angle(pip, dip, tip)
            # MCP angle is meaningless for the thumb in this framework.
            mcp_angle: Optional[float] = None
        else:
            pip_angle = self._curl_angle(mcp, pip, dip)
            dip_angle = self._curl_angle(pip, dip, tip)
            # MCP joint angle: WRIST -> MCP -> PIP.
            # Detects "on knuckles" (angle ~180°) vs. domed knuckles (~110°).
            mcp_angle = self._curl_angle(wrist, mcp, pip)
        
        posture = self._classify_finger_posture(pip_angle)
        
        # Flat-fingertip / occluded-tip override: a near-straight DIP joint
        # means the distal phalanx is collinear with the middle phalanx,
        # which is a piano-posture fault even if the PIP angle is ideal.
        dip_flat = self.DIP_FLAT_THRESHOLDS.get(finger_name)
        if dip_flat is not None and dip_angle > dip_flat:
            posture = FingerPosture.FLAT
        
        # Per-finger knuckle-dome classification (non-thumb only). Uses the
        # same thresholds as the legacy global mean-MCP arch, so an
        # individual flat knuckle ("on knuckles" on just the pinky, say)
        # is now visible instead of being averaged away.
        mcp_posture: Optional[HandArchStatus] = None
        if mcp_angle is not None:
            mcp_posture = self._classify_hand_arch(mcp_angle)
        
        return FingerMetrics(
            name=finger_name,
            pip_angle=pip_angle,
            dip_angle=dip_angle,
            mcp_angle=mcp_angle,
            posture=posture,
            mcp_posture=mcp_posture,
        )
    
    def _calculate_wrist_angle(self, landmarks: HandLandmarks) -> Tuple[float, float]:
        """Wrist pitch computed in the world Y-Z plane.

        Under a front-facing monocular camera the image-space "wrist angle"
        is dominated by perspective: a flat hand pointing at the lens looks
        identical to a drooped one. Working in world landmarks (metric,
        hand-centred) and in the Y-Z plane gives a genuine pitch of the
        wrist->MCP_centroid vector relative to the camera's optical axis.

        Sign convention (calibrated against 2026-04-28 labelled data):
        good piano arch reads around -12°; flat / "pancake" hand becomes
        more negative (-24° to -33°); approximately level reads near -3°.
        See the threshold block at the top of this class for full
        empirical anchors.

        Returns:
            (raw_angle, deviation_from_ideal)
        """
        wrist_w = landmarks.get_world_wrist()
        mcp_centroid_w = landmarks.get_world_mcp_centroid()
        
        delta_y = mcp_centroid_w[1] - wrist_w[1]
        delta_z = mcp_centroid_w[2] - wrist_w[2]
        # Sign convention is empirically calibrated to the labelled rows in
        # stats_log.csv: with this -delta_y, "good piano arch" reads ~-12°,
        # "flat / pancake hand" reads more negative (-24° to -33°), and
        # "wrist too high" reads at -19°. Don't flip without re-labelling.
        raw_angle = np.degrees(np.arctan2(-delta_y, abs(delta_z) + 1e-9))
        
        deviation = raw_angle - self.WRIST_IDEAL_ANGLE
        
        # Pixel-space points are what the visualiser overlays, so keep them
        # in the debug dict even though the angle itself is world-space.
        self._debug_wrist = {
            'wrist': landmarks.get_wrist(),
            'mcp_centroid': landmarks.get_mcp_centroid(),
            'raw_angle': raw_angle,
            'deviation': deviation,
        }
        
        return raw_angle, deviation
    
    def _calculate_wrist_height(self, landmarks: HandLandmarks) -> Tuple[float, float]:
        """
        Calculate wrist position using angle-based approach.
        Returns (raw_angle, normalized_deviation) for compatibility.
        """
        raw_angle, deviation = self._calculate_wrist_angle(landmarks)
        # Normalize deviation to roughly -1 to 1 range (30° deviation = 1.0)
        normalized = deviation / 30.0
        return raw_angle, normalized
    
    def _calculate_hand_arch(
        self, fingers: Dict[str, FingerMetrics]
    ) -> Tuple[float, float]:
        """Hand arch metric derived from the MCP joint angles.

        Averages the sagittal MCP angle across the four non-thumb fingers
        (the thumb's MCP is not comparable). The mean angle is the "raw"
        arch metric in degrees; we also return it unchanged as the second
        slot to preserve the previous (raw, normalised) tuple shape used
        by ``analyze``.

        Replaces the previous pixel-based palm-height heuristic, which
        depended on a hand-clicked keyboard line and conflated camera
        zoom with actual posture.
        """
        mcp_angles = [
            f.mcp_angle for f in fingers.values()
            if f.mcp_angle is not None
        ]
        if not mcp_angles:
            return 180.0, 180.0  # degenerate -- treat as flat
        mean_mcp = float(np.mean(mcp_angles))
        return mean_mcp, mean_mcp
    
    # Per-finger weight for the knuckle-dome (MCP) channel. Pinky gets
    # double weight because pinky-knuckle collapse is the single most
    # common piano-posture fault. Index/middle/ring are equally weighted.
    # Total over non-thumb fingers = 20 points. Thumb is excluded (MCP
    # angle is meaningless in this framework for the thumb).
    MCP_WEIGHTS: Dict[str, float] = {
        'index':  6.0,
        'middle': 6.0,
        'ring':   6.0,
        'pinky': 12.0,
    }
    
    def _calculate_score(self, metrics: 'PostureMetrics') -> float:
        """Calculate overall posture score (0-100).

        Scoring breakdown:
        - Finger PIP postures:   50 pts  (10 per finger)
        - Wrist pitch:           20 pts
        - Per-finger MCP dome:   30 pts  (6+6+6+12 pinky-weighted; thumb excl.)
        """
        score = 0.0
        
        # Finger PIP scores (10 points each, 50 total).
        finger_scores = {
            FingerPosture.CURVED: 10.0,
            FingerPosture.NEUTRAL: 7.0,
            FingerPosture.COLLAPSED: 3.0,
            FingerPosture.FLAT: 0.0,
        }
        for finger_metrics in metrics.fingers.values():
            score += finger_scores[finger_metrics.posture]
        
        # Wrist score (20 points).
        wrist_scores = {
            WristPosition.NEUTRAL: 20.0,
            WristPosition.HIGH: 8.0,
            WristPosition.LOW: 4.0,
        }
        score += wrist_scores[metrics.wrist_position]
        
        # Per-finger knuckle-dome score (30 points total, pinky weighted).
        # Using fractions (1.0 / 0.4 / 0.0) of each finger's allocated
        # weight so the total sums to 30 in the all-good case.
        mcp_fraction = {
            HandArchStatus.GOOD: 1.0,
            HandArchStatus.EXCESSIVE: 0.4,   # over-clenched on that knuckle
            HandArchStatus.COLLAPSED: 0.0,   # flat knuckle -- hard fail
        }
        for finger_name, weight in self.MCP_WEIGHTS.items():
            fm = metrics.fingers.get(finger_name)
            if fm is None or fm.mcp_posture is None:
                continue
            score += weight * mcp_fraction[fm.mcp_posture]
        
        return score
    
    def _compute_openness(self, landmarks: HandLandmarks) -> float:
        """2D image-space openness: mean(tip→palm-centre) / palm size.

        Pure pixel-space measurement, no depth involved — immune to the
        projection / pronation issues that affect the 3D angle pipeline.
        Used to detect a closed hand (fist) and force a sane low score.
        """
        wrist = landmarks.get_wrist()
        middle_mcp = landmarks.get_landmark(LandmarkIndex.MIDDLE_MCP)
        palm_size = float(np.linalg.norm(middle_mcp - wrist))
        if palm_size < 1e-6:
            return 0.0
        palm_centre = landmarks.get_mcp_centroid()
        tip_indices = [
            LandmarkIndex.INDEX_TIP, LandmarkIndex.MIDDLE_TIP,
            LandmarkIndex.RING_TIP, LandmarkIndex.PINKY_TIP,
        ]
        tip_dists = [
            float(np.linalg.norm(landmarks.get_landmark(i) - palm_centre))
            for i in tip_indices
        ]
        return float(np.mean(tip_dists)) / palm_size
    
    def analyze(self, landmarks: HandLandmarks) -> PostureMetrics:
        """
        Perform full posture analysis on hand landmarks.
        
        Args:
            landmarks: HandLandmarks from hand tracker
            
        Returns:
            PostureMetrics with all measurements and classifications
        """
        # Calculate finger metrics
        fingers = {}
        for finger_name in FINGER_LANDMARKS.keys():
            fingers[finger_name] = self._calculate_finger_metrics(landmarks, finger_name)
        
        # Calculate wrist angle
        raw_wrist_angle, wrist_height = self._calculate_wrist_height(landmarks)
        wrist_position = self._classify_wrist_position(raw_wrist_angle)
        
        # Calculate hand arch from per-finger MCP angles
        _, hand_arch = self._calculate_hand_arch(fingers)
        hand_arch_status = self._classify_hand_arch(hand_arch)
        
        # 2D openness override: if the hand is closed (fist or near-fist),
        # the angle pipeline is unreliable -- force every non-thumb finger
        # to COLLAPSED, both for the PIP posture and for the knuckle dome.
        # This kills the "fist scores 91/100" failure mode at its root.
        openness = self._compute_openness(landmarks)
        closed_hand = openness < CLOSED_HAND_OPENNESS
        if closed_hand:
            for fname, fm in fingers.items():
                fm.posture = FingerPosture.COLLAPSED
                if fname != 'thumb':
                    fm.mcp_posture = HandArchStatus.COLLAPSED
            hand_arch_status = HandArchStatus.COLLAPSED
        
        # Create metrics object (score calculated after)
        metrics = PostureMetrics(
            fingers=fingers,
            wrist_height=wrist_height,
            wrist_position=wrist_position,
            hand_arch=hand_arch,
            hand_arch_status=hand_arch_status,
            score=0.0,
            openness=openness,
            closed_hand=closed_hand,
        )
        
        # Calculate overall score
        metrics.score = self._calculate_score(metrics)
        
        return metrics


class PostureScorer:
    """
    Aggregates posture metrics over time to produce a running score.
    """
    
    def __init__(self, window_size: int = 30):
        """
        Initialize scorer.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.score_history: List[float] = []
        self.metrics_history: List[PostureMetrics] = []
    
    def update(self, metrics: PostureMetrics) -> float:
        """
        Update with new frame metrics.
        
        Args:
            metrics: PostureMetrics for current frame
            
        Returns:
            Running average score
        """
        self.score_history.append(metrics.score)
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.score_history) > self.window_size:
            self.score_history.pop(0)
            self.metrics_history.pop(0)
        
        return self.get_average_score()
    
    def get_average_score(self) -> float:
        """Get average score over the window."""
        if not self.score_history:
            return 0.0
        return np.mean(self.score_history)
    
    def get_finger_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get per-finger posture statistics over the window.
        
        Returns:
            Dict mapping finger name to posture distribution
        """
        if not self.metrics_history:
            return {}
        
        stats = {}
        for finger_name in FINGER_LANDMARKS.keys():
            posture_counts = {p.value: 0 for p in FingerPosture}
            
            for metrics in self.metrics_history:
                posture = metrics.fingers[finger_name].posture
                posture_counts[posture.value] += 1
            
            total = len(self.metrics_history)
            stats[finger_name] = {
                posture: count / total 
                for posture, count in posture_counts.items()
            }
        
        return stats
    
    def reset(self):
        """Clear history."""
        self.score_history.clear()
        self.metrics_history.clear()
