import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, static_mode=False, max_hands=1, detection_confidence=0.7, tracking_confidence=0.6):
                #Initialize MediaPipe hand detection model with specified parameters
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, image, draw=True):
        """Find hands in the image, return processed image and detection results."""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return image, self.results

    def find_positions(self, image, hand_index=0):
        """Return a list of (id, x, y, z) and dict[id] = (x, y, z) for landmarks."""
        img_h, img_w, _ = image.shape
        landmark_list = []
        landmark_dict = {}

        if self.results.multi_hand_landmarks and hand_index < len(self.results.multi_hand_landmarks):
            hand = self.results.multi_hand_landmarks[hand_index]
            for i, lm in enumerate(hand.landmark):
                x, y, z = int(lm.x * img_w), int(lm.y * img_h), lm.z
                landmark_list.append([i, x, y, z])
                landmark_dict[i] = (x, y, z)

        return landmark_list, landmark_dict

    def get_landmark_array(self, image, hand_index=0):
        """Return a normalized flat array of [x, y, z] per landmark with extras for ML (91 features total)."""
        landmarks, _ = self.find_positions(image, hand_index)

        if not landmarks:
            return None

        coords = np.array([[x, y, z] for _, x, y, z in landmarks])

        # Normalize: wrist as origin, scale relative to wrist–middle tip
        origin = coords[0]
        coords -= origin

        scale = np.linalg.norm(coords[9]) or 1.0
        coords /= scale

        # Fingertip-to-palm distances
        palm = np.mean(coords[[0, 5, 9, 13, 17]], axis=0)
        tips = coords[[4, 8, 12, 16, 20]]

        tip_to_palm = [np.linalg.norm(t - palm) for t in tips]

        # Inter-finger distances
        inter_finger = [np.linalg.norm(tips[i] - tips[j]) for i in range(5) for j in range(i+1, 5)]

        # 🛠️ Correct joint angle calculation for 2 angles per finger
        finger_joints = [
            [1, 2, 3, 4],    # Thumb
            [5, 6, 7, 8],    # Index
            [9, 10, 11, 12], # Middle
            [13, 14, 15, 16],# Ring
            [17, 18, 19, 20] # Pinky
        ]

        joint_angles = []
        for finger in finger_joints:
            for i in range(len(finger) - 2):   # Two angles per finger
                p1, p2, p3 = coords[finger[i]], coords[finger[i+1]], coords[finger[i+2]]
                v1 = p1 - p2
                v2 = p3 - p2
                dot = np.dot(v1, v2)
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                angle = np.arccos(np.clip(dot / norm, -1.0, 1.0)) if norm > 0 else 0.0
                joint_angles.append(angle)

        # Add palm normal vector (3 features) to reach 91
        if len(coords) > 17:
            v1 = coords[5] - coords[0]
            v2 = coords[17] - coords[0]
            palm_normal = np.cross(v1, v2)
            if np.linalg.norm(palm_normal) > 0:
                palm_normal = palm_normal / np.linalg.norm(palm_normal)
            else:
                palm_normal = np.zeros(3)
        else:
            palm_normal = np.zeros(3)

        # Now combine everything
        final_array = np.concatenate([
            coords.flatten(), 
            np.array(tip_to_palm),
            np.array(inter_finger),
            np.array(joint_angles),
            palm_normal
        ])

        return np.nan_to_num(final_array)

