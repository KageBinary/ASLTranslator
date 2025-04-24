import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, static_mode=False, max_hands=1, detection_confidence=0.7, tracking_confidence=0.6):
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
        """Return a normalized flat array of [x, y, z] per landmark with extras for ML."""
        landmarks, _ = self.find_positions(image, hand_index)

        if not landmarks:
            return None

        coords = np.array([[x, y, z] for _, x, y, z in landmarks])

        # Normalize: wrist as origin, scale relative to wrist–middle tip
        origin = coords[0]
        coords -= origin

        scale = np.linalg.norm(coords[9]) or 1.0
        coords /= scale

        # Optional: append fingertip-to-palm and inter-finger distances
        palm = np.mean(coords[[0, 5, 9, 13, 17]], axis=0)
        tips = coords[[4, 8, 12, 16, 20]]

        tip_to_palm = [np.linalg.norm(t - palm) for t in tips]
        inter_finger = [np.linalg.norm(tips[i] - tips[j]) for i in range(5) for j in range(i+1, 5)]

        final_array = np.concatenate([coords.flatten(), tip_to_palm, inter_finger])
        return final_array
