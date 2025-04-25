import cv2
import numpy as np
import pandas as pd
import os
import time
import string
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.hand_detector import HandDetector

def collect_static_data():
    """
    Collect static ASL sign data using a webcam.
    
    This script captures hand landmarks for ASL letters (A-Z except J and Z)
    and saves the data to a CSV file for training a recognition model.
    """
    # Initialize the hand detector
    detector = HandDetector(max_hands=1, detection_confidence=0.7)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Create data directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Get list of ASL signs to collect (letters A-Z except J and Z which require movement)
    signs = [letter for letter in list(string.ascii_lowercase)[:26] if letter not in ['j', 'z']]
    print(f"Will collect data for {len(signs)} signs: {', '.join(signs).upper()}")
    
    # Initialize empty list to store all data
    all_data = []
    
    # Get feature count to ensure consistency
    # First read a test frame to get landmark count
    success, test_img = cap.read()
    if success:
        test_img = cv2.flip(test_img, 1)
        _, _ = detector.find_hands(test_img)
        landmarks_array = detector.get_landmark_array(test_img)
        if landmarks_array is not None:
            feature_count = len(landmarks_array)
            print(f"Detected {feature_count} features from hand landmarks")
        else:
            feature_count = 91  # Default if detection fails
            print(f"Could not detect hand features, using default count: {feature_count}")
    else:
        feature_count = 91  # Default if camera fails
        print("Camera read failed, using default feature count: 91")
    
    # Create column names for features
    column_names = [f'feature_{i}' for i in range(feature_count)]
    column_names.append('label')
    
    # Initialize index to label mapping
    sign_to_label = {sign: idx for idx, sign in enumerate(signs)}
    
    # Number of samples to collect per sign
    base_samples = 75
    
    # Extra samples for commonly confused letters
    problem_letters = ['a', 's', 'n', 'm', 't', 'e', 'c', 'o', 'p']
    extra_samples = {letter: 25 for letter in problem_letters}
    
    # Track current sign index
    current_sign_index = 0
    
    # Continue until all signs are processed
    while current_sign_index < len(signs):
        sign = signs[current_sign_index]
        
        # Set sample count for this sign
        num_samples = base_samples + extra_samples.get(sign, 0)
        collected_samples = 0
        
        # Define reference image path for this sign
        reference_path = f"reference/asl_{sign}.jpg"
        reference_img = None
        
        # Try to load reference image if available
        if os.path.exists(reference_path):
            reference_img = cv2.imread(reference_path)
            # Resize to a small thumbnail
            if reference_img is not None:
                reference_img = cv2.resize(reference_img, (200, 200))
        
        print(f"\nCollecting samples for '{sign.upper()}' ({num_samples} samples)")
        print("Press 's' to start collecting. Press 'p' to pause. Press 'r' to restart letter. Press 'q' to quit.")
        print("Press 'n' to skip to the next letter.")
        
        # Guidance for this sign
        guidance = get_sign_guidance(sign)
        
        collecting = False
        paused = False
        
        while collected_samples < num_samples:
            # Read frame from webcam
            success, img = cap.read()
            if not success:
                print("Failed to capture image from camera")
                break
                
            # Flip the image horizontally for a more intuitive experience
            img = cv2.flip(img, 1)
            
            # Find hands in the image
            img, _ = detector.find_hands(img)
            
            # Create a copy for display
            display_img = img.copy()
            
            # Display sign instructions and progress
            cv2.rectangle(display_img, (0, 0), (display_img.shape[1], 90), (0, 0, 0), cv2.FILLED)
            cv2.putText(display_img, f"Sign: {sign.upper()} ({collected_samples}/{num_samples}) | Letter {current_sign_index+1}/{len(signs)}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, guidance, 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Show reference image if available
            if reference_img is not None:
                h, w = reference_img.shape[:2]
                display_img[90:90+h, 20:20+w] = reference_img
            
            # Status indicator
            if collecting and not paused:
                cv2.putText(display_img, "RECORDING", (display_img.shape[1]-200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Get hand landmarks as a flat array
                landmarks_array = detector.get_landmark_array(img)
                
                if landmarks_array is not None:
                    # If landmark detection was successful, add the sample
                    if len(landmarks_array) == feature_count:
                        # Add label to the data
                        sample = np.append(landmarks_array, sign_to_label[sign])
                        all_data.append(sample)
                        collected_samples += 1
                        
                        # Show visual feedback for successful sample
                        cv2.circle(display_img, (display_img.shape[1]-30, 30), 10, (0, 255, 0), -1)
                        
                        # Small delay to avoid duplicate frames
                        time.sleep(0.1)
                    else:
                        print(f"Warning: Feature count mismatch - got {len(landmarks_array)}, expected {feature_count}")
            elif paused:
                cv2.putText(display_img, "PAUSED", (display_img.shape[1]-200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Instructions at the bottom
            cv2.rectangle(display_img, (0, display_img.shape[0]-40), 
                        (display_img.shape[1], display_img.shape[0]), (0, 0, 0), cv2.FILLED)
            cv2.putText(display_img, "s: Start/Resume | p: Pause | r: Restart letter | n: Next letter | q: Quit", 
                       (20, display_img.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow("ASL Static Data Collection", display_img)
            
            # Key handling
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("Quitting data collection")
                # Save current progress before quitting
                save_progress(all_data, column_names, feature_count, signs)
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('s'):
                collecting = True
                paused = False
                print("Started/Resumed collecting...")
            elif key == ord('p'):
                paused = True
                print("Paused collection")
            elif key == ord('r'):
                # Restart this letter
                collecting = False
                paused = False
                collected_samples = 0
                all_data = [d for d in all_data if int(d[-1]) != sign_to_label[sign]]  # Remove samples for current sign
                print(f"Restarting collection for letter '{sign}'")
            elif key == ord('n'):
                print(f"Skipping to next letter. Collected {collected_samples}/{num_samples} for '{sign}'")
                break
            
            # If we've collected enough samples, move to the next sign
            if collected_samples >= num_samples:
                collecting = False
                print(f"✓ Completed {collected_samples} samples for '{sign.upper()}'")
                break
        
        # Move to the next sign
        current_sign_index += 1
        
        # Save interim progress after each letter
        if current_sign_index % 3 == 0 or current_sign_index == len(signs):
            save_progress(all_data, column_names, feature_count, signs)
    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Final save
    save_progress(all_data, column_names, feature_count, signs)

def save_progress(all_data, column_names, feature_count, signs):
    """Save current progress to CSV file"""
    if not all_data:
        print("No data to save.")
        return
    
    # Convert the list of arrays to a DataFrame
    df = pd.DataFrame(all_data, columns=column_names)
    
    # Check for consistent feature count
    expected_columns = feature_count + 1  # add one for label
    if len(df.columns) != expected_columns:
        print(f"WARNING: DataFrame has {len(df.columns)} columns, expected {expected_columns}")
        print("This may indicate inconsistent feature extraction")
    
    # Save to CSV
    csv_path = 'data/processed/training_data_letters.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} samples to {csv_path}")
    
    # Print letter statistics
    print("\nCurrent data collection statistics:")
    label_counts = df['label'].value_counts().sort_index()
    total_samples = len(df)
    
    print("\nSamples per letter:")
    for label, count in label_counts.items():
        label_index = int(label)
        if label_index < len(signs):
            letter = signs[label_index].upper()
            percentage = (count / total_samples) * 100
            print(f"  {letter}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nTotal: {total_samples} samples for {len(label_counts)} letters")

def get_sign_guidance(letter):
    """Provides guidance for properly forming each ASL sign"""
    guidance = {
        'a': "Make a fist with thumb alongside fingers",
        'b': "Hold hand up, palm facing forward, fingers together and extended",
        'c': "Curve hand in C shape",
        'd': "Make 'D' shape with thumb and index finger, other fingers curled",
        'e': "Curl fingers to palm",
        'f': "Connect thumb and index finger in circle, other fingers up",
        'g': "Point index finger out, thumb parallel",
        'h': "Extend index and middle finger together, parallel to thumb",
        'i': "Make a fist with pinky extended",
        'k': "Index and middle finger up in V shape, thumb touches middle finger",
        'l': "Extend thumb and index finger in L shape",
        'm': "Place thumb between curled fingers",
        'n': "Place thumb between middle and index finger",
        'o': "Form a circle/O with all fingers",
        'p': "Point index down, thumb out to side",
        'q': "Point index down, thumb to side",
        'r': "Cross index and middle finger",
        's': "Make a fist with thumb over fingers",
        't': "Make a fist with thumb between index and middle finger",
        'u': "Extend index and middle finger together",
        'v': "Extend index and middle finger in V",
        'w': "Extend thumb, index, middle fingers in W shape",
        'x': "Curl index finger to touch thumb tip",
        'y': "Extend thumb and pinky, curl other fingers"
    }
    
    return guidance.get(letter, "Form the ASL sign as shown in the reference")

if __name__ == "__main__":
    print("=== ASL Static Data Collection ===")
    print("This script will collect hand landmark data for static ASL signs")
    print("Ensure your hand is well-lit and centered in the frame")
    input("Press Enter to begin...")
    collect_static_data()