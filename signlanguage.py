import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
import time

class SignLanguageDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.labels = {}
        self.reverse_labels = {}
        
        # Data storage
        self.data = []
        self.target = []
        
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmark coordinates and calculate features"""
        if hand_landmarks is None:
            return np.zeros(63)  # 21 landmarks * 3 coordinates
        
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks)
    
    def calculate_distances(self, landmarks):
        """Calculate distances between key points for better feature representation"""
        if len(landmarks) < 63:
            return np.zeros(20)
        
        # Reshape landmarks to (21, 3) format
        points = landmarks.reshape(21, 3)
        
        # Calculate key distances (thumb to fingers, palm center, etc.)
        distances = []
        
        # Thumb tip to other fingertips
        thumb_tip = points[4]
        finger_tips = [points[8], points[12], points[16], points[20]]  # Index, Middle, Ring, Pinky
        
        for tip in finger_tips:
            dist = np.linalg.norm(thumb_tip - tip)
            distances.append(dist)
        
        # Palm center (approximate)
        palm_center = np.mean(points[0:5], axis=0)
        
        # Fingertips to palm center
        for tip in finger_tips:
            dist = np.linalg.norm(tip - palm_center)
            distances.append(dist)
        
        # Wrist to fingertips
        wrist = points[0]
        for tip in finger_tips:
            dist = np.linalg.norm(wrist - tip)
            distances.append(dist)
        
        # Additional features: finger bending angles (simplified)
        for i in range(4):  # For each finger except thumb
            finger_base = points[5 + i*4]
            finger_tip = points[8 + i*4]
            bend_ratio = np.linalg.norm(finger_tip - palm_center) / np.linalg.norm(finger_base - palm_center)
            distances.append(bend_ratio)
        
        return np.array(distances)
    
    def collect_data(self, sign_name, num_samples=100):
        """Collect training data for a specific sign"""
        print(f"Collecting data for sign: {sign_name}")
        print("Press 's' to start collecting, 'q' to quit, 'n' for next sign")
        
        cap = cv2.VideoCapture(0)
        collected_samples = 0
        collecting = False
        
        while collected_samples < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    if collecting:
                        # Extract features
                        landmarks = self.extract_landmarks(hand_landmarks)
                        distances = self.calculate_distances(landmarks)
                        features = np.concatenate([landmarks, distances])
                        
                        self.data.append(features)
                        self.target.append(sign_name)
                        collected_samples += 1
                        
                        # Visual feedback
                        cv2.putText(frame, f"Collected: {collected_samples}/{num_samples}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Instructions
            cv2.putText(frame, f"Sign: {sign_name}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, "Press 's' to start collecting" if not collecting else "Collecting...", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not collecting:
                collecting = True
                print("Started collecting...")
            elif key == ord('q'):
                break
            elif key == ord('n'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Collected {collected_samples} samples for {sign_name}")
    
    def train_model(self):
        """Train the classification model"""
        if len(self.data) == 0:
            print("No data collected yet!")
            return
        
        # Convert to numpy arrays
        X = np.array(self.data)
        y = np.array(self.target)
        
        # Create label mapping
        unique_labels = list(set(y))
        self.labels = {label: i for i, label in enumerate(unique_labels)}
        self.reverse_labels = {i: label for label, i in self.labels.items()}
        
        # Convert labels to integers
        y_encoded = [self.labels[label] for label in y]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2f}")
        
        # Show class distribution
        print("Class distribution:")
        for label, count in Counter(y).items():
            print(f"  {label}: {count} samples")
        
        self.is_trained = True
    
    def save_model(self, filename="sign_language_model.pkl"):
        """Save the trained model and labels"""
        if not self.is_trained:
            print("Model not trained yet!")
            return
        
        model_data = {
            'model': self.model,
            'labels': self.labels,
            'reverse_labels': self.reverse_labels
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename="sign_language_model.pkl"):
        """Load a pre-trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.labels = model_data['labels']
            self.reverse_labels = model_data['reverse_labels']
            self.is_trained = True
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found!")
            return False
    
    def predict_sign(self, hand_landmarks):
        """Predict sign from hand landmarks"""
        if not self.is_trained:
            return "Model not trained", 0.0
        
        landmarks = self.extract_landmarks(hand_landmarks)
        distances = self.calculate_distances(landmarks)
        features = np.concatenate([landmarks, distances]).reshape(1, -1)
        
        # Get prediction and confidence
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        predicted_sign = self.reverse_labels[prediction]
        return predicted_sign, confidence
    
    def real_time_detection(self):
        """Real-time sign language detection"""
        if not self.is_trained:
            print("Please train the model first!")
            return
        
        cap = cv2.VideoCapture(0)
        prediction_history = []
        
        print("Starting real-time detection. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            prediction_text = "No hand detected"
            confidence_text = ""
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Predict sign
                    predicted_sign, confidence = self.predict_sign(hand_landmarks)
                    
                    # Smooth predictions using history
                    if confidence > 0.7:  # Only consider high-confidence predictions
                        prediction_history.append(predicted_sign)
                        if len(prediction_history) > 10:
                            prediction_history.pop(0)
                        
                        # Get most common prediction
                        if prediction_history:
                            most_common = Counter(prediction_history).most_common(1)[0]
                            prediction_text = most_common[0]
                            confidence_text = f"Confidence: {confidence:.2f}"
            
            # Display results
            cv2.putText(frame, f"Sign: {prediction_text}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Sign Language Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = SignLanguageDetector()
    
    print("Sign Language Gesture Detection Program")
    print("=====================================")
    
    # Try to load existing model
    if detector.load_model():
        print("Loaded existing model!")
    else:
        print("No existing model found. You'll need to collect data and train.")
    
    while True:
        print("\nOptions:")
        print("1. Collect training data")
        print("2. Train model")
        print("3. Save model")
        print("4. Load model")
        print("5. Start real-time detection")
        print("6. Quit")
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            signs_to_collect = []
            print("Enter sign names to collect (press enter with empty input to finish):")
            while True:
                sign = input("Sign name: ").strip()
                if not sign:
                    break
                signs_to_collect.append(sign)
            
            for sign in signs_to_collect:
                samples = int(input(f"Number of samples for '{sign}' (default 100): ") or "100")
                detector.collect_data(sign, samples)
        
        elif choice == '2':
            detector.train_model()
        
        elif choice == '3':
            filename = input("Enter filename (default: sign_language_model.pkl): ").strip()
            if not filename:
                filename = "sign_language_model.pkl"
            detector.save_model(filename)
        
        elif choice == '4':
            filename = input("Enter filename (default: sign_language_model.pkl): ").strip()
            if not filename:
                filename = "sign_language_model.pkl"
            detector.load_model(filename)
        
        elif choice == '5':
            detector.real_time_detection()
        
        elif choice == '6':
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()