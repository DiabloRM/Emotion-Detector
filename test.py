import cv2
import numpy as np
from deepface import DeepFace
import datetime
import time
import threading
from collections import deque

# Configuration parameters
FRAME_SKIP = 2  # Process every nth frame for emotion detection
ANALYSIS_RESOLUTION = 0.5  # Scale factor for analysis (smaller = faster)
FACE_DETECTION_COLOR = (0, 255, 255)  # Yellow color for face detection box
UI_THEME = {
    'background': (40, 44, 52),  # Dark background
    'text': (255, 255, 255),  # White text
    'accent': (0, 120, 212),  # Blue accent
    'border': (75, 75, 75),  # Gray border
    'emotion_colors': {
        'angry': (0, 0, 255),     # Red (BGR)
        'disgust': (0, 140, 255), # Orange
        'fear': (0, 0, 128),      # Dark red
        'happy': (0, 255, 0),     # Green
        'sad': (255, 0, 0),       # Blue
        'surprise': (255, 255, 0), # Cyan
        'neutral': (255, 255, 255) # White
    }
}

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get webcam properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize variables for emotion detection
emotion_results = None
processing_frame = False
frame_count = 0
fps_history = deque(maxlen=30)  # Store last 30 frame times for FPS calculation
last_frame_time = time.time()

# Function to analyze emotions in a separate thread
def analyze_emotion(frame):
    global emotion_results, processing_frame
    try:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=ANALYSIS_RESOLUTION, fy=ANALYSIS_RESOLUTION)
        result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False)
        emotion_results = result[0]
        
        # Extract face location information if available
        if 'region' in emotion_results:
            region = emotion_results['region']
            # Convert coordinates back to original frame size
            x = int(region['x'] / ANALYSIS_RESOLUTION)
            y = int(region['y'] / ANALYSIS_RESOLUTION)
            w = int(region['w'] / ANALYSIS_RESOLUTION)
            h = int(region['h'] / ANALYSIS_RESOLUTION)
            emotion_results['face_box'] = (x, y, w, h)
    except Exception as e:
        emotion_results = None
    finally:
        processing_frame = False

# Function to draw a modern UI border
def draw_border(frame):
    # Add a subtle border gradient
    height, width = frame.shape[:2]
    border_size = 5
    # Draw border
    cv2.rectangle(frame, (0, 0), (width, height), UI_THEME['border'], border_size)
    return frame

# Function to draw face detection box
def draw_face_box(frame, face_box, emotion=None):
    if not face_box:
        return frame
    
    x, y, w, h = face_box
    
    # Choose color based on emotion if available
    if emotion and emotion in UI_THEME['emotion_colors']:
        box_color = UI_THEME['emotion_colors'][emotion]
    else:
        box_color = FACE_DETECTION_COLOR
    
    # Draw rectangle around face
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
    
    return frame

# Function to create emotion indicator bar
def draw_emotion_indicator(frame, emotion_scores):
    if not emotion_scores:
        return frame
    
    # Create a horizontal bar chart for emotions
    bar_start_x = 20
    bar_start_y = frame_height - 100
    bar_width = 30
    bar_gap = 10
    max_bar_height = 80
    
    # Draw background for emotion bars
    cv2.rectangle(frame, 
                 (bar_start_x - 10, bar_start_y - max_bar_height - 10),
                 (bar_start_x + 7 * (bar_width + bar_gap), bar_start_y + 30),
                 (0, 0, 0, 180), -1)
    
    # Draw each emotion bar
    for i, (emotion, score) in enumerate(emotion_scores.items()):
        # Calculate bar height based on score
        bar_height = int((score / 100) * max_bar_height)
        
        # Draw the bar
        bar_x = bar_start_x + i * (bar_width + bar_gap)
        color = UI_THEME['emotion_colors'].get(emotion, UI_THEME['text'])
        
        # Draw bar
        cv2.rectangle(frame,
                     (bar_x, bar_start_y - bar_height),
                     (bar_x + bar_width, bar_start_y),
                     color, -1)
        
        # Draw emotion label
        cv2.putText(frame, emotion[:3].upper(),
                   (bar_x, bar_start_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_THEME['text'], 1)
        
        # Draw score on top of bar
        cv2.putText(frame, f"{score:.0f}%",
                   (bar_x, bar_start_y - bar_height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_THEME['text'], 1)
    
    return frame

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate FPS
    current_time = time.time()
    fps_history.append(1 / (current_time - last_frame_time))
    fps = sum(fps_history) / len(fps_history)
    last_frame_time = current_time
    
    # Add border to frame
    frame = draw_border(frame)
    
    # Process every nth frame for emotion detection
    if frame_count % FRAME_SKIP == 0 and not processing_frame:
        processing_frame = True
        threading.Thread(target=analyze_emotion, args=(frame.copy(),), daemon=True).start()
    
    frame_count += 1
    
    # Create status panel background (semi-transparent overlay)
    status_panel = frame.copy()
    overlay = np.zeros_like(frame)
    cv2.rectangle(overlay, (10, 10), (350, 140), UI_THEME['background'], -1)
    cv2.addWeighted(overlay, 0.7, status_panel, 0.3, 0, status_panel)
    cv2.rectangle(status_panel, (10, 10), (350, 140), UI_THEME['accent'], 2)
    
    # Display time and FPS
    current_time_str = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(status_panel, f'Time: {current_time_str}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_THEME['text'], 2)
    cv2.putText(status_panel, f'FPS: {fps:.1f}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_THEME['text'], 2)
    
    # Display emotion information if available
    if emotion_results:
        dominant_emotion = emotion_results['dominant_emotion']
        emotion_scores = emotion_results['emotion']
        
        # Display dominant emotion with color matching the emotion
        emotion_color = UI_THEME['emotion_colors'].get(dominant_emotion, UI_THEME['text'])
        cv2.putText(status_panel, f'Emotion: {dominant_emotion.upper()}', (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, emotion_color, 2)
        
        # Draw emotion indicator bars
        status_panel = draw_emotion_indicator(status_panel, emotion_scores)
    else:
        # Display message when no face is detected
        cv2.putText(status_panel, 'No face detected', (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Combine the status panel with the main frame
    frame_with_status = status_panel
    
    # Draw face detection box if face location is available
    if emotion_results and 'face_box' in emotion_results:
        frame_with_status = draw_face_box(frame_with_status, emotion_results['face_box'], emotion_results['dominant_emotion'])
    
    # Show the frame
    cv2.imshow('Emotion Detection - Press Q to Quit', frame_with_status)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
