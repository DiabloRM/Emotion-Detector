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
    'background': (25, 25, 35),  # Darker modern background
    'panel_bg': (45, 50, 65),    # Panel background
    'text': (255, 255, 255),     # White text
    'text_secondary': (180, 180, 180),  # Secondary text
    'accent': (64, 158, 255),    # Modern blue accent
    'accent_secondary': (100, 200, 255),  # Light blue
    'border': (85, 85, 95),      # Subtle border
    'success': (46, 204, 113),   # Green for positive emotions
    'warning': (241, 196, 15),   # Yellow for neutral
    'danger': (231, 76, 60),     # Red for negative emotions
    'emotion_colors': {
        'angry': (60, 76, 231),      # Modern red (BGR)
        'disgust': (15, 140, 241),   # Orange
        'fear': (128, 60, 180),      # Purple
        'happy': (113, 204, 46),     # Green
        'sad': (231, 76, 60),        # Red
        'surprise': (255, 196, 15),  # Yellow
        'neutral': (180, 180, 180)   # Gray
    },
    'emotion_icons': {
        'angry': '😠', 'disgust': '🤢', 'fear': '😨',
        'happy': '😊', 'sad': '😢', 'surprise': '😲', 'neutral': '😐'
    }
}

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set webcam properties to improve brightness
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Increase brightness (default is usually 100)
cap.set(cv2.CAP_PROP_CONTRAST, 150)    # Increase contrast slightly
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto exposure

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

# Function to draw modern UI elements
def draw_modern_border(frame):
    """Draw a modern gradient border with rounded corners effect"""
    height, width = frame.shape[:2]
    border_size = 3
    
    # Create gradient effect by drawing multiple rectangles
    for i in range(border_size):
        alpha = 1.0 - (i / border_size) * 0.7
        color = tuple(int(c * alpha) for c in UI_THEME['accent'])
        cv2.rectangle(frame, (i, i), (width-i-1, height-i-1), color, 1)
    
    return frame

def draw_rounded_rectangle(frame, pt1, pt2, color, thickness=1, radius=10):
    """Draw a rectangle with rounded corners"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw main rectangle
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw corner circles
    cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, thickness)
    
    return frame

# Function to draw enhanced face detection box
def draw_face_box(frame, face_box, emotion=None, confidence=None):
    if not face_box:
        return frame
    
    x, y, w, h = face_box
    
    # Choose color based on emotion if available
    if emotion and emotion in UI_THEME['emotion_colors']:
        box_color = UI_THEME['emotion_colors'][emotion]
    else:
        box_color = FACE_DETECTION_COLOR
    
    # Draw animated corner brackets instead of full rectangle
    corner_length = min(w, h) // 6
    thickness = 3
    
    # Top-left corner
    cv2.line(frame, (x, y), (x + corner_length, y), box_color, thickness)
    cv2.line(frame, (x, y), (x, y + corner_length), box_color, thickness)
    
    # Top-right corner
    cv2.line(frame, (x + w, y), (x + w - corner_length, y), box_color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_length), box_color, thickness)
    
    # Bottom-left corner
    cv2.line(frame, (x, y + h), (x + corner_length, y + h), box_color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_length), box_color, thickness)
    
    # Bottom-right corner
    cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), box_color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), box_color, thickness)
    
    # Add emotion label above the face box
    if emotion:
        label_text = f"{emotion.upper()}"
        if emotion in UI_THEME['emotion_icons']:
            label_text = f"{UI_THEME['emotion_icons'][emotion]} {label_text}"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw background for text
        label_bg_start = (x, y - text_height - 15)
        label_bg_end = (x + text_width + 10, y - 5)
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, label_bg_start, label_bg_end, box_color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw text
        cv2.putText(frame, label_text, (x + 5, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_THEME['text'], 2)
    
    return frame

# Function to create modern emotion indicator
def draw_emotion_indicator(frame, emotion_scores):
    if not emotion_scores:
        return frame
    
    # Modern circular emotion indicators
    indicator_start_x = 30
    indicator_start_y = frame_height - 120
    circle_radius = 25
    circle_gap = 65
    
    # Draw background panel with rounded corners
    panel_width = len(emotion_scores) * circle_gap + 40
    panel_height = 100
    
    # Create semi-transparent background panel
    overlay = frame.copy()
    draw_rounded_rectangle(overlay, 
                          (indicator_start_x - 20, indicator_start_y - 40),
                          (indicator_start_x + panel_width, indicator_start_y + panel_height - 20),
                          UI_THEME['panel_bg'], -1, 15)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Border removed for cleaner look
    
    # Sort emotions by score for better visualization
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Draw each emotion as a circular indicator
    for i, (emotion, score) in enumerate(sorted_emotions):
        center_x = indicator_start_x + i * circle_gap
        center_y = indicator_start_y
        
        color = UI_THEME['emotion_colors'].get(emotion, UI_THEME['text'])
        
        # Draw background circle
        cv2.circle(frame, (center_x, center_y), circle_radius + 2, UI_THEME['border'], -1)
        
        # Draw progress circle based on score
        progress_radius = int((score / 100) * circle_radius)
        cv2.circle(frame, (center_x, center_y), progress_radius, color, -1)
        
        # Draw outer ring
        cv2.circle(frame, (center_x, center_y), circle_radius, color, 2)
        
        # Add emotion icon in center
        if emotion in UI_THEME['emotion_icons']:
            icon = UI_THEME['emotion_icons'][emotion]
            # For emoji display, we'll use text representation
            cv2.putText(frame, emotion[:3].upper(),
                       (center_x - 15, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_THEME['text'], 1)
        
        # Draw emotion label below circle
        cv2.putText(frame, emotion.capitalize(),
                   (center_x - 20, center_y + circle_radius + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_THEME['text'], 1)
        
        # Draw score percentage
        cv2.putText(frame, f"{score:.0f}%",
                   (center_x - 12, center_y + circle_radius + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, UI_THEME['text_secondary'], 1)
    
    return frame

def draw_status_panel(frame, current_time_str, fps, emotion_results):
    """Draw an enhanced status panel with modern design"""
    # Enhanced status panel with gradient background
    panel_width = 380
    panel_height = 160
    
    # Create gradient background
    overlay = frame.copy()
    
    # Draw main panel background
    draw_rounded_rectangle(overlay, (15, 15), (panel_width, panel_height), 
                          UI_THEME['panel_bg'], -1, 20)
    
    # Add gradient effect
    for i in range(5):
        alpha = 0.1 + (i * 0.05)
        gradient_color = tuple(int(c * (1 + alpha)) for c in UI_THEME['panel_bg'])
        gradient_color = tuple(min(255, c) for c in gradient_color)
        draw_rounded_rectangle(overlay, (15 + i, 15 + i), 
                              (panel_width - i, panel_height - i), 
                              gradient_color, 1, 20 - i)
    
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    
    # Border removed for cleaner look
    
    # Add header with icon
    cv2.putText(frame, '🎭 EMOTION DETECTOR', (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, UI_THEME['accent'], 2)
    
    # Draw separator line
    cv2.line(frame, (30, 55), (panel_width - 30, 55), UI_THEME['border'], 1)
    
    # Display time with icon
    cv2.putText(frame, f'🕐 Time: {current_time_str}', (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_THEME['text'], 1)
    
    # Display FPS with performance indicator
    fps_color = UI_THEME['success'] if fps > 25 else UI_THEME['warning'] if fps > 15 else UI_THEME['danger']
    cv2.putText(frame, f'⚡ FPS: {fps:.1f}', (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 1)
    
    # Display emotion information if available
    if emotion_results:
        dominant_emotion = emotion_results['dominant_emotion']
        emotion_scores = emotion_results['emotion']
        confidence = emotion_scores[dominant_emotion]
        
        # Display dominant emotion with enhanced styling
        emotion_color = UI_THEME['emotion_colors'].get(dominant_emotion, UI_THEME['text'])
        icon = UI_THEME['emotion_icons'].get(dominant_emotion, '🤔')
        
        cv2.putText(frame, f'{icon} Emotion: {dominant_emotion.upper()}', (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        
        # Add confidence indicator
        cv2.putText(frame, f'Confidence: {confidence:.1f}%', (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_THEME['text_secondary'], 1)
    else:
        # Enhanced no face detected message
        cv2.putText(frame, '👤 No face detected', (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, UI_THEME['warning'], 2)
        cv2.putText(frame, 'Please position your face in view', (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, UI_THEME['text_secondary'], 1)
    
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
    
    # Add modern border to frame
    frame = draw_modern_border(frame)
    
    # Process every nth frame for emotion detection
    if frame_count % FRAME_SKIP == 0 and not processing_frame:
        processing_frame = True
        threading.Thread(target=analyze_emotion, args=(frame.copy(),), daemon=True).start()
    
    frame_count += 1
    
    # Get current time
    current_time_str = datetime.datetime.now().strftime("%H:%M:%S")
    
    # Draw enhanced status panel
    frame_with_status = draw_status_panel(frame, current_time_str, fps, emotion_results)
    
    # Draw emotion indicators if available
    if emotion_results:
        emotion_scores = emotion_results['emotion']
        frame_with_status = draw_emotion_indicator(frame_with_status, emotion_scores)
    
    # Draw face detection box if face location is available
    if emotion_results and 'face_box' in emotion_results:
        frame_with_status = draw_face_box(frame_with_status, emotion_results['face_box'], 
                                        emotion_results['dominant_emotion'])
    
    # Show the frame
    cv2.imshow('Emotion Detection - Press Q to Quit', frame_with_status)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
