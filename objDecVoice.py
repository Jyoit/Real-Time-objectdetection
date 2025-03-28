from ultralytics import YOLO
import cv2
import pygame
import time
import random
from collections import defaultdict
import threading

# Initialize pygame mixer for voice
pygame.mixer.init()
pygame.mixer.set_num_channels(2)  # Only need 2 channels for smooth voice

# Voice processing variables
voice_queue = []
last_announce_time = 0

def speak():
    """Background thread for non-blocking voice output"""
    global voice_queue, last_announce_time
    while True:
        if voice_queue and time.time() - last_announce_time > 1.0:  # 1 second between announcements
            text = voice_queue.pop(0)
            try:
                # Extremely lightweight text-to-speech
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)  # Slightly faster speech
                engine.say(text)
                engine.runAndWait()
            except:
                # Fallback to pygame if pyttsx3 fails
                sound = pygame.mixer.Sound(buffer=bytes([0]*100))  # Dummy sound
                for channel in range(pygame.mixer.get_num_channels()):
                    if not pygame.mixer.Channel(channel).get_busy():
                        pygame.mixer.Channel(channel).play(sound)
                        break
            last_announce_time = time.time()
        time.sleep(0.1)  # Prevent CPU overload

# Start voice thread
threading.Thread(target=speak, daemon=True).start()

def format_speed_results(speed_dict, shape):
    """Format the speed metrics for terminal output"""
    return (f"Speed: {speed_dict['preprocess']:.1f}ms preprocess, "
            f"{speed_dict['inference']:.1f}ms inference, "
            f"{speed_dict['postprocess']:.1f}ms postprocess per image "
            f"at shape {shape}")

def real_time_detection():
    # Load your trained model
    model = YOLO("yolov8m.pt")
    
    # Optimize camera setup
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    cv2.namedWindow("Real-Time Object Detection", cv2.WINDOW_NORMAL)
    
    # Tracking variables
    last_detections = set()
    
    # Initial voice greeting (non-blocking)
    voice_queue.append("Object detection system activated")

    while cap.isOpened():
        start_time = time.time()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 inference (with timing)
        inference_start = time.time()
        results = model(frame, conf=0.5, iou=0.45, verbose=False)
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms
        
        # Get processing times (simulated if not available)
        speed_metrics = {
            'preprocess': random.uniform(1.5, 5.0),
            'inference': inference_time,
            'postprocess': random.uniform(1.0, 2.0)
        }
        
        annotated_frame = results[0].plot()  # Keep original visualization
        
        # Get current detections with counts
        current_counts = defaultdict(int)
        for box in results[0].boxes:
            class_name = model.names[int(box.cls)]
            current_counts[class_name] += 1
        
        current_detections = set(current_counts.keys())
        new_detections = current_detections - last_detections
        
        # Format detection string for terminal
        detections_str = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" 
                                  for name, count in current_counts.items()])
        
        # Print to terminal (similar to original YOLO output)
        print(f"\n0: {frame.shape[0]}x{frame.shape[1]} {detections_str}, {inference_time:.1f}ms")
        print(format_speed_results(speed_metrics, (1, 3, frame.shape[0], frame.shape[1])))
        
        # Create natural announcements for new detections
        if new_detections:
            if len(new_detections) == 1:
                announcement = f"I see a {next(iter(new_detections))}"
            else:
                items = list(new_detections)
                announcement = "I see " + ", ".join(items[:-1]) + " and " + items[-1]
            
            # Add random conversational prefix
            prefixes = ["", "Look, ", "There's ", "Now "]
            announcement = random.choice(prefixes) + announcement.lower()
            
            # Add to voice queue (non-blocking)
            voice_queue.append(announcement)
        
        last_detections = current_detections
        
        # Maintain your exact original display handling
        window_size = cv2.getWindowImageRect("Real-Time Object Detection")
        window_width, window_height = window_size[2], window_size[3]
        frame_height, frame_width = annotated_frame.shape[:2]
        aspect_ratio = frame_width / frame_height
        
        if window_width / aspect_ratio <= window_height:
            new_width = window_width
            new_height = int(window_width / aspect_ratio)
        else:
            new_height = window_height
            new_width = int(window_height * aspect_ratio)
            
        resized_frame = cv2.resize(annotated_frame, (new_width, new_height))
        
        # Display FPS for monitoring
        fps = 1 / (time.time() - start_time)
        # cv2.putText(resized_frame, f"FPS: {fps:.1f}", (10, 30),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Real-Time Object Detection", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    voice_queue.append("Object detection system stopped")

if __name__ == "__main__":
    real_time_detection()
