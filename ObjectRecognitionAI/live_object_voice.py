import cv2
from ultralytics import YOLO
import pyttsx3
import time

# Load YOLO model
model = YOLO("yolov8n.pt")  # pre-trained

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Open webcam
cap = cv2.VideoCapture(0)

last_spoken = ""
last_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    annotated_frame = results[0].plot()

    # Speak detected objects
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            current_time = time.time()
            if label != last_spoken and current_time - last_time > 3:
                engine.say(f"{label} detected")
                engine.runAndWait()
                last_spoken = label
                last_time = current_time

    cv2.imshow("Live Object Recognition with Voice", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
