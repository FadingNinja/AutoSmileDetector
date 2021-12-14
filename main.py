import cv2
import datetime

cap = cv2.VideoCapture(0)
FaceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml.xml')
SmileCascade = cv2.CascadeClassifier('haarcascade_smile.xml.xml')

while True :
    success, frame = cap.read()
    original_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = FaceCascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in face :
        cv2.rectangle(frame, (x, y,), (x + w, y + h), (80, 255, 67), 2)
        face_roi = frame[y : y + h, x : x + w]
        gray_roi = frame[y : y + h, x : x + w]

        smile = SmileCascade.detectMultiScale(gray_roi, 1.3, 25)

        for x1, y1, w1, h1 in smile :
            cv2.rectangle(frame, (x1, y1,), (x1 + w1, y1 + h1), (255, 0, 0), 2)
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            filename = f"selfie-{timestamp}.png"
            cv2.imwrite(filename, original_frame)
    
    cv2.imshow('Capture', frame)

    if cv2.waitKey(10) == ord('q') :
        break