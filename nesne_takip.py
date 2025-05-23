import cv2
import numpy as np
import subprocess

def move_left():
    print("SOLA DÖNÜYOR")

def move_right():
    print("SAĞA DÖNÜYOR")

def move_forward():
    print("İLERİ GİDİYOR")

def stop():
    print("DURDU")

# GStreamer borusu tanımı (H264 encode + UDP çıkış)
gst_str = (
    'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! '
    'rtph264pay config-interval=1 pt=96 ! '
    'udpsink host=PC_IP_ADRESINIZ port=5000'
)

# VideoCapture başlat
cap = cv2.VideoCapture(0)

# GStreamer çıkışı için VideoWriter
out = cv2.VideoWriter(
    gst_str,
    cv2.CAP_GSTREAMER,
    0,  # fourcc değil çünkü GStreamer string kullanıyoruz
    30.0,
    (640, 480)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = frame.shape[:2]
    center_x = width // 2

    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        object_center = x + w // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (object_center, y + h // 2), 5, (0, 0, 255), -1)

        if abs(object_center - center_x) < 30:
            move_forward()
        elif object_center < center_x:
            move_left()
        else:
            move_right()
    else:
        stop()

    # Görüntüyü ağdan gönder
    out.write(frame)

cap.release()
out.release()