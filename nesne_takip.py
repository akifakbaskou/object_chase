import cv2
import numpy as np

# Motor sürücü kontrol fonksiyonları (örnek)
def move_left():
    print("SOLA DÖNÜYOR")

def move_right():
    print("SAĞA DÖNÜYOR")

def move_forward():
    print("İLERİ GİDİYOR")

def stop():
    print("DURDU")

# Kamera başlat
cap = cv2.VideoCapture(0)  # Gerekirse 1, 2 yapın

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Aynalı görüntü
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mavi renk aralığı (HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Maskeyi oluştur
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Gürültü azaltmak için erode/dilate
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Konturları bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = frame.shape[:2]
    center_x = width // 2

    if contours:
        # En büyük konturu al
        c = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        object_center = x + w // 2

        # en büyük konturu çerçevele ve merkezine nokta koy
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (object_center, y + h // 2), 5, (0, 0, 255), -1)
        cv2.putText(frame, "Object Center", (object_center - 20, y + h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, "Center Line", (center_x - 20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Kontrole göre karar ver
        if abs(object_center - center_x) < 30:
            move_forward()
        elif object_center < center_x:
            move_left()
        else:
            move_right()
    else:
        stop()

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
