import cvzone  # cvzone kütüphanesini içe aktarıyoruz
from cvzone.SelfiSegmentationModule import SelfiSegmentation  # SelfiSegmentation modülünü içe aktarıyoruz
import cv2 as cv  # OpenCV kütüphanesini içe aktarıyoruz
import os  # os kütüphanesini içe aktarıyoruz

# Kameranın başlatılması ve boyut ayarlamaları için
cap = cv.VideoCapture(0)  # Web kamerasını başlatıyoruz
cap.set(3, 640)  # Kameranın genişlik ayarını 640 piksel olarak belirliyoruz
cap.set(4, 960)  # Kameranın yükseklik ayarını 960 piksel olarak belirliyoruz

# Segmentasyon modülünü başlatıyoruz
segmentor = SelfiSegmentation()  # SelfiSegmentation modülünü başlatıyoruz

# Arka plan videosunu yüklüyoruz
video_path = "video/Dalgalar.mp4"  # Arka plan videosunun dosya yolunu belirtiyoruz
video = cv.VideoCapture(video_path)  # Belirtilen video dosyasını açıyoruz

while True:
    success, frame = cap.read()  # Web kamerasından görüntü alıyoruz
    if not success:
        print("Failed to capture frame.")  # Görüntü alınamazsa hatayı konsola yazdırıyoruz
        break

    # Videodan arka plan karesi okumak için
    ret, background_frame = video.read()  # Arka plan karesini videodan okuyoruz
    if not ret:
        video.set(cv.CAP_PROP_POS_FRAMES, 0)  # Eğer video biterse başa sarıyoruz
        _, background_frame = video.read()  # Videoyu tekrar baştan okuyarak arka plan karesini alıyoruz

    # Arka planı boyutlandırmak için
    background_frame = cv.resize(background_frame, (frame.shape[1], frame.shape[0]))  # Arka plan karesini kameradan alınan görüntünün boyutlarına göre yeniden boyutlandırıyoruz

    # Segmentasyon işlemi (Arka planı değiştirir)
    imgOut = segmentor.removeBG(frame, background_frame, cutThreshold=0.8)  # Segmentasyon yaparak arka planı değiştiriyoruz, cutThreshold değeri segmentasyon hassasiyetini belirler
    
    # İki görüntüyü yan yana ekranda göstermek için
    imgStacked = cv.hconcat([frame, imgOut])  # Orijinal ve değiştirilmiş görüntüyü yatay olarak birleştiriyoruz

    # Yan yana olan görüntüyü göstermek için
    cv.imshow("Selfie Segmentation", imgStacked)  # Birleştirilmiş görüntüyü ekranda gösteriyoruz
    
    if cv.waitKey(1) & 0xFF == ord("q"):  # 'q' tuşuna basıldığında döngüden çıkmak için
        break

# Pencereleri kapatmak için
cv.destroyAllWindows()  # Tüm OpenCV pencerelerini kapatıyoruz
cap.release()  # Web kamerasını serbest bırakıyoruz
video.release()  # Video dosyasını serbest bırakıyoruz
