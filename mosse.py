import numpy as np
import cv2
import os
from utils import linear_mapping, pre_process


class mosse:
    def __init__(self,frame,roi,sigma,lr):

        #Alınan ilk kare init-img değişkenine atanır.
        self.init_img = frame

        #Sigma değeri belirlenir.
        self.sigma=sigma

        #Öğrenme katsayısı(learning rate) belirlenir.
        self.lr=lr

        #Görüntü griye döndürülür.
        init_frame = cv2.cvtColor(self.init_img, cv2.COLOR_BGR2GRAY)

        #Görüntü float32 tipine dönüştürülür.
        self.init_frame = init_frame.astype(np.float32)
        
        #Seçilen hedefi içine alan pencere init_gt değişkenine atanır.
        init_gt = roi

        #Pencere int64 tipine dönüştürülür.
        self.init_gt = np.array(init_gt).astype(np.int64)

        #Yapay hedef için gerekli gauss dağılımı fonksiyonu çalıştırılır.
        self.response_map = self._get_gauss_response(self.init_frame, self.init_gt)
        
        #Gauss dağılımından hedefimizin bulunduğu pencere çıkartılır ve g değişkenine atanır.
        g = self.response_map[self.init_gt[1]:self.init_gt[1]+self.init_gt[3], self.init_gt[0]:self.init_gt[0]+self.init_gt[2]]
        
        #g değişkeni fourier alanına dönüştürülür. (İşlemlerin daha hızlı çalışması için her değişken fourier alanına dönüştürülür)
        self.G = np.fft.fft2(g)

        #Hedefimizin bulunduğu pencere tüm resimden kesilir ve fi değişkenine atanır.
        self.fi = self.init_frame[self.init_gt[1]:self.init_gt[1]+self.init_gt[3], self.init_gt[0]:self.init_gt[0]+self.init_gt[2]]
        
        #fi değişkeni ön işlemeden geçirilir.
        self.fi = pre_process(self.fi)

        #fi değişkeni fourier alanına dönüştürülür.
        self.fi = np.fft.fft2(self.fi)

        #Ai ve Bi değerleri bulunur
        self.Ai, self.Bi = self._pre_training(self.fi, self.G)

        #Seçilen alanın x y ve w h değerleri pos dizisine alınır.
        self.pos = self.init_gt.copy()

        #Yine aynı alanın (x,y) ve ((x+w),(y+h)) değerleri clip_pos dizisine alınır.
        self.clip_pos = np.array([self.pos[0], self.pos[1], self.pos[0]+self.pos[2], self.pos[1]+self.pos[3]]).astype(np.int64)
      


    def update(self,frame):

        #Güncel kare alınır ve current_frame değişkenine atanır.
        current_frame = frame

        #Kare griye dönüştürülür.
        frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = frame_gray.astype(np.float32)

        #Formülden Hi değeri (tam filtre) bulunur.
        Hi = self.Ai / self.Bi
        height, width = self.fi.shape
        
        #Önceki takip pencerisi yeni kare üzerinden alınır ve fi değişkenine atanır.
        fi = frame_gray[self.clip_pos[1]:self.clip_pos[3], self.clip_pos[0]:self.clip_pos[2]]

        #Yeni fi değeri ön işlemeden geçirilir ve fourier alanına dönüştürülür.
        self.fi = np.fft.fft2(pre_process(cv2.resize(fi, (self.init_gt[2], self.init_gt[3]))))

        #Formülden Gi değeri bulunur.
        Gi = Hi * self.fi

        #Gi değeri üzerinden ters fft ve normalizasyon yapılarak gauss dağılımı elde edilir.
        gi = linear_mapping(np.fft.ifft2(Gi))
        
        #Dağılımın maksimum değeri alınır
        max_value = np.max(gi)
        
        #Yeni dağılımın tepe noktası bulunur ve merkez noktası ile arasındaki fark dx ve dy değişkenlerine atanır.
        max_pos = np.where(gi == max_value)
        dy = int(np.mean(max_pos[0]) - gi.shape[0] / 2)
        dx = int(np.mean(max_pos[1]) - gi.shape[1] / 2)
        
        # Koordinatlar güncellenir.
        self.pos[0] = self.pos[0] + dx
        self.pos[1] = self.pos[1] + dy

        # Yeni koordinatlar kare dışına taşmayacak şekilde clip_pos değişkenine atanır. [xmin, ymin, xmax, ymax]
        self.clip_pos[0] = np.clip(self.pos[0], 0, current_frame.shape[1])
        self.clip_pos[1] = np.clip(self.pos[1], 0, current_frame.shape[0])
        self.clip_pos[2] = np.clip(self.pos[0]+self.pos[2], 0, current_frame.shape[1])
        self.clip_pos[3] = np.clip(self.pos[1]+self.pos[3], 0, current_frame.shape[0])
        self.clip_pos = self.clip_pos.astype(np.int64)

        # Yeni fi değeri bulunur.
        fi = frame_gray[self.clip_pos[1]:self.clip_pos[3], self.clip_pos[0]:self.clip_pos[2]]

        #Yeni fi değeri ön işlemeden geçirilir ve fourier alanına dönüştürülür.
        if fi.size==0:
            return []
        self.fi = np.fft.fft2(pre_process(cv2.resize(fi, (self.init_gt[2], self.init_gt[3]))))

        # Formül kullanılara Ai ve Bi değerleri güncellenir.
        self.Ai = self.lr * (self.G * np.conjugate(self.fi)) + (1 - self.lr) * self.Ai
        self.Bi = self.lr * (self.fi * np.conjugate(self.fi)) + (1 - self.lr) * self.Bi
        
        # Takip penceresi kare üzerine çizilir.
        cv2.rectangle(current_frame, (self.pos[0], self.pos[1]), (self.pos[0]+self.pos[2], self.pos[1]+self.pos[3]), (255, 0, 0), 2)
        return current_frame
                
   
    def _pre_training(self, fi, G):
        
        #fi değişkeninin eşleniği alınır.
        fiC=np.conjugate(fi)

        #Formüldeki Ai bulunur.
        Ai = G * fiC
        
        #Formüldeki Bi bulunur.
        Bi = fi * fiC
        
        return Ai, Bi

    def _get_gauss_response(self, img, gt):

        #Görüntüden yükseklik ve genişlik değerleri alınır.
        height, width = img.shape

        # 2d boyutlu bir ızgara oluşturmak için gerekli x ve y değerleri alınır.
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        # Hedefin merkez koordinatları alınır.
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]

        
        # Formüldeki üs değeri hesaplanır.
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.sigma)

        # e üzeri dist değeri bulunur.
        response = np.exp(-dist)
      
        # Min-Max normalizasyonu yapılır.
        response = linear_mapping(response)
 
        return response
    
if __name__ == "__main__":
    # Test kodu
    video_path = 0  # Webcam için 0 kullanabilirsiniz

    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)

    # İlk kareyi oku
    ret, frame = cap.read()

    # ROI'yi seç
    roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    # MOSSE nesnesini oluştur
    tracker = mosse(frame, roi, sigma=20, lr=0.125)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Takip et
        output_frame = tracker.update(frame)

        if len(output_frame) == 0:
            break

        # Çerçeveyi göster
        cv2.imshow("Tracking", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
