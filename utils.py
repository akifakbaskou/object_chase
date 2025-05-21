import numpy as np
import cv2

def linear_mapping(img):
    """
    Görüntüyü [0, 1] aralığına normalize eder.
    """
    min_val = np.min(img)
    max_val = np.max(img)

    # Min ve max aynıysa sıfır döndür (sabit bir görüntü)
    if max_val - min_val == 0:
        return np.zeros_like(img)
    else:
        return (img - min_val) / (max_val - min_val)


def pre_process(img):
    """
    Görüntüyü MOSSE için ön işler:
    - Log dönüştürme
    - Normalize etme
    - Pencereleme (Hanning)
    """

    # Küçük sayı hatalarını önlemek için 1 eklendi
    img = np.log(img.astype(np.float32) + 1)

    # Ortalamayı sıfıra getir
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)

    # Hanning (cosine) penceresi uygula
    win = np.hanning(img.shape[0])[:, None] * np.hanning(img.shape[1])[None, :]
    img = img * win

    return img
