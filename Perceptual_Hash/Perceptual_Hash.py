import cv2
from PIL import Image
import imagehash
import numpy as np

# Función para convertir de imagen OpenCV (BGR) a PIL (RGB o L)
def cv2_to_pil(cv2_img):
    cv2_img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img_rgb)
    return pil_img.convert("L")  # Convertir a escala de grises


class PerceptualHash:
    def __init__(self, path_img_1, path_img_2):
        self.path_img_1 = path_img_1
        self.path_img_2 = path_img_2
        self.per_hash_1 = None
        self.per_hash_2 = None
        self.hamming_distance_value = None

    def calculate_perceptual_hashes(self):
        # load the images
        print("Calculating perceptual hashes... ... ... ")
        img1_cv = cv2.imread(self.path_img_1)
        img2_cv = cv2.imread(self.path_img_2)

        # convert to PIL for hashing
        img1_pil = cv2_to_pil(img1_cv)
        img2_pil = cv2_to_pil(img2_cv)

        # Calculating perceptual hashes
        self.per_hash_1  = imagehash.phash(img1_pil)
        self.per_hash_2  = imagehash.phash(img2_pil)

        return self.per_hash_1 , self.per_hash_2

    def hamming_distance(self):
        hash1, hash2 = self.calculate_perceptual_hashes()
        print("Hash perceptual 1:", hash1)
        print("Hash perceptual 2:", hash2)
        self.hamming_distance_value = abs(hash1 - hash2)
        print("Hamming Distance:", abs(hash1 - hash2))
        return self.hamming_distance_value

    def similarity(self):
        diff = abs(self.per_hash_1  - self.per_hash_2)
        similarity = 1 - (diff / 64)
        print(f"Similarity : {similarity:.2f}")

    def valid_image(self):
        print("The Hamming Distance is:", self.hamming_distance_value)
        d = self.hamming_distance_value

        if d == 0:
            print("Hamming Distance: Valid. Images are identical")
            return True
        elif 0 < d <= 5:
            print("Hamming Distance: Valid. Images are visually very similar")
            return True
        elif 6 <= d <= 15:
            print("Hamming Distance: Caution. Images are similar but not identical")
            return False  # En enfoque conservador, ya no es válida
        elif 16 <= d <= 30:
            print("Hamming Distance: Not valid. Noticeable differences in content")
            return False
        elif 31 <= d <= 45:
            print("Hamming Distance: Not valid. Content is largely different")
            return False
        else:
            print("Hamming Distance: Not valid. Images are completely different")
            return False

