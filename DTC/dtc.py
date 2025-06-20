import cv2
import numpy as np
from scipy.fftpack import dct, idct

# --- Helper Functions ---
# Apply 2D Discrete Cosine Transform (DCT)
def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Apply 2D Inverse DCT
def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


class StegoDTC:
    def __init__(self, img_path="image.png", message="hola"):
        self.image_path = img_path
        self.message = "Alejandro Salinas Victorino, Esta es mi Marca de Agua"
        # Coordinates of DCT coefficients to be modified
        self.u1, self.v1 = 2, 3
        self.u2, self.v2 = 3, 3
        self.x = 10.0  # Minimum difference between coefficients for robustness
        self.block_size = 8  # Image is processed in 8x8 blocks
        self.img = None
        self.h, self.w = 0, 0
        self.max_mark_size = None
        self.stego_img = None

    def open_show_image_grayscale(self, save=False):
        """
        Loads the image and converts it to grayscale.
        Optionally, it saves the grayscale version.
        """
        image_opened = cv2.imread(self.image_path)
        image_opened_gray = cv2.cvtColor(image_opened, cv2.COLOR_BGR2GRAY)
        cv2.imshow(self.image_path, image_opened_gray)

        if save:
            cv2.imshow("grayscale_" + self.image_path, image_opened_gray)
            cv2.imwrite("grayscale_" + self.image_path, image_opened_gray)

        return image_opened_gray

    def load_image(self):
        """
        Loads the grayscale image, stores dimensions, and crops it to fit 8x8 blocks.
        """
        self.img = self.open_show_image_grayscale()
        self.h, self.w = self.img.shape
        print("The shape is:", self.h, self.w)
        self.max_mark_size = self.h * self.w // 64 // 8
        print("The maximum size of bytes is:", self.max_mark_size)
        print("The message size is:", len(self.message))
        self.img = self.img[:self.h - self.h % self.block_size, :self.w - self.w % self.block_size]

    def embedded_bit(self, block, bit):
        """
        Embeds a single bit into an 8x8 block using the DCT coefficient comparison.
        """
        B = dct2(block.astype(float))

        c1 = B[self.u1, self.v1]
        c2 = B[self.u2, self.v2]

        # Adjust coefficient order based on bit
        if bit == 0 and c1 > c2:
            c1, c2 = c2, c1
        elif bit == 1 and c1 < c2:
            c1, c2 = c2, c1

        # Ensure difference is greater than threshold x
        if abs(c1 - c2) < self.x:
            delta = (self.x - abs(c1 - c2)) / 2
            if c1 > c2:
                c1 += delta
                c2 -= delta
            else:
                c1 -= delta
                c2 += delta

        # Store adjusted coefficients
        B[self.u1, self.v1] = c1
        B[self.u2, self.v2] = c2

        # Convert block back to spatial domain
        return np.round(idct2(B)).clip(0, 255).astype(np.uint8)

    def create_marking_text2bin_vector(self):
        """
        Converts the input message to a binary vector (ASCII per character).
        """
        bin_vector = []
        for character in self.message:
            for i in range(7, -1, -1):
                bit_character = (ord(character) >> i) & 1
                bin_vector.append(bit_character)
        print("The bit vector is: ", bin_vector)
        return bin_vector

    def embedded_text_message(self):
        """
        Embeds the entire binary message into the image block by block.
        """
        text2bin_vector = self.create_marking_text2bin_vector()
        embedded_blocks = []
        index = 0

        for i in range(0, self.h, self.block_size):
            for j in range(0, self.w, self.block_size):
                block = self.img[i:i + self.block_size, j:j + self.block_size]
                if index < len(text2bin_vector):
                    embedded = self.embedded_bit(block, text2bin_vector[index])
                    embedded_blocks.append(((i, j), embedded))
                    index += 1
                else:
                    embedded_blocks.append(((i, j), block))

        # Reconstruct the stego image from the modified blocks
        self.stego_img = np.zeros_like(self.img)
        for (i, j), block in embedded_blocks:
            self.stego_img[i:i + self.block_size, j:j + self.block_size] = block

        cv2.imshow("stego_" + self.image_path, self.stego_img)
        cv2.imwrite("stego_dtc_" + self.image_path, self.stego_img)

    def extract_bit(self, block):
        """
        Extracts a bit from a block based on the DCT coefficient comparison.
        """
        B = dct2(block.astype(float))
        return 1 if B[self.u1, self.v1] > B[self.u2, self.v2] else 0

    def recover_text_watermarking(self):
        """
        Recovers the embedded text message from the stego image.
        """
        extracted_bits = []
        message_len = len(self.message) * 8  # total bits expected
        index = 0
        for i in range(0, self.h, self.block_size):
            for j in range(0, self.w, self.block_size):
                if index > message_len - 1:
                    break
                block = self.stego_img[i:i + self.block_size, j:j + self.block_size]
                bit = self.extract_bit(block)
                extracted_bits.append(bit)
                index += 1

        print("The extracted message is:", extracted_bits)

        # Group bits into 8-bit chunks and convert to characters
        chunks = [extracted_bits[i:i + 8] for i in range(0, len(extracted_bits), 8)]
        marking_recovered = ''.join([chr(int(''.join(map(str, byte)), 2)) for byte in chunks])
        print(f"Recovered watermarking:")
        print(marking_recovered)

    def finish_opencv_session(self):
        """
        Finalizes OpenCV windows.
        """
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Finish all the OpenCV sessions")


# --- Execution ---
watermarking_dtc = StegoDTC("image.png")     # Initialize with image file
watermarking_dtc.load_image()                # Load and preprocess image
watermarking_dtc.embedded_text_message()     # Embed watermark into image
watermarking_dtc.recover_text_watermarking() # Recover watermark
watermarking_dtc.finish_opencv_session()     # Close windows
