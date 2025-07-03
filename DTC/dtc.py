"""
This code add the watermarking in a grayscale and RGB image
using the DCT concept.

By: Alejandro Salinas V.
"""

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

def calculate_psnr(original, modified):
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')  # they are the same
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

#PSNR measurement method
def psnr_measurement(original_path, modified_path, grayscale=True):
    if grayscale:
        # Load grayscale images
        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        modified = cv2.imread(modified_path, cv2.IMREAD_GRAYSCALE)

        if original is None or modified is None:
            print("Error: Could not load one or both images.")
            return

        if original.shape != modified.shape:
            print("Error: Images must be the same size and type.")
            return

        psnr_value = calculate_psnr(original, modified)
        print(f"[Grayscale] PSNR: {psnr_value:.2f} dB")

    else:
        # Load color images
        original = cv2.imread(original_path)
        modified = cv2.imread(modified_path)

        if original is None or modified is None:
            print("Error: Could not load one or both images.")
            return

        if original.shape != modified.shape:
            print("Error: Images must be the same size and type.")
            return

        # Split channels
        channels_original = cv2.split(original)
        channels_modified = cv2.split(modified)
        channel_names = ['Blue', 'Green', 'Red']

        # Calculate and display PSNR for each channel
        for i in range(3):
            psnr_val = calculate_psnr(channels_original[i], channels_modified[i])
            print(f"[Color - {channel_names[i]}] PSNR: {psnr_val:.2f} dB")

# --- --- --- --- --- --- #

class StegoDCT:
    def __init__(self, img_path="image.png", message="WaterMarking", message2=None, message3=None):
        self.image_path = img_path
        self.message = message
        self.message2 = message2 # for RGB
        self.message3 = message3 # for RBG
        # Coordinates of DCT coefficients to be modified
        self.u1, self.v1 = 2, 1
        self.u2, self.v2 = 3, 3
        self.x = 10.0  # Minimum difference between coefficients for robustness
        self.block_size = 8  # Image is processed in 8x8 blocks
        self.img = None
        self.h, self.w = 0, 0
        self.max_mark_size = None
        self.stego_img = None


    def open_show_image_grayscale(self, image_path=None, save=False):
        """
        Loads the image and converts it to grayscale.
        Optionally, it saves the grayscale version.
        """
        if image_path is None:
            image_path = self.image_path

        image_opened = cv2.imread(image_path)
        image_opened_gray = cv2.cvtColor(image_opened, cv2.COLOR_BGR2GRAY)
        cv2.imshow(image_path, image_opened_gray)

        if save:
            cv2.imshow("grayscale_" + image_path, image_opened_gray)
            cv2.imwrite("grayscale_" + image_path, image_opened_gray)

        return image_opened_gray

    def load_gray_image(self):
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


    def load_image_rgb(self, image_path=None):
        """
        Loads the image in color (BGR), stores its dimensions, and crops it to fit 8x8 blocks.
        """
        if image_path is None:
            image_path = self.image_path

        self.img = cv2.imread(image_path)
        cv2.imshow("Original_" + image_path, self.img)
        if self.img is None:
            raise ValueError("Image could not be loaded.")

        self.h, self.w, _ = self.img.shape
        print("Original shape:", self.img.shape)
        self.max_mark_size = self.h * self.w // 64 // 8
        print("Maximum embeddable bytes:", self.max_mark_size)
        print("Message size:", len(self.message))

        # Crop image to be divisible by block_size
        self.img = self.img[:self.h - self.h % self.block_size, :self.w - self.w % self.block_size]
        self.h, self.w, _ = self.img.shape

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
        block_reconstructed = idct2(B)
        block_reconstructed = np.clip(np.round(block_reconstructed), 0, 255)
        return block_reconstructed.astype(np.uint8)
        #return np.round(idct2(B)).clip(0, 255).astype(np.uint8)

    def create_marking_text2bin_vector(self, message=None):
        """
        Converts the input message to a binary vector (ASCII per character).
        """
        if message is None:
            message = self.message
        
        bin_vector = []
        for character in message:
            for i in range(7, -1, -1):
                bit_character = (ord(character) >> i) & 1
                bin_vector.append(bit_character)
        #print("The bit vector is: ", bin_vector)
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

        cv2.imshow("stego_dtc_" + self.image_path, self.stego_img)
        cv2.imwrite("stego_dtc_" + self.image_path, self.stego_img)

    def extract_bit(self, block):
        """
        Extracts a bit from a block based on the DCT coefficient comparison.
        """
        B = dct2(block.astype(float))
        return 1 if B[self.u1, self.v1] > B[self.u2, self.v2] else 0


    def recover_text_watermarking(self, message_len, path_name:str):
        """
        Recovers the embedded text message from the stego image.
        """
        stego_img = self.open_show_image_grayscale(path_name)
        h, w = stego_img.shape
        extracted_bits = []
        message_len = message_len * 8 # total bits expected
        index = 0
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                if index > message_len - 1:
                    break
                block = stego_img[i:i + self.block_size, j:j + self.block_size]
                bit = self.extract_bit(block)
                extracted_bits.append(bit)
                index += 1

        #print("The extracted message is:", extracted_bits)

        # Group bits into 8-bit chunks and convert to characters
        chunks = [extracted_bits[i:i + 8] for i in range(0, len(extracted_bits), 8)]
        marking_recovered = ''.join([chr(int(''.join(map(str, byte)), 2)) for byte in chunks])
        print(f"Recovered watermarking in grayscale:", marking_recovered)
        return marking_recovered

    def embedded_text_message_rgb(self):
        """
        Embed the binary message into the blue channel of the RGB image.
        """
        # Split image into BGR channels
        b_channel, g_channel, r_channel = cv2.split(self.img)
        # Pair each channel with its respective message
        channels = [b_channel, g_channel, r_channel]
        messages = [self.message, self.message2, self.message3]

        stego_channels = []
        for channel, message in zip(channels, messages):
            text2bin_vector = self.create_marking_text2bin_vector(message)
            embedded_blocks = []
            index = 0
            for i in range(0, self.h, self.block_size):
                for j in range(0, self.w, self.block_size):
                    block = channel[i:i + self.block_size, j:j + self.block_size]
                    if index < len(text2bin_vector):
                        embedded = self.embedded_bit(block, text2bin_vector[index])
                        embedded_blocks.append(((i, j), embedded))
                        index += 1
                    else:
                        embedded_blocks.append(((i, j), block))

            # # Reconstruct the modified channel
            stego_channel = np.zeros_like(channel)
            for (i, j), block in embedded_blocks:
                stego_channel[i:i + self.block_size, j:j + self.block_size] = block
            stego_channels.append(stego_channel)
        # Merge channels and save
        self.stego_img = cv2.merge(stego_channels)
        cv2.imshow("stego_rgb_dtc_" + self.image_path, self.stego_img)
        cv2.imwrite("stego_rgb_dtc_" + self.image_path, self.stego_img)

    def recover_text_watermarking_rgb(self, message_1_len, message_2_len, message_3_len, path_name: str):
        """
        Recover the watermarks from the RGB channels of the stego image.
        """
        stego_img = cv2.imread(path_name)
        if stego_img is None:
            raise ValueError("Stego image could not be loaded.")

        h, w, _ = stego_img.shape
        b_channel, g_channel, r_channel = cv2.split(stego_img)  # <-- FIX: use stego_img not self.img

        channels = [b_channel, g_channel, r_channel]
        messages_len = [message_1_len, message_2_len, message_3_len]
        channel_names = ["Blue", "Green", "Red"]

        extracted_messages = []

        for channel, message_len, name in zip(channels, messages_len, channel_names):
            extracted_bits = []
            index = 0
            for i in range(0, h, self.block_size):
                if index >= (message_len * 8):
                    break
                for j in range(0, w, self.block_size):
                    if index >= (message_len * 8):
                        break
                    block = channel[i:i + self.block_size, j:j + self.block_size]
                    bit = self.extract_bit(block)
                    extracted_bits.append(bit)
                    index += 1

            # Convert bits to characters

            chunks = [extracted_bits[i:i + 8] for i in range(0, len(extracted_bits), 8)]
            recovered_message = ''.join([chr(int(''.join(map(str, byte)), 2)) for byte in chunks])
            extracted_messages.append(recovered_message)
            # Truncar a los bits esperados


            print(f"Recovered watermark from {name} channel: {recovered_message}")
        return extracted_messages

    def embed_rgb_image_watermark(self, watermark_path, output_image_path=None):
        """
        Incrusta una imagen RGB como marca de agua en la imagen base RGB (self.img),
        utilizando bloques 8x8 y comparaciones de coeficientes DCT.
        """
        # Cargar imagen de marca de agua y redimensionarla
        watermark = cv2.imread(watermark_path)
        if watermark is None:
            raise ValueError("No se pudo cargar la imagen de marca de agua.")

        wm_h_blocks = int(self.max_mark_size ** 0.5) #self.h // self.block_size
        #print(f"the wm_h_blocks {wm_h_blocks}")
        wm_w_blocks = int(self.max_mark_size ** 0.5)  #self.w // self.block_size
        #print(f"the wm_w_blocks {wm_w_blocks}")
        watermark = cv2.resize(watermark, (wm_w_blocks, wm_h_blocks))  # Redimensionar a bloques disponibles
        cv2.imshow("scaling_" + watermark_path, watermark)
        cv2.imwrite("scaling_" + watermark_path, watermark) # scaling
        wm_b, wm_g, wm_r = cv2.split(watermark)
        wm_channels = [wm_b, wm_g, wm_r]

        cover_channels = cv2.split(self.img)
        stego_channels = []

        for channel_cover, channel_wm in zip(cover_channels, wm_channels):
            embedded_blocks = []
            index = 0
            wm_flat_bits = []

            # Convertir la imagen de marca en una lista de bits
            for pixel in channel_wm.flatten():
                for i in range(7, -1, -1):
                    wm_flat_bits.append((pixel >> i) & 1)
            #print("wm_flat_bits", wm_flat_bits)
            #print("wm_flat_bits len:", len(wm_flat_bits))
            for i in range(0, self.h, self.block_size):
                for j in range(0, self.w, self.block_size):
                    if index < len(wm_flat_bits):
                        block = channel_cover[i:i + self.block_size, j:j + self.block_size]
                        bit = wm_flat_bits[index]
                        embedded = self.embedded_bit(block, bit)
                        embedded_blocks.append(((i, j), embedded))
                        index += 1
                    else:
                        block = channel_cover[i:i + self.block_size, j:j + self.block_size]
                        embedded_blocks.append(((i, j), block))

            stego_channel = np.zeros_like(channel_cover)
            for (i, j), block in embedded_blocks:
                stego_channel[i:i + self.block_size, j:j + self.block_size] = block

            stego_channels.append(stego_channel)

        self.stego_img = cv2.merge(stego_channels)
        if output_image_path is None:
            cv2.imshow("stego_rgb_image_dtc_" + self.image_path, self.stego_img)
            cv2.imwrite("stego_rgb_image_dtc_" + self.image_path, self.stego_img)

        else:
            cv2.imshow("DCT_" + self.image_path, self.stego_img)
            cv2.imwrite(output_image_path, self.stego_img)

        return wm_h_blocks, wm_w_blocks

    def extract_rgb_image_watermark(self, wm_shape, stego_path, output_path=None):
        """
        Extrae una imagen RGB incrustada como marca de agua desde una imagen esteganográfica.
        `wm_shape` debe ser una tupla con las dimensiones de la imagen de marca (alto, ancho).
        """
        stego_img = cv2.imread(stego_path)
        if stego_img is None:
            raise ValueError("No se pudo cargar la imagen esteganográfica.")

        h, w, _ = stego_img.shape
        channels = cv2.split(stego_img)
        recovered_channels = []

        wm_h, wm_w = wm_shape
        total_bits = wm_h * wm_w * 8  # Por canal

        for channel in channels:
            extracted_bits = []
            index = 0
            for i in range(0, h, self.block_size):
                for j in range(0, w, self.block_size):
                    if index >= total_bits:
                        break
                    block = channel[i:i + self.block_size, j:j + self.block_size]
                    bit = self.extract_bit(block)
                    extracted_bits.append(bit)
                    index += 1
                if index >= total_bits:
                    break

            # Agrupar bits en bytes
            pixels = []
            for i in range(0, len(extracted_bits), 8):
                byte = extracted_bits[i:i + 8]
                if len(byte) < 8:
                    break
                value = int(''.join(str(b) for b in byte), 2)
                pixels.append(value)

            recovered_channel = np.array(pixels, dtype=np.uint8).reshape((wm_h, wm_w))
            recovered_channels.append(recovered_channel)

        watermark_recovered = cv2.merge(recovered_channels)
        cv2.imshow("Recovered RGB Watermark", watermark_recovered)
        if output_path is None:
            output_path = "recovered_rgb_watermark.png"
        cv2.imwrite(output_path, watermark_recovered)
        return watermark_recovered

    @staticmethod
    def finish_opencv_session():
        """
        Finalizes OpenCV windows.
        """
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Finish all the OpenCV sessions")



def main_gray_scale():
    watermarking = "Alejandro Salinas Victorino, Esta es mi Marca de Agua. Saludos!"
    image_name = "image.png"
    watermarking_dtc = StegoDCT(img_path=image_name, message=watermarking)  # Initialize with image file
    watermarking_dtc.load_gray_image()  # Load and preprocess image
    watermarking_dtc.embedded_text_message()  # Embed watermark into image
    watermarking_dtc.recover_text_watermarking(message_len=len(watermarking),
                                               path_name="stego_dtc_" + image_name)  # Recover watermark
    psnr_measurement(original_path=image_name, modified_path= "stego_dtc_" + image_name)
    watermarking_dtc.finish_opencv_session()

def main_rgb_scale():
    watermarking_rgb_1 = "Alejandro Salinas Victorino, Esta es mi Marca de Agua, saludos!"
    watermarking_rgb_2 = "Esta es mi Segunda Marca de agua que he incrustado"
    watermarking_rgb_3 = "Incrustando una Tercera Marca de agua para que sea recuperada"
    image_name_rgb = "image.png"
    watermarking_rgb_dtc = StegoDCT(img_path=image_name_rgb, message=watermarking_rgb_1, message2=watermarking_rgb_2, message3=watermarking_rgb_3)
    watermarking_rgb_dtc.load_image_rgb()
    watermarking_rgb_dtc.embedded_text_message_rgb()
    watermarking_rgb_dtc.recover_text_watermarking_rgb(message_1_len=len(watermarking_rgb_1),
                                                       message_2_len=len(watermarking_rgb_2),
                                                       message_3_len=len(watermarking_rgb_3),
                                                       path_name="stego_rgb_dtc_" + image_name_rgb)
    psnr_measurement(original_path=image_name_rgb, modified_path="stego_rgb_dtc_" + image_name_rgb, grayscale=False)
    watermarking_rgb_dtc.finish_opencv_session()


def main_watermarking_rgb():
    image_name_rgb = "image.png"
    watermarking = "alex_watermarking.png"
    watermarking_rgb_dtc = StegoDCT(img_path=image_name_rgb)
    watermarking_rgb_dtc.load_image_rgb()
    wm_h, wm_w = watermarking_rgb_dtc.embed_rgb_image_watermark(watermarking)
    # "stego_rgb_image_dtc_" + self.image_path,
    watermarking_rgb_dtc.extract_rgb_image_watermark(wm_shape=(wm_h,wm_w),stego_path="stego_rgb_image_dtc_" +image_name_rgb )

    watermarking_rgb_dtc.finish_opencv_session()


if __name__ == "__main__":
    #main_gray_scale()
    #main_rgb_scale()
    main_watermarking_rgb()