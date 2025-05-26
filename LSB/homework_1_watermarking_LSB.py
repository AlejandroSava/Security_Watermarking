"""
Developed by Alejandro Salinas V.
Watermarking applying LSB
"""

import cv2
import numpy as np
import random
import string

def calculate_psnr(original, modified):
    mse = np.mean((original.astype(np.float64) - modified.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')  # they are the same
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

class WaterMarking:
    def __init__(self, marking:str, image_path_name:str='kemonito.png' ):
        self.marking = marking
        self.size_watermarking_characters = len(marking)
        self.image_path_name = image_path_name
        self.image_gray_data = None
        self.random_position = None
        self.psnr = None
        self.channel_blue = None
        self.channel_red = None
        self.channel_green = None


    def load_show_rgb_channels_opencv(self, image_path):
        # Load the image in color (BGR format)
        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            print("Error: Image could not be loaded.")
            return

        # Split BGR channels
        self.channel_blue, self.channel_green, self.channel_red = cv2.split(image_bgr)

        # Show each channel in grayscale
        cv2.imshow("Original (BGR)", image_bgr)
        cv2.imshow("Blue Channel", self.channel_blue)
        cv2.imshow("Green Channel", self.channel_green)
        cv2.imshow("Red Channel", self.channel_red)


    def open_show_image_grayscale(self, image_path:str = None, save = False):
        """
        Open an image according to the image_name, and only shows it in grayscale
        after that it save the image in this scale
        """
        if image_path is None:
            image_path = self.image_path_name

        image_opened = cv2.imread(image_path)
        image_opened_gray = cv2.cvtColor(image_opened, cv2.COLOR_BGR2GRAY)

        cv2.imshow(image_path, image_opened_gray)

        if save:
            cv2.imshow(str("grayscale_") + image_path, image_opened_gray)
            cv2.imwrite(str("grayscale_") + image_path, image_opened_gray)

        return image_opened_gray

    def adding_marking_grayscale(self, data_layer:np.ndarray):
        row, col = data_layer.shape
        print(f"The maximum characters size of the marking is: { (row * col) // 8}")
        bin_vector = self.create_marking_bin_vector()
        index_bin_vector = 0
        len_bin_vector = len(bin_vector)
        print(f"the len of bin vector is {len_bin_vector}")
        while index_bin_vector < len_bin_vector:
            for r in range(row):
                for c in range(col):
                    if index_bin_vector >= len_bin_vector:
                        break
                    data_layer[r][c] = (int(data_layer[r][c]) & ~1) | bin_vector[index_bin_vector] # (new_element & 1111 1110) | bit
                    index_bin_vector += 1
        print("The data was added")
        return data_layer



    def adding_marking_grayscale_random(self, data_layer: np.ndarray):
        row, col = data_layer.shape
        total_pixels = row * col
        bin_vector = self.create_marking_bin_vector()
        len_bin_vector = len(bin_vector)
        if len_bin_vector > total_pixels:
            raise ValueError("This messages is bigger thant the image")

        print(f"Inserting {len_bin_vector} bits into a space of {total_pixels} pixels (randomized)")
        # Generate the positions to add the LSB bits in random way
        if self.random_position is None:
            all_positions = [(r, c) for r in range(row) for c in range(col)]
            random.seed()
            selected_positions = random.sample(all_positions, len_bin_vector)
            self.random_position = selected_positions  # saving in memory to recover the message
        else:
            selected_positions = self.random_position

        for i, (r, c) in enumerate(selected_positions):
            data_layer[r][c] = (int(data_layer[r][c]) & ~1) | bin_vector[i]


        print("The data was added randomly")
        return data_layer


    def recover_marking_grayscale_random(self, path_image_watermarking_grayscale_random:str):
        """
        Recover the data from the image
        :param path_image_watermarking_grayscale:
        :return:
        """
        print("Recover the watermarking in grayscale random")
        watermarking_grayscale = self.open_show_image_grayscale(path_image_watermarking_grayscale_random)
        bin_vector = []
        index_bin_vector = 0
        len_bin_vector = self.size_watermarking_characters * 8
        print(f"the len of bin vector is {len_bin_vector}")

        while index_bin_vector < len_bin_vector:
            for (r, c) in self.random_position:
                if index_bin_vector >= len_bin_vector:
                    break
                bin_vector.append((int(watermarking_grayscale[r][c]) & 1)) # (element & 0000 0001)
                index_bin_vector += 1
                #print("The index of the bin vector is: ", index_bin_vector)
        #print ("the bin vector recovered is: ", bin_vector)

        # Split into chunks of 8 bits
        chunks = [bin_vector[i:i + 8] for i in range(0, len(bin_vector), 8)]
        # Convert each chunk to a character
        marking_recovered = ''.join([chr(int(''.join(map(str, byte)), 2)) for byte in chunks])
        print(marking_recovered)

    def recover_marking_grayscale(self, path_image_watermarking_grayscale:str):
        watermarking_grayscale = self.open_show_image_grayscale(path_image_watermarking_grayscale)
        bin_vector = []
        index_bin_vector = 0
        len_bin_vector = self.size_watermarking_characters * 8
        print(f"the len of bin vector is {len_bin_vector}")
        row, col = watermarking_grayscale.shape
        while index_bin_vector < len_bin_vector:
            for r in range(row):
                for c in range(col):
                    if index_bin_vector >= len_bin_vector:
                        break
                    bin_vector.append((int(watermarking_grayscale[r][c]) & 1)) # (element & 0000 0001)
                    index_bin_vector += 1
                    #print("The index of the bin vector is: ", index_bin_vector)
        #print ("the bin vector recovered is: ", bin_vector)

        # Split into chunks of 8 bits
        chunks = [bin_vector[i:i + 8] for i in range(0, len(bin_vector), 8)]
        # Convert each chunk to a character
        marking_recovered = ''.join([chr(int(''.join(map(str, byte)), 2)) for byte in chunks])
        print(marking_recovered)

    def create_marking_bin_vector(self):
        bin_vector = []
        for character in self.marking:
            #print("The current element is:", character)
            #print("The binary ASCII decomposition of the element is:")

            for i in range(7, -1, -1):  # Iterate from bit 7 to 0
                bit_character = (ord(character) >> i) & 1  # Extract each bit
                #print(i, ":", bit_character)
                bin_vector.append(bit_character)
        #print("The bit vector is: ", bin_vector)
        return bin_vector

    def plane_zero(self, path_image_watermarking_grayscale:str):
        watermarking_grayscale = self.open_show_image_grayscale(path_image_watermarking_grayscale)
        plane_0 = watermarking_grayscale & 1  # LSB
        #print("The plane 0 is: ", plane_0)
        plane_0_vis = (plane_0 * 255).astype(np.uint8)  # Scale plane 0 to be able to display it visually
        self.save_and_show_image("Plane_0_" + path_image_watermarking_grayscale, plane_0_vis)

    def psrn_measurement(self, original:str, modified:str):
        # load the images in grayscale
        original = cv2.imread(original, cv2.IMREAD_GRAYSCALE)
        modified = cv2.imread(modified, cv2.IMREAD_GRAYSCALE)

        # verify that they have the same size
        if original.shape != modified.shape:
            print("The images must have the same size")
        else:
            self.psnr = calculate_psnr(original, modified)
            print(f"PSNR: {self.psnr:.2f} dB")

    def save_and_show_image(self, image_name:str, current_data):
        """
        Save the image using the image_name and show the image information
        using the current data
        """
        cv2.imshow(image_name, current_data)
        cv2.imwrite(image_name, current_data)

    def finish_opencv_session(self):
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Finish all the OpenCV sessions")

    def water_marking_rgb(self):
        data_blue_marking_added = self.adding_marking_grayscale(self.channel_blue)
        data_green_marking_added = self.adding_marking_grayscale(self.channel_green)
        data_red_marking_added = self.adding_marking_grayscale(self.channel_red)
        merged_image = cv2.merge((data_blue_marking_added, data_green_marking_added, data_red_marking_added))
        self.save_and_show_image("RGB_adding_marking_" + self.image_path_name, merged_image)

        return data_blue_marking_added, data_green_marking_added, data_red_marking_added

    def psnr_measurement_rgb(self, original:str, modified:str):
        # Load the image in color (BGR format)
        image_original = cv2.imread(original)
        if image_original is None:
            print("Error: Image could not be loaded.")
            return
        # Split BGR channels
        self.channel_blue, self.channel_green, self.channel_red = cv2.split(image_original)

        # modify
        image_modified = cv2.imread(modified)
        if image_modified is None:
            print("Error: Image could not be loaded.")
            return
        # Split BGR channels
        data_blue_marking_added, data_green_marking_added, data_red_marking_added = cv2.split(image_modified)

        blue_psnr = calculate_psnr(self.channel_blue, data_blue_marking_added)
        green_psnr = calculate_psnr(self.channel_green, data_green_marking_added)
        red_psnr = calculate_psnr(self.channel_red, data_red_marking_added)

        print("The PSNR  is")
        print(f"For blue channel PSNR: {blue_psnr:.2f} dB")
        print(f"For green channel PSNR: {green_psnr:.2f} dB")
        print(f"For red channel PSNR: {red_psnr:.2f} dB")


    def water_marking_rgb_random(self):
        data_blue_marking_added = self.adding_marking_grayscale_random(self.channel_blue)
        data_green_marking_added = self.adding_marking_grayscale_random(self.channel_green)
        data_red_marking_added = self.adding_marking_grayscale_random(self.channel_red)
        merged_image = cv2.merge((data_blue_marking_added, data_green_marking_added, data_red_marking_added))
        self.save_and_show_image("RGB_adding_marking_random_" + self.image_path_name, merged_image)


    def recover_marking_rgb_random(self, path_image_watermarking_rgb:str):
        print("Recover random data RGB")
        bin_vector = []
        index_bin_vector = 0
        len_bin_vector = self.size_watermarking_characters * 8
        print(f"the len of bin vector is {len_bin_vector}")
        # Load the image in color (BGR format)
        image_bgr = cv2.imread(path_image_watermarking_rgb)
        if image_bgr is None:
            print("Error: Image could not be loaded.")
            return

        # Split BGR channels
        channel_blue, channel_green, channel_red = cv2.split(image_bgr)
        row, col = channel_blue.shape

        print("Data recovered from channels")
        for name, channel in zip(['Blue', 'Green', 'Red'], [channel_blue, channel_green, channel_red]):
            while index_bin_vector < len_bin_vector:
                for (r, c) in self.random_position:
                    if index_bin_vector >= len_bin_vector:
                        break
                    bin_vector.append((int(channel[r][c]) & 1))  # (element & 0000 0001)
                    index_bin_vector += 1
                    # print("The index of the bin vector is: ", index_bin_vector)
                # print ("the bin vector recovered is: ", bin_vector)

            chunks = [bin_vector[i:i + 8] for i in range(0, len(bin_vector), 8)]
            marking_recovered = ''.join([chr(int(''.join(map(str, byte)), 2)) for byte in chunks])
            print(f"Recuperado desde {name}:")
            print(marking_recovered)

    def recover_marking_rgb(self, path_image_watermarking_rgb:str):
        len_bin_vector = self.size_watermarking_characters * 8
        print(f"the len of bin vector is {len_bin_vector}")
        # Load the image in color (BGR format)
        image_bgr = cv2.imread(path_image_watermarking_rgb)
        if image_bgr is None:
            print("Error: Image could not be loaded.")
            return

        # Split BGR channels
        channel_blue, channel_green, channel_red = cv2.split(image_bgr)
        row, col = channel_blue.shape
        print("Data recovered from channels")
        for name, channel in zip(['Blue', 'Green', 'Red'], [channel_blue, channel_green, channel_red]):
            bin_vector = []
            index_bin_vector = 0
            while index_bin_vector < len_bin_vector:
                for r in range(row):
                    for c in range(col):
                        if index_bin_vector >= len_bin_vector:
                            break
                        bin_vector.append((int(channel[r][c]) & 1))
                        index_bin_vector += 1

            chunks = [bin_vector[i:i + 8] for i in range(0, len(bin_vector), 8)]
            marking_recovered = ''.join([chr(int(''.join(map(str, byte)), 2)) for byte in chunks])
            print(f"Recuperado desde {name}:")
            print(marking_recovered)
    def plan_zero_rgb(self, original, modified):
        # open the original image
        image_original = cv2.imread(original)
        image_original_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

        # open the modified watermarking added image
        image_modified = cv2.imread(modified)
        image_modified_gray = cv2.cvtColor(image_modified, cv2.COLOR_BGR2GRAY)

        # plan zero for original
        plane_0_original = image_original_gray & 1  # LSB
        # print("The plane 0 is: ", plane_0)
        plane_0_original_vis = (plane_0_original * 255).astype(
            np.uint8)  # Scale plane 0 to be able to display it visually
        self.save_and_show_image("Plane_0_" + original, plane_0_original_vis)

        # plan zero for modified
        plane_0_modified = image_modified_gray & 1  # LSB
        # print("The plane 0 is: ", plane_0)
        plane_0_modified_vis = (plane_0_modified * 255).astype(np.uint8)  # Scale plane 0 to be able to display it visually
        self.save_and_show_image("Plane_0_" + modified, plane_0_modified_vis)

def main_grain_scale_process():
    # Define the characters to choose from (printable characters)
    chars = string.ascii_letters + string.digits + string.punctuation + ' '
    # Generate a random string of exactly 6427 characters
    random_text = ''.join(random.choices(chars, k=8000))
    marking_text = "This is my own watermarking by alex" + random_text
    print("The marking is: ", marking_text)
    name_image_path = "kemonito.png"
    water_marking = WaterMarking(marking=marking_text, image_path_name=name_image_path)

    # adding watermarking LSB
    data_grey = water_marking.open_show_image_grayscale(save=True)
    print("The size of the image is:", data_grey.shape)
    data_gray_marking_added = water_marking.adding_marking_grayscale(data_grey)
    water_marking.save_and_show_image("water_marking_" + name_image_path, data_gray_marking_added)
    water_marking.recover_marking_grayscale("water_marking_" + name_image_path)
    water_marking.plane_zero("water_marking_" + name_image_path)
    water_marking.psrn_measurement(original=name_image_path, modified="water_marking_" + name_image_path)

    # adding watermarking LSB in random
    data_grey = water_marking.open_show_image_grayscale(save=False)
    data_gray_marking_added_random = water_marking.adding_marking_grayscale_random(data_grey)
    water_marking.save_and_show_image("water_marking_random_" + name_image_path, data_gray_marking_added_random)
    # water_marking.recover_marking_grayscale("platform_validation_water_marking.png")
    water_marking.plane_zero("water_marking_random_" + name_image_path)
    water_marking.recover_marking_grayscale_random("water_marking_random_" + name_image_path)
    water_marking.plane_zero(name_image_path)
    water_marking.psrn_measurement(original=name_image_path,
                                   modified="water_marking_random_" + name_image_path)
    water_marking.finish_opencv_session()

def main_rgb_process():
     # Define the characters to choose from (printable characters)
     chars = string.ascii_letters + string.digits + string.punctuation + ' '
     # Generate a random string of exactly 6427 characters
     random_text = ''.join(random.choices(chars, k=8000))
     marking_text = "This is my own water marking " + random_text
     print("The marking is: ", marking_text)
     image_path = "kemonito.png"
     water_marking = WaterMarking(marking=marking_text, image_path_name=image_path)
     water_marking.load_show_rgb_channels_opencv(image_path)
     blue_added_marking, red_added_marking, green_added_marking = water_marking.water_marking_rgb()
     water_marking.recover_marking_rgb("RGB_adding_marking_" + image_path)
     water_marking.psnr_measurement_rgb(image_path, "RGB_adding_marking_" + image_path)

    # random
     water_marking.water_marking_rgb_random()
     water_marking.recover_marking_rgb_random("RGB_adding_marking_random_" + image_path)
     water_marking.psnr_measurement_rgb(image_path, "RGB_adding_marking_random_" + image_path)

    # plan zero
     water_marking.plan_zero_rgb(original=image_path, modified="RGB_adding_marking_" + image_path)
     water_marking.plan_zero_rgb(original=image_path, modified="RGB_adding_marking_random_" + image_path)
     water_marking.finish_opencv_session()

if __name__ == "__main__":
    # uncomment according which process would you like to execute
    #main_grain_scale_process()
    main_rgb_process()


