import cv2
import numpy as np
import random
import string



class WaterMarking:
    def __init__(self, marking:str, image_path_name:str='platform_validation.png' ):
        self.marking = marking
        self.size_watermarking_characters = len(marking)
        self.image_path_name = image_path_name
        self.image_gray_data = None

    def open_show_image_grayscale(self, image_path:str = None):
        """
        Open an image according to the image_name, and only shows it in grayscale
        after that it save the image in this scale
        """
        if image_path is None:
            image_path = self.image_path_name

        image_opened = cv2.imread(image_path)
        image_opened_gray = cv2.cvtColor(image_opened, cv2.COLOR_BGR2GRAY)
        cv2.imshow(image_path, image_opened_gray)
        cv2.imwrite(str("grayscale_") + image_path, image_opened_gray)
        cv2.imshow(str("grayscale_") + image_path, image_opened_gray)
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

    def recover_marking_grayscale(self, path_image_watermarking_grayscale:str):
        """
        Recover the data from the image
        :param path_image_watermarking_grayscale:
        :return:
        """

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
        print ("the bin vector recovered is: ", bin_vector)

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
        print("The bit vector is: ", bin_vector)
        return bin_vector

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

if __name__ == "__main__":
    # Define the characters to choose from (printable characters)
    chars = string.ascii_letters + string.digits + string.punctuation + ' '
    # Generate a random string of exactly 6427 characters
    random_text = ''.join(random.choices(chars, k=6427))
    marking_text = random_text
    print("The marking is: ", marking_text)
    water_marking = WaterMarking(marking = marking_text, image_path_name = "platform_validation.png")

    # End the OpenCV session
    data_grey = water_marking.open_show_image_grayscale()
    data_gray_marking_added = water_marking.adding_marking_grayscale(data_grey)
    water_marking.save_and_show_image("platform_validation_water_marking.png", data_gray_marking_added)
    water_marking.recover_marking_grayscale("platform_validation_water_marking.png")
    water_marking.finish_opencv_session()

    print(data_grey)