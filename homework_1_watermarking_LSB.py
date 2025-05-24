import cv2
import numpy as np

class WaterMarking:
    def __init__(self, marking:str, image_path_name:str='platform_validation.png' ):
        self.marking = marking
        self.image_path_name = image_path_name
        self.image_gray_data = None

    def open_show_image_grayscale(self):
        """
        Open an image according to the image_name, and only shows it in grayscale
        after that it save the image in this scale
        """
        image_opened = cv2.imread(self.image_path_name)
        image_opened_gray = cv2.cvtColor(image_opened, cv2.COLOR_BGR2GRAY)
        cv2.imshow(self.image_path_name, image_opened_gray)
        cv2.imwrite(str("grayscale_") + self.image_path_name, image_opened_gray)
        cv2.imshow(str("grayscale_") + self.image_path_name, image_opened_gray)
        self.image_gray_data = image_opened_gray

    def adding_marking(self, data_layer:np.ndarray):
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
    water_marking = WaterMarking(marking = "Hi I'm a WaterMarking ", image_path_name = "platform_validation.png")
    water_marking.open_show_image_grayscale()
    # End the OpenCV session
    data_grey = water_marking.image_gray_data
    print(data_grey.shape)
    print(data_grey[0][1])

    print(data_grey.dtype)
    new_element = int(data_grey[0][1])
    bit = 0
    new = (new_element & ~1) | bit # (new_element & 1111 1110) | bit
    letter = "a"
    print(f"the letter is:{letter} and int {ord(letter)}, the ")
    print(new)

    #water_marking.save_and_show_image(marking, data_grey)
    water_marking.adding_marking(data_grey)
    water_marking.finish_opencv_session()

    print(data_grey)