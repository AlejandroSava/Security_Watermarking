"""
The shamir code was taken from:
https://github.com/handsomelky/Shamir-Image-Secret-Sharing/tree/main
"""

import time
import numpy as np
import png
import sys
import os
from PIL import Image
from Crypto.Util.number import *


class ShamirSecretSharing:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_shape = None
        self.n_shares = None
        self.k_shares = None

    def preprocessing(self, path=None):
        if path is None:
            path = self.image_path
        img = Image.open(path)
        data = np.asarray(img)
        self.image_shape = data.shape
        return data.flatten(), data.shape

    def polynomial(self, img, n, k):
        num_pixels = img.shape[0]
        # 生成多项式系数
        coefficients = np.random.randint(low=0, high=257, size=(num_pixels, k - 1))
        secret_imgs = []
        imgs_extra = []
        for i in range(1, n + 1):
            # 构造(r-1)次多项式
            base = np.array([i ** j for j in range(1, k)])
            base = np.matmul(coefficients, base)

            secret_img = (img + base) % 257

            indices = np.where(secret_img == 256)[0]
            img_extra = indices.tolist()
            secret_img[indices] = 0

            secret_imgs.append(secret_img)
            imgs_extra.append(img_extra)
        return np.array(secret_imgs), imgs_extra

    def insert_text_chunk(self, src_png, dst_png, text):
        '''在png中的第二个chunk插入自定义内容'''
        reader = png.Reader(filename=src_png)
        chunks = reader.chunks()  # 创建一个每次返回一个chunk的生成器
        chunk_list = list(chunks)
        chunk_item = tuple([b'tEXt', text])

        index = 1
        chunk_list.insert(index, chunk_item)

        with open(dst_png, 'wb') as dst_file:
            png.write_chunks(dst_file, chunk_list)

    def get_file_size(self, file_path):
        """ 获取文件大小并格式化输出 """
        try:
            size = os.path.getsize(file_path)
            return self.format_size(size)
        except OSError as e:
            return f"Error: {e}"

    def read_text_chunk(self, src_png, index=1):
        '''读取png的第index个chunk'''
        reader = png.Reader(filename=src_png)
        chunks = reader.chunks()
        chunk_list = list(chunks)
        img_extra = chunk_list[index][1].decode()
        img_extra = eval(img_extra)
        return img_extra

    def lagrange(self, x, y, num_points, x_test):
        l = np.zeros(shape=(num_points,))
        for k in range(num_points):

            l[k] = 1
            for k_ in range(num_points):

                if k != k_:
                    d = int(x[k] - x[k_])
                    inv_d = inverse(d, 257)
                    l[k] = l[k] * (x_test - x[k_]) * inv_d % 257

                else:
                    pass
        L = 0
        for i in range(num_points):
            L += y[i] * l[i]
        return L

    def decode(self,imgs, imgs_extra, index, r):
        assert imgs.shape[0] >= r
        x = np.array(index)
        dim = imgs.shape[1]
        img = []

        print("decoding:")
        last_percent_reported = None
        imgs_add = np.zeros_like(imgs, dtype=np.int32)
        for i in range(r):
            for indices in imgs_extra[i]:
                imgs_add[i][indices] = 256

        for i in range(dim):
            y = imgs[:, i]
            ex_y = imgs_add[:, i]
            y = y + ex_y
            pixel = self.lagrange(x, y, r, 0) % 257
            img.append(pixel)

            # 计算当前进度
            percent_done = (i + 1) * 100 // dim
            if last_percent_reported != percent_done:
                if percent_done % 1 == 0:
                    last_percent_reported = percent_done
                    bar_length = 50
                    block = int(bar_length * percent_done / 100)
                    text = "\r[{}{}] {:.2f}%".format("█" * block, " " * (bar_length - block), percent_done)
                    sys.stdout.write(text)
                    sys.stdout.flush()

        print()
        return np.array(img)

    def format_size(self, size_bytes):
        """ 根据字节大小自动调整单位 """
        if size_bytes == 0:
            return "0B"
        size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    def compare_images(self, image1_path, image2_path):
        print("\n=== Starting image comparison ===")
        image1 = np.array(Image.open(image1_path))
        image2 = np.array(Image.open(image2_path))
        diff = np.abs(image1 - image2)
        diff_value = round(np.mean(diff), 4)
        print("Mean difference:", diff_value)
        print("Max difference:", round(np.max(diff), 4))
        print("Min difference:", round(np.min(diff), 4))
        print("Standard deviation of difference:", round(np.std(diff), 4))
        print("=== Image comparison completed.  ===")

    ####################### main methods

    def encoding(self, n_shares = None, k_shares=None ):
        start_time = time.time()
        print("\n=== Starting image encoding process ===")
        if n_shares is None:
            print("Error: Total number 'n' of shares is required for decoding")
            return False
        else:
            self.n_shares = n_shares

        if k_shares is None:
            print("Error: Threshold number 'k' is required for decoding")
            return False
        else:
            self.k_shares = k_shares

        if k_shares > n_shares:
            print("Error: Threshold 'k' cannot be greater than the total number 'n' of shares")
            return False

        img_flattened, shape = self.preprocessing(self.image_path)
        secret_imgs, imgs_extra = self.polynomial(img_flattened, n=self.n_shares , k=self.k_shares)
        to_save = secret_imgs.reshape(self.n_shares, *shape)
        for i, img in enumerate(to_save):
            secret_img_path = f"secret_{i+1}_{self.image_path}"
            Image.fromarray(img.astype(np.uint8)).save(secret_img_path)
            img_extra = str(list((imgs_extra[i]))).encode()
            self.insert_text_chunk(secret_img_path, secret_img_path, img_extra)
            size = self.get_file_size(secret_img_path)
            print(f"{secret_img_path} saved.", size)

        end_time = time.time()
        print("=== Image encoding completed. Time elapsed: {:.2f} seconds ===".format(end_time - start_time))

    def decoding(self, decode_img:str, index:list, k_decode = None):
        start_time = time.time()
        print("\n=== Starting image decoding process ===")

        if k_decode is None:
            print("Threshold number 'k' is required for decoding, taking default in the class")
            k_decode = self.k_shares


        input_imgs = []
        input_imgs_extra = []
        for i in index:
            secret_img_path = f"secret_{i}_{self.image_path}"
            img_extra = self.read_text_chunk(secret_img_path)
            img, shape = self.preprocessing(secret_img_path)
            input_imgs.append(img)
            input_imgs_extra.append(img_extra)
        input_imgs = np.array(input_imgs)
        origin_img = self.decode(input_imgs, input_imgs_extra, index, r=k_decode)
        origin_img = origin_img.reshape(*shape)
        Image.fromarray(origin_img.astype(np.uint8)).save(decode_img)
        size = self.get_file_size(decode_img)
        print(f"{decode_img} saved.", size)

        end_time = time.time()
        print("=== Image decoding completed. Time elapsed: {:.2f} seconds ===".format(end_time - start_time))


def main():
    image_path = "stego_rgb_image_dtc_image.png"
    shamir = ShamirSecretSharing(image_path)
    shamir.encoding(n_shares=7, k_shares=3)
    shamir.decoding(decode_img="decoded_" + image_path, index=[1, 6, 2] )
    #img_a = "recovered_rgb_watermark.png"
    #img_b = "scaling_alex_watermarking.png"
    #shamir.compare_images(img_a, img_b)

if __name__ == "__main__":
    main()

