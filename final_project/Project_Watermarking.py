"""
Advance Security and Watermarking
Final Project: Robust Image Protection through Watermark
Embedding using DCT, Shamir Secret Sharing, and Perceptual Hash Verification

By:Alejandro Salinas V.


The shamir code was taken from:
https://github.com/handsomelky/Shamir-Image-Secret-Sharing/tree/main
"""
from DTC.dtc import StegoDCT, psnr_measurement
from Shamir.shamir_secret import ShamirSecretSharing
from Perceptual_Hash.Perceptual_Hash import PerceptualHash

def main_process():

    # PARAMETERS
    image_cover_name_rgb = "image.png" # cover image
    watermarking_image = "alex_watermarking.png" # watermarking
    watermarking_recovered = "recover_" + watermarking_image
    stego_name = "stego_dct_" + image_cover_name_rgb
    shamir_decoded = "decoded_sss_" + stego_name
    n_shares = 6 # n shares for SSS
    k_shares = 3 # k shares for SSS
    ## PROCESS First Part
    print(" ***** Main Process *****")
    print(" ***** Generating Stego *****")
    stego = StegoDCT(img_path=image_cover_name_rgb)
    stego.load_image_rgb()

    wm_h, wm_w = stego.embed_rgb_image_watermark(watermarking_image, stego_name)
    print(" ***** Stego Generated *****")


    print(" ----- Shamir Shared Secret ----- ")
    shamir = ShamirSecretSharing(stego_name)
    print("\n ----- Generating Secrets ----- ")
    print(f"There are n = {n_shares} shares and k = {k_shares} shares ")
    shamir.encoding(n_shares, k_shares)


    # Second Part
    print("\n ----- Recover the Shamir Secret ----- ")
    shamir.decoding(decode_img=shamir_decoded, index=[1, 2, 6])

    print(" ----- Extracting the watermarking from stego and shamir secret sharing ----- ")
    stego.extract_rgb_image_watermark(wm_shape=(wm_h, wm_w),stego_path=shamir_decoded, output_path=watermarking_recovered)

    # third Phase, verification
    print("\n ----- Perceptual Hash ----- ")
    perceptual_hash = PerceptualHash(path_img_1=watermarking_image, path_img_2=watermarking_recovered)
    perceptual_hash.hamming_distance()
    perceptual_hash.similarity()
    perceptual_hash.valid_image()


    # results:
    print(f"\n ***** PSNR between {image_cover_name_rgb} and {stego_name}*****")
    psnr_measurement(image_cover_name_rgb, stego_name, grayscale=False)


    print(f" ***** PSNR between {"scaling_"+watermarking_image} and {watermarking_recovered}*****")
    psnr_measurement("scaling_"+watermarking_image, watermarking_recovered, grayscale=False)

    ### getting the difference pixel per pixel
    print(f" ***** Getting the difference pixel per pixel from: {watermarking_image} and {watermarking_recovered}*****")
    shamir.compare_images(image1_path="scaling_"+watermarking_image, image2_path=watermarking_recovered)


    ## adding watermarking attacks
    compression = 100
    jpeg_30 = stego.jpeg_compression(stego_name, compression)
    jpeg_30_wm_recovered = f"jpeg_{compression}_wm_recovered_compression.jpeg"
    print(" ----- Extracting the watermarking from jpeg compression ----- ")
    stego.extract_rgb_image_watermark(wm_shape=(wm_h, wm_w), stego_path=jpeg_30,
                                      output_path=jpeg_30_wm_recovered)
    print(f"\n ***** PSNR between {watermarking_image} and {jpeg_30_wm_recovered} *****")
    psnr_measurement("scaling_"+watermarking_image, jpeg_30_wm_recovered, grayscale=False)

    # salt and pepper
    salt = 0.01
    pepper = 0.01
    noisy_image = stego.add_salt_pepper_noise(stego_name,salt, pepper)
    noisy_image_wm_recovered = "wm_recovered_" + noisy_image
    print(" ----- Extracting the watermarking from salt and pepper noise ----- ")
    stego.extract_rgb_image_watermark(wm_shape=(wm_h, wm_w), stego_path=noisy_image,
                                      output_path=noisy_image_wm_recovered)
    print(f"\n ***** PSNR between {watermarking_image} and {noisy_image_wm_recovered} *****")
    psnr_measurement("scaling_" + watermarking_image, noisy_image_wm_recovered, grayscale=False)


    #finishing the session
    stego.finish_opencv_session()

if __name__ == '__main__':
    main_process()
