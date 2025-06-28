"""
Advance Security and Watermarking
Final Project: Robust Image Protection through Watermark
Embedding using DCT, Shamir Secret Sharing, and Perceptual Hash Verification

By:Alejandro Salinas V.


The shamir code was taken from:
https://github.com/handsomelky/Shamir-Image-Secret-Sharing/tree/main
"""
from DTC.dtc import StegoDCT
from Shamir.shamir_secret import ShamirSecretSharing

def main_process():

    # PARAMETERS
    image_cover_name_rgb = "image.png" # cover image
    watermarking_image = "alex_watermarking.png" # watermarking
    stego_name = "stego_dct_" + image_cover_name_rgb
    shamir_decoded = "decoded_sss_" + stego_name
    n_shares = 6 # n shares for SSS
    k_shares = 3 # k shares for SSS
    ## PROCESS
    print(" ***** Main Process *****")
    print(" ***** Generating Stego *****")
    stego = StegoDCT(img_path=image_cover_name_rgb)
    stego.load_image_rgb()

    wm_h, wm_w = stego.embed_rgb_image_watermark(watermarking_image, stego_name)
    print(" ***** Stego Generated *****")
    print(" ----- Shamir Shared Secret ----- ")
    shamir = ShamirSecretSharing(stego_name)
    print(" ----- Generating Secrets ----- ")
    print(f"There are n = {n_shares} shares and k = {k_shares} shares ")
    shamir.encoding(n_shares, k_shares)

    print(" ----- Recover the Shamir Secret ----- ")
    shamir.decoding(decode_img=shamir_decoded, index=[1, 6, 2])

    print(" ----- Extracting the watermarking from stego and shamir secret sharing ----- ")
    stego.extract_rgb_image_watermark(wm_shape=(wm_h, wm_w),
                                                     stego_path=shamir_decoded)

    stego.finish_opencv_session()

if __name__ == '__main__':
    main_process()
