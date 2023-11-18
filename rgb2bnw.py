from PIL import Image
import os
from tqdm import tqdm
import argparse

def convert_to_bnw(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_list = sorted(os.listdir(input_folder))

    for file_name in tqdm(file_list):
        input_path = os.path.join(input_folder, file_name)
        image = Image.open(input_path)

        # convert img to grayscale
        bnw_image = image.convert("L")

        # convert grayscale img to RGB by replicating the single channels
        bnw_image = bnw_image.convert("RGB")
        
        # Converted Grayscale images have the below properties:
        # value range: [0, 255]
        # num of channels: 3

        output_path = os.path.join(output_folder, file_name)
        bnw_image.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', type=str, default='', help='')
    parser.add_argument('--output_path', type=str, default='', help='')
    args = parser.parse_args()
    
    # input_folder = "/Users/gohyixian/Desktop/LLVIP/visible/test"
    # output_folder = "/Users/gohyixian/Desktop/LLVIP_visible_test_bnw"
    convert_to_bnw(args.input_path, args.output_path)
    print("DONE")
