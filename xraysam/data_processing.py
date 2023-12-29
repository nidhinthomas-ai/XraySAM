import argparse
from PIL import Image
import os
import numpy as np

class DataProcessor:
    def __init__(self):
        self.args = self.parse_args()

        self.base_dir = self.args.base_dir
        self.train_ratio, self.val_ratio, self.test_ratio = self.args.split_ratios
        self.output_dir = self.args.output_dir

        self.image_dir = os.path.join(self.base_dir, 'images')
        self.mask_dir = os.path.join(self.base_dir, 'masks')
        self.processed_image_dir = os.path.join(self.output_dir, 'processed_images')
        self.processed_mask_dir = os.path.join(self.output_dir, 'processed_masks')

        os.makedirs(self.processed_image_dir, exist_ok=True)
        os.makedirs(self.processed_mask_dir, exist_ok=True)

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Process and split image data for lung segmentation.')
        parser.add_argument('--base_dir', type=str, required=True, help='Path to the base directory containing images and masks')
        parser.add_argument('--split_ratios', type=float, nargs=3, default=[0.8, 0.1, 0.1], help='Split ratios for train, validation, and test sets. Example: 0.8 0.1 0.1')
        parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed and split data')
        return parser.parse_args()

    def resize_image(self, input_path, output_path, size, mode):
        with Image.open(input_path) as img:
            img = img.resize(size, Image.ANTIALIAS)
            img = img.convert(mode)
            img.save(output_path)

    def load_image(self, path):
        return np.array(Image.open(path))

    def process_and_split(self):
        image_paths = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith('.png')])
        mask_paths = sorted([os.path.join(self.mask_dir, f) for f in os.listdir(self.mask_dir) if f.endswith('.png')])

        assert len(image_paths) == len(mask_paths), "Number of images and masks should be the same"

        total_images = len(image_paths)
        train_split = int(total_images * self.train_ratio)
        val_split = int(total_images * (self.train_ratio + self.val_ratio))

        for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
            resized_img_path = os.path.join(self.processed_image_dir, os.path.basename(img_path))
            resized_mask_path = os.path.join(self.processed_mask_dir, os.path.basename(mask_path))
            self.resize_image(img_path, resized_img_path, (1024, 1024), 'RGB')
            self.resize_image(mask_path, resized_mask_path, (1024, 1024), 'L')

            img = self.load_image(resized_img_path)
            mask = self.load_image(resized_mask_path)

            if idx < train_split:
                img_dir = os.path.join(self.output_dir, 'train_images')
                mask_dir = os.path.join(self.output_dir, 'train_masks')
            elif idx < val_split:
                img_dir = os.path.join(self.output_dir, 'val_images')
                mask_dir = os.path.join(self.output_dir, 'val_masks')
            else:
                img_dir = os.path.join(self.output_dir, 'test_images')
                mask_dir = os.path.join(self.output_dir, 'test_masks')

            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            img_slice_path = os.path.join(img_dir, os.path.basename(img_path))
            mask_slice_path = os.path.join(mask_dir, os.path.basename(mask_path))
            Image.fromarray(img).save(img_slice_path)
            Image.fromarray(mask).save(mask_slice_path)

            print(f'Processed and saved image and mask for index {idx}: {img_slice_path} & {mask_slice_path}')

if __name__ == '__main__':
    processor = DataProcessor()
    processor.process_and_split()
    print("Image and Mask processing and splitting complete.")
