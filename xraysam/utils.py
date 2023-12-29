
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SamProcessor

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

class SAMDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.images = sorted([os.path.join(img_dir, file) for file in os.listdir(img_dir)])
        self.masks = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(image_path).convert("RGB")  # Ensure the image is RGB.

        # Open the mask image and convert it to grayscale
        mask = Image.open(mask_path).convert('L')

        # Convert the image to a numpy array
        ground_truth_mask = np.array(mask)

        # Ensure the mask is 2D right after loading and before any transformation.
        if len(ground_truth_mask.shape) != 2:
            raise ValueError(f"Mask at {mask_path} is not 2D. Shape found: {ground_truth_mask.shape}")

        # Normalize the mask to have values only 0 or 1
        ground_truth_mask = (ground_truth_mask > 0).astype(np.int8)

        # Convert the numpy array to a torch tensor
        ground_truth_mask = torch.from_numpy(ground_truth_mask).float()

        # Add a batch dimension (1, height, width) for interpolation
        ground_truth_mask = ground_truth_mask.unsqueeze(0).unsqueeze(0)

        # Resize the mask to 256x256 using PyTorch
        ground_truth_mask = F.interpolate(ground_truth_mask, size=(256, 256), mode='nearest')

        # Remove the batch dimension to get the final mask (1, 256, 256)
        ground_truth_mask = ground_truth_mask.squeeze(0).squeeze(0)

        # Ensure the mask is 2D.
        if len(ground_truth_mask.shape) != 2:
            raise ValueError(f"Mask at {mask_path} is not 2D. Shape found: {ground_truth_mask.shape}")

        # get bounding box prompt
        prompt = get_bounding_box(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(images=image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs
