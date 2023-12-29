
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import SamProcessor, SamModel
import numpy as np
from tqdm import tqdm

from utils import SAMDataset
from loss import LungLoss
from metrics import calculate_metrics

# Initialize the SAM processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Set your image and mask directories
train_img_dir = "/content/train_images"
train_mask_dir = "/content/train_masks"
valid_img_dir = "/content/val_images"
valid_mask_dir = "/content/val_masks"
test_img_dir = "/content/test_images"
test_mask_dir = "/content/test_masks"

# Create the dataset and DataLoader for training, validation, and test sets
train_dataset = SAMDataset(img_dir=train_img_dir, mask_dir=train_mask_dir, processor=processor)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

valid_dataset = SAMDataset(img_dir=valid_img_dir, mask_dir=valid_mask_dir, processor=processor)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=True)

test_dataset = SAMDataset(img_dir=test_img_dir, mask_dir=test_mask_dir, processor=processor)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

# Load the pretrained weights for finetuning
model = SamModel.from_pretrained("facebook/sam-vit-base")

# Freeze encoder weights
for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=0)
loss_fn = LungLoss()

# Training parameters
num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_losses, val_losses = [], []
    train_acc, train_dice, train_iou = [], [], []

    for batch in tqdm(train_dataloader):
        images = batch["pixel_values"].to(device)
        masks = batch["ground_truth_mask"].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=images, input_boxes=batch["input_boxes"].to(device), multimask_output=False)
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = masks.unsqueeze(1)
        if ground_truth_masks.ndim == 5:
            ground_truth_masks = ground_truth_masks.squeeze(2)
        ground_truth_masks = F.interpolate(ground_truth_masks, size=predicted_masks.shape[-2:], mode='nearest')
        loss = loss_fn(predicted_masks, ground_truth_masks)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        accuracy, dice, iou = calculate_metrics(predicted_masks, ground_truth_masks)
        train_acc.append(accuracy)
        train_dice.append(dice)
        train_iou.append(iou)

    # Validation
    model.eval()
    val_acc, val_dice, val_iou = [], [], []
    with torch.no_grad():
        for val_batch in valid_dataloader:
            images = val_batch["pixel_values"].to(device)
            masks = val_batch["ground_truth_mask"].to(device)
            outputs = model(pixel_values=images, input_boxes=val_batch["input_boxes"].to(device), multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = masks.unsqueeze(1)
            if ground_truth_masks.ndim == 5:
                ground_truth_masks = ground_truth_masks.squeeze(2)
            ground_truth_masks = F.interpolate(ground_truth_masks, size=predicted_masks.shape[-2:], mode='nearest')
            val_loss = loss_fn(predicted_masks, ground_truth_masks)
            val_losses.append(val_loss.item())
            accuracy, dice, iou = calculate_metrics(predicted_masks, ground_truth_masks)
            val_acc.append(accuracy)
            val_dice.append(dice)
            val_iou.append(iou)

    # Calculate mean stats
    mean_train_loss = np.mean(train_losses)
    mean_val_loss = np.mean(val_losses)
    mean_train_acc = np.mean(train_acc)
    mean_train_dice = np.mean(train_dice)
    mean_train_iou = np.mean(train_iou)
    mean_val_acc = np.mean(val_acc)
    mean_val_dice = np.mean(val_dice)
    mean_val_iou = np.mean(val_iou)

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {mean_train_loss}, Train Acc: {mean_train_acc}, Train Dice: {mean_train_dice}, Train IoU: {mean_train_iou}')
    print(f'Val Loss: {mean_val_loss}, Val Acc: {mean_val_acc}, Val Dice: {mean_val_dice}, Val IoU: {mean_val_iou}')

    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        torch.save(model.state_dict(), "best_model_weights.pth")
        print("Best model saved with loss: ", best_val_loss)
