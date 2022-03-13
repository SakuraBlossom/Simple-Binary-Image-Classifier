
import argparse
import numpy as np
from pathlib import Path

from PIL import Image

from tqdm import tqdm
from glob import glob

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from infer import SETTINGS_TILE_SIZE, SETTINGS_PATH_DATASET_IMAGES, NORMALISER, readMask, readImage, getTiles, display, getDevice
from model import build_model

EPOCHS=50
BATCH_SIZE=16

class MyBinaryClassDataset(Dataset):
    def __init__(self, x, y, oversample_ratio:float=None):
        
        pos_case = [(x[idx], y[idx]) for idx in range(len(x)) if y[idx] > 0.5]
        neg_case = [(x[idx], y[idx]) for idx in range(len(x)) if y[idx] < 0.5]

        # Oversample
        if oversample_ratio is None:
            self.x = x
            self.y = y

        else:
            print("Oversampling")
            print(f"Size before: {len(x)} - {len(pos_case)} Blur, {len(neg_case)} Sharp")

            imbalance_num_instances = int(len(neg_case)  - len(pos_case) * oversample_ratio)
            if imbalance_num_instances > 0:
                # More negative than positive classes
                # Oversample positive classes
                choices_indices = np.random.choice(len(pos_case), imbalance_num_instances, replace=True)
                pos_case.extend([pos_case[idx] for idx in choices_indices])

            elif imbalance_num_instances < 0:
                # More positive than negative classes
                # Oversample negative classes
                choices_indices = np.random.choice(len(pos_case), -imbalance_num_instances, replace=True)
                pos_case.extend([pos_case[idx] for idx in choices_indices])

            self.x = [case[0] for case in pos_case]
            self.y = [case[1] for case in pos_case]

            self.x.extend([case[0] for case in neg_case])
            self.y.extend([case[1] for case in neg_case])


        print(f"Dataset Size: {len(self.x)} - {len(pos_case)} Blur, {len(neg_case)} Sharp")
        print()


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_all_filenames():
    return [Path(file_path).stem for file_path in glob(f"{SETTINGS_PATH_DATASET_IMAGES}*.jpg")]


def readDataset(overlap_factor : float):

    list_filenames = get_all_filenames()
    print(f"Found {len(list_filenames)} images...")

    dataset_img = list()
    dataset_mask = list()

    for filename in tqdm(list_filenames, desc="Loading dataset"):
        raw_img = readImage(filename)
        raw_mask = readMask(filename)
        _, _, tiles_imgs = getTiles(raw_img, overlap_factor, NORMALISER)
        _, _, tiles_mask = getTiles(raw_mask, overlap_factor, None)
        
        dataset_img.extend(tiles_imgs)
        for mask in tiles_mask:
            dataset_mask.append(1 if (np.count_nonzero(mask) > 0.5 * SETTINGS_TILE_SIZE**2) else 0)

        del raw_img, raw_mask, tiles_imgs, tiles_mask

    assert (len(dataset_mask) == len(dataset_img))

    return MyBinaryClassDataset(dataset_img, dataset_mask, 0.8)


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='my_model.pt',
                        help='name of output trained pytorch model')
    parser.add_argument('--overlap', '-o', type=float, default='0.5',
                        help='Percentage of overlap per tile [0, 0.9] inclusive')
    params = parser.parse_args(raw_args)

    # Check if GPU is available
    device = getDevice()

    # Construct model
    model = build_model()
    model = model.to(device)

    # Read dataset image
    my_train_dataset = readDataset(params.overlap)
    train_dataloader = DataLoader(my_train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=4)

    # Get Stats of the model to be trained
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'{total_params:,} total parameters.')
    print(f'{total_trainable_params:,} training parameters.')
    print()

    criterion = nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01)

    best_loss = 100000
    best_model_state = None
    print(f"Training model for {EPOCHS} epochs - Batch size {BATCH_SIZE}")
    for epoch_idx in range(EPOCHS):
        train_loss = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_idx}")
        for batch_idx, (x, y) in enumerate(pbar):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.reshape((-1, 1)).type(torch.float).to(device)
            y_pred = model.forward(x)
            
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=train_loss/(batch_idx+1))

        train_loss = train_loss/len(train_dataloader)
        
        if float(train_loss) < best_loss:
            best_loss = float(train_loss)
            best_model_state = model.state_dict()
        
        if train_loss < 0.01:
            break

    print("Done Training!")
    if best_model_state:
        print(f"Least loss: {best_loss}")
        model.load_state_dict(best_model_state)
        torch.save(model, params.model)

    return model

if __name__ == "__main__":
    main()
