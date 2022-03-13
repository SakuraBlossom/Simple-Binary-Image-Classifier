import torch
from torchvision import transforms
import argparse
import numpy as np
from PIL import Image
from itertools import product
from typing import List, Tuple

SETTINGS_TILE_SIZE = 224
SETTINGS_PATH_DATASET_IMAGES = "./dataset/image/"
SETTINGS_PATH_DATASET_MASKS = "./dataset/mask/"

#NORMALISER = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # magic numbers - the mean and stdev from imagenet dataset
NORMALISER = None

def readImage(img_name : str):
    img = Image.open(f"{SETTINGS_PATH_DATASET_IMAGES}{img_name}.jpg")
    return np.asarray(img)


def readMask(mask_name : str):
    img = Image.open(f"{SETTINGS_PATH_DATASET_MASKS}{mask_name}.png")
    return np.asarray(img)


def display(img : np.array, title : str):
    is_mono = (len(img.shape) == 2)
    pil_img = Image.fromarray(np.uint8(img)).convert('RGB') if is_mono else Image.fromarray(img, 'RGB')
    pil_img.show(title=title)
    return pil_img

def resizeImage(img : np.ndarray, shape : Tuple[int, int]):
    # Should use cv2.resize but not sure if everyone is willing to install opencv for this one function
    img = Image.fromarray(img)
    img = img.resize(size=shape)
    return np.asarray(img)


def getDevice():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        print()

    return device


def getTiles(img : np.array, overlap_factor : float, normalise : object):
    '''
    Args:
        img: An image in numpy (H, W, C) of uint8 type
        overlap_factor: Percentage of tile overlap
    Return:
        heatmap_size: The expected size of output (H, W)
        tile_pos:  A List of tuple of 2d (row, col) coords which corresponds to each element in the column tensor 
        y: list of tiles (H_out, W_out, C)
    '''

    convert_tensor = transforms.ToTensor()
    in_height, in_width = img.shape[:2]

    # get stride size
    stride = int(np.clip((1.0 - overlap_factor)*SETTINGS_TILE_SIZE, 1, SETTINGS_TILE_SIZE*0.9))

    # get padding
    padding_width = -in_width % SETTINGS_TILE_SIZE
    padding_height = -in_height % SETTINGS_TILE_SIZE

    padding_left   = padding_width // 2
    padding_right  = padding_width - padding_left
    padding_top    = padding_height // 2
    padding_bottom = padding_height - padding_top

    padding = [(padding_top,padding_bottom), (padding_left,padding_right)]
    if len(img.shape) == 3:
        padding.append((0,0))

    img_pad = np.pad(img, padding, mode='constant')
    img_pad_shape = np.array(img_pad.shape[:2])

    # perform tiling
    img_out_shape = (img_pad_shape - SETTINGS_TILE_SIZE - 1) // stride + 1
    h_out, w_out = img_out_shape

    # Convert to list of image tiles
    tiles_pos = list(product(range(h_out), range(w_out)))
    tiles_pixels = [(h_idx * stride, w_idx * stride) for h_idx, w_idx in tiles_pos]
    output_tiles = [convert_tensor(img_pad[h_offset:h_offset+SETTINGS_TILE_SIZE, w_offset:w_offset+SETTINGS_TILE_SIZE]) for h_offset, w_offset in tiles_pixels]
    if normalise:
        output_tiles = [normalise(tile) for tile in output_tiles]

    return img_out_shape, tiles_pos, output_tiles


def convertToHeatMap(heatmap_size : Tuple[int, int], tile_pos : List[Tuple[int, int]], x : torch.tensor, threshold : float):
    '''
    Args:
        x: A boolean torch tensor with size (N, 1),
        tile_pos: A List of tuple of 2d (row, col) coords which corresponds to each element in the column tensor 
        heatmap_size: A tuple of the size of the output heatmap
    Return:
        y: torch tensor of size (N, C_out, H_out, W_out)
    '''

    out_heatmap = np.zeros(heatmap_size, dtype=np.uint8)
    x = x.cpu().detach().numpy()
    for idx, (row, col) in enumerate(tile_pos):
        assert 0 <= row < heatmap_size[0]
        assert 0 <= col < heatmap_size[1]

        out_heatmap[row, col] = 255 if (x[idx] >= threshold) else 0

    return out_heatmap


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='my_model.pt',
                        help='relative path to trained pytorch model')
    parser.add_argument('--overlap', '-o', type=float, default='0.5',
                        help='Percentage of overlap per tile [0, 0.9] inclusive')
    parser.add_argument('--threshold', '-t', type=float, default='0.5',
                        help='Threshold for blurry classification [0, 1.0] inclusive')
    parser.add_argument('image', type=str, help="file path to target image to run inference on")
    params = parser.parse_args(raw_args)

    # Load target image
    target_image = readImage(params.image)

    # Split image into tiles
    heatmap_size, tiles_pos, tiles_img = getTiles(target_image, params.overlap, NORMALISER)
    print(heatmap_size, tiles_pos)

    # Check if GPU is available
    device = getDevice()

    # Load pytorch model
    model = torch.load(params.model, map_location=device)

    # Run image inference
    x = torch.stack(tiles_img).to(device)
    y_pred = model(x)

    # Convert to heatmap
    heatmap_raw = convertToHeatMap(heatmap_size, tiles_pos, y_pred, params.threshold)
    heatmap_img = resizeImage(heatmap_raw, (target_image.shape[1], target_image.shape[0]))

    # For debugging
    print("Heatmap array:")
    print(heatmap_raw)
    print()

    display(heatmap_img, "heat map")
    display(target_image, "Input image")

    # For debugging - Show Ground truth if available
    try:
        target_mask = readMask(params.image)
        display(target_mask, "Ground truth")
    except:
        pass

    print("DONE!")


if __name__ == "__main__":
    main()
