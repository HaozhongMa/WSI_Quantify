import openslide
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
# from vit_pytorch import ViT
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image, ImageDraw
import math
import cv2
from tqdm import tqdm
import argparse
from collections import Counter
from collections import OrderedDict

def is_white(image, threshold=0.95):
    data = np.array(image)
    white_pixels = np.sum(np.all(data >= [230, 230, 230], axis=-1))
    total_pixels = data.shape[0] * data.shape[1]
    return (white_pixels / total_pixels) > threshold

def split_to_label(svs_file_path, tile_size, model_path=r"/data2/mahaozhong/code/Model/5class_best_resnet50_model.pth", gpu_id=0, threshold=0.95):
    slide = openslide.OpenSlide(svs_file_path)
    w, h = slide.level_dimensions[0]
    step = tile_size // 2
    # cols, rows = math.ceil(w / step), math.ceil(h / step)
    
    # 数据预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])


    # 加载训练好的ResNet50模型
    loaded_model = models.resnet50(weights=None)  # Create an instance of ResNet50
    num_ftrs = loaded_model.fc.in_features
    loaded_model.fc = nn.Linear(num_ftrs, 5)
    state_dict = torch.load(model_path, map_location=torch.device(f'cuda:{gpu_id}'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  
        new_state_dict[name] = v

    loaded_model.load_state_dict(new_state_dict)
    # GPU availability
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    loaded_model = loaded_model.to(device)

    loaded_model.eval()  # Set the model to evaluation mode

    filename_label_dict = {}

    for row in tqdm(range(0, h - tile_size + 1, step), desc="Processing rows"):
        for col in tqdm(range(0, w - tile_size + 1, step), desc="Processing columns", leave=False):
            image = slide.read_region((col, row), 0, (tile_size, tile_size)).convert('RGB')

            if image.size != (tile_size, tile_size):
                padded_image = Image.new('RGB', (tile_size, tile_size), (255, 255, 255))
                padded_image.paste(image, (0, 0))
                image = padded_image

            if not is_white(image, threshold=threshold):
                # 预处理图像
                input_tensor = preprocess(image)
                input_batch = input_tensor.unsqueeze(0)
                input_batch = input_batch.to(device)

                # 进行预测
                with torch.no_grad():
                    output = loaded_model(input_batch)
                    label = output.argmax(1).item()
                    filename_label_dict[f"{col}_{row}.jpg"] = label


                    
    return(filename_label_dict)
def get_common_label(col, row, filename_label_dict, tile_step):
    labels = []
    
    neighbor_key1 = f"{col}_{row}.jpg"
    if neighbor_key1 in filename_label_dict:
        labels.append(filename_label_dict[neighbor_key1])   

    neighbor_key2 = f"{col - tile_step}_{row}.jpg"
    if neighbor_key2 in filename_label_dict:
        labels.append(filename_label_dict[neighbor_key2])    

    neighbor_key3 = f"{col - tile_step}_{row - tile_step}.jpg"
    if neighbor_key3 in filename_label_dict:
        labels.append(filename_label_dict[neighbor_key3])       

    neighbor_key4 = f"{col - tile_step}_{row - tile_step}.jpg"
    if neighbor_key4 in filename_label_dict:
        labels.append(filename_label_dict[neighbor_key4])      


    if labels:
        return Counter(labels).most_common(1)[0][0]
    return None
def create_colored_wsi(filename_label_dict, original_dimensions, tile_size, scale_percent=10):
    wsi_image = Image.new('RGB', original_dimensions, (255, 255, 255))
    draw = ImageDraw.Draw(wsi_image)
    label_colors = {
        0: (255, 255, 0),   
        1: (255, 255, 255), 
        2: (0, 0, 255),    
        3: (0, 255, 0),     
        4: (255, 0, 0)      
    }
    
    tile_step = tile_size // 2  

    for key in tqdm(filename_label_dict.keys(), desc="Drawing tiles"):
        col, row = map(int, key.replace('.jpg', '').split('_'))
        

        common_label = filename_label_dict.get(key, 1)
        if common_label is not None:
            color = label_colors.get(common_label, (255, 255, 255))
            draw.rectangle([col, row, col + tile_size//2, row + tile_size//2], fill=color)

    wsi_image = wsi_image.resize((original_dimensions[0] // scale_percent, original_dimensions[1] // scale_percent))
    return(wsi_image)

def extract_color_areas(image, lower_bound, upper_bound):
    mask = cv2.inRange(image, lower_bound, upper_bound)
    return cv2.bitwise_and(image, image, mask=mask)

def remove_small_areas(image, min_size):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)


    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        if cv2.contourArea(contour) < min_size:
            cv2.drawContours(image, [contour], -1, (0, 0, 0), -1)

    return image
def change_color(image, target_color, new_color):

    mask = cv2.inRange(image, np.array(target_color), np.array(target_color))

    image[np.where(mask != 0)] = new_color

    return image

def count_values(dictionary):

    value_counts = {}
    

    for value in dictionary.values():

        if value not in value_counts:
            value_counts[value] = 0

        value_counts[value] += 1
    

    return value_counts



parser = argparse.ArgumentParser(description='')
parser.add_argument('--WSI', type=str, required=True, help='Path to the WSI')
args = parser.parse_args()
pic_path = args.WSI

slide = openslide.OpenSlide(pic_path)
w, h = slide.level_dimensions[0]
original_dimensions = (w,h)
tile_size = 224
filename_label_dict = split_to_label(pic_path,224)
value_count = count_values(filename_label_dict)
ADI_count = value_count[0]
BAC_count = value_count[1]
LYM_count = value_count[2]
STR_count = value_count[3]
TUM_count = value_count[4]
wsi_image = create_colored_wsi(filename_label_dict, original_dimensions, tile_size)
save_path = r"test.jpg"
    
save_path_out = save_path.split(".jp")[0]+f"_{ADI_count}_{BAC_count}_{LYM_count}_{STR_count}_{TUM_count}.jpg"
wsi_image.save(save_path_out)
        