import os
import shutil
import yaml
import random
import torch
import argparse
import numpy as np
from ultralytics import YOLO

# Function to set the random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Function to split dataset among multiple clients
def split_dataset(full_data_yaml, num_clients=3):
    with open(full_data_yaml, 'r') as f:
        full_data = yaml.safe_load(f)
    
    full_train_path = full_data['train']
    full_val_path = full_data['val']
    
    all_train_images = [f for f in os.listdir(full_train_path) if f.endswith('.jpg') or f.endswith('.png')]
    all_val_images = [f for f in os.listdir(full_val_path) if f.endswith('.jpg') or f.endswith('.png')]
    
    random.shuffle(all_train_images)
    random.shuffle(all_val_images)
    
    train_split_size = len(all_train_images) // num_clients
    val_split_size = len(all_val_images) // num_clients    
    
    for client_id in range(num_clients):
        start_train_idx = client_id * train_split_size
        end_train_idx = (client_id + 1) * train_split_size
        client_train_images = all_train_images[start_train_idx:end_train_idx]
        
        start_val_idx = client_id * val_split_size
        end_val_idx = (client_id + 1) * val_split_size
        client_val_images = all_val_images[start_val_idx:end_val_idx]
        
        client_train_dir = f'./client_{client_id + 1}/train'
        client_val_dir = f'./client_{client_id + 1}/val'
        
        client_train_images_dir = os.path.join(client_train_dir, 'images')
        client_train_labels_dir = os.path.join(client_train_dir, 'labels')
        client_val_images_dir = os.path.join(client_val_dir, 'images')
        client_val_labels_dir = os.path.join(client_val_dir, 'labels')
        
        os.makedirs(client_train_images_dir, exist_ok=True)
        os.makedirs(client_train_labels_dir, exist_ok=True)
        os.makedirs(client_val_images_dir, exist_ok=True)
        os.makedirs(client_val_labels_dir, exist_ok=True)
        
        for img_file in client_train_images:
            shutil.copy(os.path.join(full_train_path, img_file), client_train_images_dir)
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            full_train_path_lb = full_train_path[:-len('images')] + 'labels'
            if os.path.exists(os.path.join(full_train_path_lb, label_file)):
                shutil.copy(os.path.join(full_train_path_lb, label_file), client_train_labels_dir)
            else:
                print(f"Warning: Label file {label_file} for image {img_file} not found in train folder. Skipping.")
        
        for img_file in client_val_images:
            shutil.copy(os.path.join(full_val_path, img_file), client_val_images_dir)
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            full_val_path_lb = full_val_path[:-len('images')] + 'labels'
            if os.path.exists(os.path.join(full_val_path_lb, label_file)):
                shutil.copy(os.path.join(full_val_path_lb, label_file), client_val_labels_dir)
            else:
                print(f"Warning: Label file {label_file} for image {img_file} not found in val folder. Skipping.")
        
        # Save client data configuration
        client_data_yaml = {
            'train': f'./train',
            'val': f'./val',
            'nc': full_data['nc'],
            'names': full_data['names']
        }
        with open(f'./client_{client_id + 1}/data.yaml', 'w') as f:
            yaml.dump(client_data_yaml, f)

# Main function to parse arguments and call necessary functions
def main(args):
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Split the dataset across clients
    split_dataset(args.path, num_clients=args.clients)
    
    print(f"Dataset split for {args.clients} clients.")

# If this file is executed as a script, run the main function
if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Split dataset for clients and prepare YOLO training data')
    
    # Adding arguments for the script
    parser.add_argument('--path', type=str, required=True, help='Path to the dataset YAML file')
    parser.add_argument('--clients', type=int, default=3, help='Number of clients for data split')
    parser.add_argument('--seed', type=int, default=2508, help='Random seed for reproducibility')
    
    # Parse arguments and call the main function
    args = parser.parse_args()
    main(args)
