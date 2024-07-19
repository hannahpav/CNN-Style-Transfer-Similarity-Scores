import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16
from sklearn.metrics.pairwise import cosine_similarity

import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

# Load pre-trained VGG16 model
model = vgg16(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer

# Function to preprocess images
def preprocess_image(img_path, target_size=(224, 224)):
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img).unsqueeze(0)  # Add batch dimension
    return img

# Function to extract features
def extract_features(img_path):
    img = preprocess_image(img_path)
    with torch.no_grad():
        features = model(img).squeeze().numpy()
    return features.flatten()

# Function to load images from a directory and extract features
def load_images_and_extract_features(folder):
    images = []
    features = []
    image_files = sorted(os.listdir(folder))  # Sort to maintain consistent order
    for filename in image_files:
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path)
            img.verify()  # Verify that it is an image
            images.append(filename)
            features.append(extract_features(img_path))
        except (UnidentifiedImageError, OSError):
            print(f"Skipping non-image file: {filename}")
    return images, np.array(features)

# Load images
building_folder = 'evaluation_test_images/archeticture_Images'
celebrity_folder = 'evaluation_test_images/celeb_photos'

building_images, building_features = load_images_and_extract_features(building_folder)
celebrity_images, celebrity_features = load_images_and_extract_features(celebrity_folder)

# Combine all images and features
all_images = building_images + celebrity_images
all_features = np.vstack((building_features, celebrity_features))

# Calculate cosine similarity matrix
similarity_matrix = cosine_similarity(all_features)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix, xticklabels=all_images, yticklabels=all_images, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Cosine Similarity Heatmap of VGG16 Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()
