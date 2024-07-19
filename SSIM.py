import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim

# Function to load and resize images from a directory
def load_images_from_folder(folder, size=(256, 256)):
    images = []
    image_files = sorted(os.listdir(folder))  # Sort to maintain consistent order
    for filename in image_files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized_img = cv2.resize(img, size)
            images.append((filename, resized_img))
    return images

# Desired image size
image_size = (256, 256)  # You can change this to the desired size

# Load images
building_folder = 'evaluation_test_images/archeticture_Images'
celebrity_folder = 'evaluation_test_images/celeb_photos'

building_images = load_images_from_folder(building_folder, size=image_size)
celebrity_images = load_images_from_folder(celebrity_folder, size=image_size)

# Combine all images
all_images = building_images + celebrity_images

# Calculate SSIM index for each pair of images
num_images = len(all_images)
ssim_matrix = np.zeros((num_images, num_images))

for i in range(num_images):
    for j in range(num_images):
        _, img1 = all_images[i]
        _, img2 = all_images[j]
        ssim_value = ssim(img1, img2)
        ssim_matrix[i, j] = ssim_value

# Get image filenames for labeling
image_labels = [name for name, _ in all_images]

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(ssim_matrix, xticklabels=image_labels, yticklabels=image_labels, cmap='coolwarm', annot=True, fmt=".2f")
plt.title('SSIM Heatmap of Images')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()
