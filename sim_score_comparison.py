import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load pre-trained VGG16 model
model = vgg16(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layers
model.eval()  # Set to evaluation mode

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

# Paths to images
content_image_path = 'gatys_original_images/Tuebingen_Neckarfront.jpg'
style_image_path = 'gatys_original_images/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'
generated_images_folder = 'image_gens/gatys_recreate_3'  # Folder containing generated images

# Extract features for content and style images
content_features = extract_features(content_image_path)
style_features = extract_features(style_image_path)

# Extract features for generated images at each step
generated_image_files = sorted(os.listdir(generated_images_folder))
generated_image_features = [extract_features(os.path.join(generated_images_folder, f)) for f in generated_image_files]

# Calculate cosine similarities
content_similarities = [cosine_similarity([content_features], [gen_feat])[0][0] for gen_feat in generated_image_features]
style_similarities = [cosine_similarity([style_features], [gen_feat])[0][0] for gen_feat in generated_image_features]

# Steps
steps = [i * 5 for i in range(len(generated_image_files))]

# Plotting
fig, ax1 = plt.subplots()

# Set the same scale for both y-axes
min_similarity = min(min(content_similarities), min(style_similarities))
max_similarity = max(max(content_similarities), max(style_similarities))

color = 'tab:blue'
ax1.set_xlabel('Steps')
ax1.set_ylabel('Content Similarity', color=color)
ax1.scatter(steps, content_similarities, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()  # Instantiate a second y-axis that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Style Similarity', color=color)
ax2.scatter(steps, style_similarities, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1)

fig.tight_layout()  # To ensure the labels do not overlap
plt.title('Cosine Similarity of Generated Images Over Training Steps')
plt.show()
