import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd


# Paths to images
content_image_path = 'final_project_images/content_images/miley_cyrus.png'
style_image_path = 'final_project_images/style_images/cathedral.png'
generated_images_folder = 'image_gens/miley_cyrus_arch/cathedral_3'  # Folder containing generated images

save_data_name = 'miley_cathedral_similarity_scores_3.csv'

step_size = 100

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
    try:
        img = preprocess_image(img_path)
        with torch.no_grad():
            features = model(img).squeeze().numpy()
        return features.flatten()
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file {img_path}")
        return None
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


# Extract features for content and style images
content_features = extract_features(content_image_path)
style_features = extract_features(style_image_path)

# Check if the content and style features are successfully extracted
if content_features is None or style_features is None:
    raise ValueError("Error: Content or style image could not be processed.")

# Extract features for generated images at each step
generated_image_files = sorted(os.listdir(generated_images_folder))

# Filter only image files
image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
generated_image_files = [f for f in generated_image_files if os.path.splitext(f)[1].lower() in image_extensions]

generated_image_features = []
filtered_image_files = []

for f in generated_image_files:
    img_path = os.path.join(generated_images_folder, f)
    features = extract_features(img_path)
    if features is not None:  # Only add if the features are successfully extracted
        generated_image_features.append(features)
        filtered_image_files.append(f)

# Calculate cosine similarities
content_similarities = [cosine_similarity([content_features], [gen_feat])[0][0] for gen_feat in generated_image_features]
style_similarities = [cosine_similarity([style_features], [gen_feat])[0][0] for gen_feat in generated_image_features]

# Steps
steps = [i * step_size for i in range(len(generated_image_files))]

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
full_path = os.path.join('final_similarity_scores_HP/', save_data_name + '.png')
plt.savefig(full_path)
plt.show()

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Step': steps,
    'Content Similarity': content_similarities,
    'Style Similarity': style_similarities
})

# Save the DataFrame to a CSV file
results_df.to_csv('final_similarity_scores_HP/' + save_data_name, index=False)