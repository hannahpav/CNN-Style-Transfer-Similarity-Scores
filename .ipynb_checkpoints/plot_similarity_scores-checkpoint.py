import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Directory containing the CSV files
csv_directory = 'final_project_similarity_scores'  # Replace with the actual path

# List to store the DataFrame for each CSV file
dataframes = []

# Load each CSV file
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_directory, filename)
        parts = filename.split('_')
        if len(parts) >= 4:
            model_type, content_image, style_image, _ = parts[:4]
            df = pd.read_csv(file_path)
            df['File'] = filename  # Add a column to keep track of the source file
            df['Content Image'] = content_image
            df['Style Image'] = style_image
            dataframes.append(df)

# Combine all dataframes into a single dataframe
combined_df = pd.concat(dataframes)

# Calculate the ratio of style similarity to content similarity
combined_df['Style/Content Ratio'] = combined_df['Style Similarity'] / combined_df['Content Similarity']

# Define colors and line styles
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
line_styles = ['-', '--', '-.', ':']
color_map = {content: colors[i % len(colors)] for i, content in enumerate(combined_df['Content Image'].unique())}
line_style_map = {style: line_styles[i % len(line_styles)] for i, style in enumerate(combined_df['Style Image'].unique())}

# Plotting
plt.figure(figsize=(12, 8))

# Plot the ratios for each file
for (filename, group) in combined_df.groupby('File'):
    content_image = group['Content Image'].iloc[0]
    style_image = group['Style Image'].iloc[0]
    color = color_map[content_image]
    line_style = line_style_map[style_image]
    plt.plot(group['Step'], group['Style/Content Ratio'], label=filename, color=color, linestyle=line_style)

# Add a horizontal line at 1
plt.axhline(y=1, color='grey', linestyle='--', linewidth=1, label='Ratio = 1')

plt.xlabel('Steps')
plt.ylabel('Style/Content Ratio')
plt.title('Style/Content Ratio Over Training Steps')

# Create custom legend
handles = []
for content_image in combined_df['Content Image'].unique():
    handles.append(mlines.Line2D([], [], color=color_map[content_image], label=content_image))
for style_image in combined_df['Style Image'].unique():
    handles.append(mlines.Line2D([], [], color='black', linestyle=line_style_map[style_image], label=style_image))
handles.append(mlines.Line2D([], [], color='grey', linestyle='--', linewidth=1, label='Ratio = 1'))
plt.legend(handles=handles)

plt.tight_layout()
plt.show()
