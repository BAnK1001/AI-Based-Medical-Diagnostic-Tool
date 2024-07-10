from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

# Ensure proper encoding for output
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load dataset
dataset = load_dataset('Falah/Alzheimer_MRI', split='train')

# Explore data structure
print("Columns in dataset:", dataset.column_names)
print("Data types:", dataset.features)

# Display first 5 samples
print("\nFirst 5 samples:")
for example in dataset[:5]:
    print(example)

# Count samples in each class
label_counts = Counter(dataset['label'])
print("\nNumber of samples in each class:", label_counts)

# Mapping labels to their descriptions
label_map = {
    0: 'Non Demented',
    1: 'Very Mild Demented',
    2: 'Mild Demented',
    3: 'Moderate Demented'
}

# Preprocessing function
def preprocess_function(example):
    image = example['image']
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Convert grayscale to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    image = F.to_tensor(image)
    image = F.resize(image, (224, 224), antialias=True)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return {'image': image, 'label': example['label']}

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess_function)

# Get and display a sample image after preprocessing
sample = dataset[0]
img_tensor = sample['image']

print("Type of sample['image']:", type(img_tensor))
print("Length of sample['image']:", len(img_tensor))
print("Type of first element:", type(img_tensor[0]))

# Convert the nested list structure to a tensor
img_tensor = torch.tensor(img_tensor)

# Reshape the tensor if necessary
if img_tensor.dim() == 1:
    img_tensor = img_tensor.view(3, 224, 224)

# Now we can proceed with the permute and numpy conversion
img_array = img_tensor.permute(1, 2, 0).numpy()  # Change tensor shape from (C, H, W) to (H, W, C)
label = sample['label']

# Get label description from the label map
label_desc = label_map[label]

# Denormalize the image for display
img_array = img_array * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
img_array = np.clip(img_array, 0, 1)

plt.figure(figsize=(8, 8))
plt.imshow(img_array)
plt.axis('off')
plt.title(f"Label: {label_desc}")
plt.show()

# Print some information about the preprocessed dataset
print("\nPreprocessed dataset information:")
print("Number of samples:", len(dataset))
print("Image shape:", img_tensor.shape)
print("Label:", label_desc)