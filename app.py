import torch
from flask import Flask, render_template, request
from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import os

# Load trained PyTorch model
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 4)  # Adjust for 4 classes
model.load_state_dict(torch.load('alzheimer_classification_model.pth'))
model.eval()

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

app = Flask(__name__, static_folder='static', template_folder='templates')

# Ensure the static folder exists
os.makedirs(app.static_folder, exist_ok=True)

# Mapping labels to their descriptions
label_map = {
    0: 'Mild_Demented',
    1: 'Moderate_Demented',
    2: 'Non_Demented',
    3: 'Very_Mild_Demented'
}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # Ensure RGB format
        
        # Preprocess the image
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.softmax(output, dim=1)
            
            _, predicted = torch.max(output, 1)
            prediction_label = predicted.item()
            prediction_desc = label_map[prediction_label]
            confidence = probabilities[0][prediction_label].item() * 100
        
        # Save the image to a temporary file for display
        img_np = np.array(img)
        plt.imsave(os.path.join(app.static_folder, 'uploaded_image.png'), img_np)
        
        return render_template(
            'result.html', 
            prediction=prediction_desc,
            confidence=f"{confidence:.2f}%",
            image_path='uploaded_image.png'
        )
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)