from flask import Flask, render_template, request, jsonify
import re
import base64
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import io
import torch
import torch.nn.functional as F
import torch.nn as nn  # Import nn module
import torchvision.transforms as transforms

app = Flask(__name__)

# Define the model architecture
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)        # BatchNorm after conv1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)        # BatchNorm after conv2
        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)
        self.bn3 = nn.BatchNorm1d(128)       # BatchNorm after fc1
        self.fc2 = nn.Linear(128, 10)
        # Dropout layer
        self.dropout = nn.Dropout(0.1)       # Reduced dropout rate

    def forward(self, x):
        # Convolutional layers with ReLU, BatchNorm, and max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        # Flatten the tensor
        x = torch.flatten(x, 1)
        # Fully connected layers with ReLU and BatchNorm
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        # Output layer
        x = self.fc2(x)  # No activation function here
        return x

# Instantiate the model
model = DigitClassifier()
# Load the trained weights
model.load_state_dict(torch.load('model/digit_classifier.pth', map_location=torch.device('cpu')))
# Set the model to evaluation mode
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    data = request.get_json()
    img_data = re.sub('^data:image/.+;base64,', '', data['image'])
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # Convert to grayscale

    # Optional: Crop the image to the bounding box of the digit
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    else:
        return jsonify({'prediction': 'No content detected'})

    # Resize to an intermediate size
    #img = img.resize((100, 100), resample=Image.LANCZOS)

    # Apply Gaussian Blur
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # Resize down to 28x28 pixels
    img = img.resize((28, 28), resample=Image.LANCZOS)

    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Apply the transform
    img_tensor = transform(img)
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    # Save the preprocessed image
    img.save('static/preprocessed_image.png')

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1, keepdim=True)
        digit = pred.item()

    return jsonify({'prediction': digit})

if __name__ == '__main__':
    app.run(debug=True)
