from flask import Flask, render_template, request, jsonify
import re
import base64
import io
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import os
import datetime
from sqlalchemy import create_engine, Column, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)

# ---------------------
# Database Setup
# ---------------------
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    predicted_digit = Column(Integer)
    timestamp = Column(DateTime)

db_url = "sqlite:///predictions.db"
engine = create_engine(db_url)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ---------------------
# Model Definition
# ---------------------
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(9216, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = DigitClassifier()
model.load_state_dict(torch.load('model/digit_classifier.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------------------
# Routes
# ---------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stats')
def stats():
    # Display total number of predictions made
    session = SessionLocal()
    count = session.query(PredictionLog).count()
    session.close()
    return render_template('stats.html', count=count)

@app.route('/api/docs')
def api_docs():
    # Serve OpenAPI spec (YAML or JSON)
    # For simplicity, we'll just return a basic JSON spec inline
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Digit Recognition API",
            "version": "1.0.0"
        },
        "paths": {
            "/predict": {
                "post": {
                    "summary": "Predict a digit",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "image": {"type": "string"}
                                    },
                                    "required": ["image"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "prediction": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return jsonify(spec)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = re.sub('^data:image/.+;base64,', '', data['image'])
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    else:
        return jsonify({'prediction': 'No content detected'})

    # Preprocessing
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img = img.resize((28, 28), Image.LANCZOS)

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    # Save preprocessed image for debugging
    img.save('static/preprocessed_image.png')

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1, keepdim=True)
        digit = pred.item()

    # Log prediction to DB
    session = SessionLocal()
    log_entry = PredictionLog(predicted_digit=digit, timestamp=datetime.datetime.utcnow())
    session.add(log_entry)
    session.commit()
    session.close()

    return jsonify({'prediction': str(digit)})

# Production-ready server entrypoint
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
