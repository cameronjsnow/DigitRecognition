import io
import base64
import re
import pytest
from PIL import Image
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_no_image(client):
    response = client.post('/predict', json={ 'image': ''})
    data = response.get_json()
    assert data['prediction'] == 'No content detected'

def test_predict_with_mock_image(client):
    # Create a simple blank image
    img = Image.new('L', (28, 28), color=255)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode('utf-8')
    data_url = 'data:image/png;base64,' + b64_str

    response = client.post('/predict', json={ 'image': data_url })
    data = response.get_json()
    # Since it's a blank image, the prediction might not be meaningful,
    # but we can check if we got a numeric response.
    assert 'prediction' in data
    # No need for exact digit check, just ensure it returned something.
