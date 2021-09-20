from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import io
from io import BufferedReader, BytesIO
from torchvision import models, transforms
import torch
import torch.nn as nn
from PIL import Image
import base64


app = Flask(__name__)

def io_wrap_img_2_np_img(image_io):
    img_bytes = image_io.read()
    image_buffer = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

def transform_np_img(np_img):
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    np_img = cv2.resize(np_img, (224, 224))
    tensor = torch.tensor(np_img, dtype=torch.float)    
    return tensor.permute(2, 0, 1).div(255.0).unsqueeze(0)

def get_prediction_2(image_io):
    np_img = io_wrap_img_2_np_img(image_io)
    transformed = transform_np_img(np_img)

    output = model(transformed).detach().cpu()    
    return output[0][0].item(), output[0][1].item()

def dec_2_np_img(b64enc_img_bytes):
    img_bytes = base64.b64decode(b64enc_img_bytes)
    jpg_enc_img = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(jpg_enc_img, cv2.IMREAD_COLOR)

def transform_np_img(np_img):
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    np_img = cv2.resize(np_img, (224, 224))
    tensor = torch.tensor(np_img, dtype=torch.float)    
    return tensor.permute(2, 0, 1).div(255.0).unsqueeze(0)

def get_prediction(enc_img):
    np_img = dec_2_np_img(enc_img)
    transformed = transform_np_img(np_img)

    output = model(transformed).detach().cpu()    
    return output[0][0].item(), output[0][1].item()

def get_best_model():
    # test version
    model = models.resnet18(pretrained=True)
    num_fts = model.fc.in_features
    model.fc = nn.Linear(num_fts, 2)
    model.eval()
    return model

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # enc_img = request.data
        enc_img = request.files["image_io"]
        v, a = get_prediction_2(enc_img) 
    return jsonify({'velance': v, 'arousal': a})

if __name__ == '__main__':
    model = get_best_model()
    app.run(debug=True, port=5000)
