# deploy
# server.py
from flask import Flask
app = flask(__name__)

custom_class_index = json.load(open('...'))
model = '...'
model.eval()


def transform_image(image):
  image = Image.open(io.BytesIO(image_bytes))
  transfroms = '...'
  transfromed_image = transforms(image)
  return transformed_image

def get_prediction(image_byte):
  transformed = transform_image(image_byte)
  output = model(transformed)
  _, y_hat = output.max(1)
  idx = str(y_hat.item)
  return custom_class_index[idx]

@app.route('/predict', method=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
    image_byte = file.read()
    class_id, class_name = get_prediction(image_byte)
    return jsonify({'class_id': class_id, 'class_name': class_name})

