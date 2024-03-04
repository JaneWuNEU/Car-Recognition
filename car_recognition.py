from utils import load_model
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def decode_predictions(preds, top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_names[i], pred[i]) for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results

def predict(img_dir, model):
    img_files = []
    for root, dirs, files in os.walk(img_dir, topdown=False):
        for name in files:
            img_files.append(os.path.join(root, name))
    img_files = sorted(img_files)

    y_pred = []
    y_test = []

    for img_path in tqdm(img_files):
        # print(img_path)
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        preds = model.predict(x[None, :, :, :])
        decoded = decode_predictions(preds, top=1)
        pred_label = decoded[0][0][0]
        print(pred_label)
        # y_pred.append(pred_label)
        # tokens = img_path.split(os.pathsep)
        # print(tokens,pred_label)
        # class_id = int(tokens[-2])
        # # print(str(class_id))
        # y_test.append(class_id)

    return y_pred, y_test

img_width, img_height = 224, 224
num_channels = 3
num_classes = 196
class_names = range(1, (num_classes + 1))
num_samples = 1629

print("\nLoad the trained ResNet model....")
model = load_model()

y_pred, y_test = predict('./image-data/', model)