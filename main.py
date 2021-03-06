from flask import Flask, redirect, request, jsonify
# from flask_restful import Api
from fastai.learner import load_learner
from kpt_utils import ClampBatch,_resnet_split, ClampBatch, get_y, get_ip, sep_points, img2kpts
import numpy as np
import logging
import tensorflow as tf
from PIL import Image
import json

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

det_model = tf.saved_model.load('01_object_detection/model/saved_model')
kpt_model = load_learner('./02_kpt_detection/pet_kpt.pkl')
catface_model = tf.saved_model.load('03_petface/catfacenet')
dogface_model = tf.saved_model.load('03_petface/dogfacenet')

logging.debug('loaded models')


def get_OD_results(img_np, det_model, box_thresh = 0.8):
    h, w, _ = img_np.shape
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(img_np, 0), dtype=tf.uint8)

    detections = det_model(input_tensor)

    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    labels = detections['detection_classes'][0].numpy()
    boxes = boxes[scores >= box_thresh]
    boxes = (boxes * [h, w, h, w]).astype(int)
    labels = labels[scores >= box_thresh]
    return boxes, labels

def get_img_processed(img_np, boxes):
    if len(boxes) > 0:
        box = boxes[0]
        y, x = box[0], box[1],
        y2, x2 = box[2], box[3]
        im_processed = img_np[y:y2, x:x2, :]
        return im_processed, x, y
    else: return np.array([]), None, None

def get_kpts(im_processed, kpt_model):
    if len(im_processed) > 0:
        kpts = kpt_model.predict(im_processed)[0].numpy()
    else:
        kpts =  np.array([])
    return kpts

@app.route('/od', methods=['POST'])
def predict_od():
    if request.method == 'POST':
        logging.debug('Getting OD results')
        if 'image' not in request.files:
            logging.debug('No image part')
            return redirect(request.url)
        file = request.files['image']
        img = Image.open(file)
        img_np = np.array(img)
        boxes, labels = get_OD_results(img_np, det_model)
        response = jsonify({'boxes':boxes.tolist()})
        return response

@app.route('/kpt', methods=['POST'])
def predict_kpt():
    if request.method == 'POST':
        logging.debug('Getting OD results')
        if 'image' not in request.files:
            logging.debug('No image part')
            return redirect(request.url)
        file = request.files['image']
        img = Image.open(file)
        img_np = np.array(img)
        kpts = get_kpts(img_np, kpt_model)
        response = jsonify({'kpts':kpts.tolist()})
        return response

@app.route('/od_kpt', methods=['POST'])
def predict():
    if request.method == 'POST':
        logging.debug('Getting OD results')
        if 'image' not in request.files:
            logging.debug('No image part')
            return redirect(request.url)
        file = request.files['image']
        img = Image.open(file)
        img_np = np.array(img)
        boxes, labels = get_OD_results(img_np, det_model)
        logging.debug("Results succ returned")
        im_processed, x, y = get_img_processed(img_np, boxes)
        logging.debug("Image box succ created")
        kpts = get_kpts(im_processed, kpt_model)
        h_kpt, w_kpt, _ = im_processed.shape
        kpts = kpts * [w_kpt / 224, h_kpt / 224]
        kpts = kpts + [x, y]
        logging.debug('Keypoints succ generated')
        response = jsonify({'kpts':kpts.tolist(), 'boxes':boxes.tolist()})
        return response

@app.route('/petface', methods=['POST'])
def get_embedding():
    if request.method == 'POST':
        json_ = json.loads(request.get_json())
        pet_type = json_['pet_type']
        img = json_['image']
        img_np = np.array(img)
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(img_np, 0), dtype=tf.float32, name='inputs')
        fc_model = catface_model if pet_type == 'cat' else dogface_model
        emb = fc_model(input_tensor)
        response = jsonify({'emb':emb.numpy().tolist(), 'pet_type':pet_type})
        return response

if __name__ == '__main__':
    app.run(debug=True)