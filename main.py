import io
from flask import Flask, request, jsonify
import numpy as np
import logging
from PIL import Image
import onnxruntime as ort
from utils import get_OD_results, parse_labels, get_kpts, preprocess4kpts, postprocess4kpts, get_embedding, preprocess4embedding, authorizeToken
import json
import base64

format = "%(levelname)s: %(name)s: %(asctime)s.%(msecs)03d: %(message)s"
logging.basicConfig(level=logging.DEBUG, format=format, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
app = Flask(__name__)
logger.info("loading models...")
det_model = ort.InferenceSession("00_models/00_pet_detection.onnx")
kpt_model = ort.InferenceSession("00_models/01_kpts_detection.onnx")
catface_model = ort.InferenceSession("00_models/02_catfacenet.onnx")
dogface_model = ort.InferenceSession("00_models/02_dogfacenet.onnx")
logging.debug('loaded models')

def predict_od_kpt_helper(img_np, img_pil):
    boxes, labels = get_OD_results(img_np, det_model)
    labels = parse_labels(labels)
    logger.debug("OD Results successfully returned")
    pets = preprocess4kpts(img_pil, boxes)
    logger.debug("number of pets: " + str(len(pets)))
    # logger.debug("processing only first pet")
    logger.debug("processing all pets")
    kpts_all = list()
    for pet in pets:
        kpt_i = get_kpts(pet['pet'], kpt_model)
        kpts_all.append(kpt_i)
    logger.debug('Keypoints successfully generated')
    kpts_all = np.array(postprocess4kpts(pets, kpts_all))
    logger.info('Keypoints successfully postprocesed')
    return boxes, labels, kpts_all

@app.route('/od', methods=['POST'])
def predict_od():
    auth_header = request.headers.get("Authorization", " ")
    access_token = auth_header.split(" ")[1]
    if authorizeToken(access_token):
        logger.debug('Getting OD results')
        if 'image' not in request.files:
            logging.debug('No image part')
            return "please provide image", 400
        file = request.files['image']
        img = Image.open(file)
        img_np = np.array(img)
        img.close()
        boxes, labels = get_OD_results(img_np, det_model)
        labels = parse_labels(labels)
        return jsonify({'boxes':boxes.tolist(), "labels":labels})
    else:
        return jsonify({"message":"Authentication failed"}), 401

@app.route('/kpt', methods=['POST'])
def predict_kpt():
    auth_header = request.headers.get("Authorization", " ")
    access_token = auth_header.split(" ")[1]
    if authorizeToken(access_token):
        logger.info('Getting Kpts results')
        if 'image' not in request.files:
            logging.debug('No image part')
            return "please provide image", 400
        file = request.files['image']
        img = Image.open(file)
        img_np = np.array(img).transpose(2, 0, 1)
        img_np = img_np[np.newaxis, ...]
        img.close()
        kpts = get_kpts(img_np, kpt_model)
        response = jsonify({'kpts': kpts.tolist()})
        return response
    else:
        return jsonify({"message":"Authentication failed"}), 401

@app.route('/od_kpt', methods=['POST'])
def predict_od_kpt():
    auth_header = request.headers.get("Authorization", " ")
    access_token = auth_header.split(" ")[1]
    if authorizeToken(access_token):
        logger.info('Getting OD and KPTS results')
        if 'image' not in request.files:
            logging.debug('No image part')
            return "please provide image", 400
        file = request.files['image']
        img_pil = Image.open(file)
        img_np = np.array(img_pil)
        boxes, labels, kpts = predict_od_kpt_helper(img_np, img_pil)
        response = jsonify({'kpts': kpts.tolist(), 'boxes': boxes.tolist(), "labels": labels})
        return response
    else:
        return jsonify({"message":"Authentication failed"}), 401

@app.route('/emb', methods=['POST'])
def get_embeddings():
    auth_header = request.headers.get("Authorization", " ")
    access_token = auth_header.split(" ")[1]
    if authorizeToken(access_token):
        logger.info('Getting emb results')
        json_ = json.loads(request.data)
        pet_type = json_.get('pet_type', "")
        if pet_type == '' or pet_type not in ['cat', 'dog']:
            logging.debug('No pet type in request, allowed pet types are "cat" and "dog"')
            return 'No pet type in request, allowed pet types are "cat" and "dog"', 400
        img_list = json_.get('image', list())
        if len(img_list) == 0:
            logging.debug('No image part')
            return "please provide image", 400
        logger.debug("got image and pet type")
        img_np = np.array(img_list)
        img_np = img_np[np.newaxis, ...]
        emb_model = catface_model if pet_type == 'cat' else dogface_model
        emb = get_embedding(img_np, emb_model)
        logger.info("got embeddings")
        response = jsonify({'emb': emb.tolist(), 'pet_type': pet_type})
        return response
    else:
        return jsonify({"message":"Authentication failed"}), 401

@app.route('/pet_face', methods=['POST'])
def process_pet_face():
    auth_header = request.headers.get("Authorization", " ")
    access_token = auth_header.split(" ")[1]
    if authorizeToken(access_token):
        logger.info('Processing image results')
        json_ = json.loads(request.data)
        pet_type = json_.get('pet_type', [])
        if len(pet_type) == 0:
            logging.debug('No pet type in request')
            return 'No pet type in request', 400
        if any([x not in ['cat','dog'] for x in pet_type]):
            logging.debug('allowed pet types are "cat" and "dog"')
            logging.debug(pet_type)
            return 'allowed pet types are "cat" and "dog"', 400
        file = json_.get('image', "")
        is_base64 = json_.get("is_base64", False)
        if file == "" or not is_base64:
            logging.debug('No image part')
            return "please provide image", 400
        img_b64 = base64.b64decode(file)
        img_pil = Image.open(io.BytesIO(img_b64))
        img_np = np.array(img_pil)
        logger.debug("got image and pet type")
        boxes, labels, kpts = predict_od_kpt_helper(img_np, img_pil)
        logger.info("calculated boxes and kpts")
        faces, labels = preprocess4embedding(img_np, labels, pet_type, kpts)
        logger.debug("preprocess and align faces")
        embs = np.empty((0,1,32))
        for i, label in enumerate(labels):
            if label == 'cat':
                emb_model = catface_model
                logger.debug("using cat model")
            else:
                emb_model = dogface_model
                logger.debug("using cat model")
            logger.debug(faces[i][0][0])
            emb = get_embedding(faces[i][np.newaxis, ...], emb_model)
            embs = np.append(embs, [emb], axis=0)
        logger.info("embeddings calculated")
        response = jsonify({"embs": embs.tolist(), "pet_types": labels, 'kpts': kpts.tolist(), 'boxes': boxes.tolist()})
        return response
    else:
        return jsonify({"message":"Authentication failed"}), 401

if __name__ == '__main__':
    app.run(debug=False)