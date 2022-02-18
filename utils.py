import os

import numpy as np
from PIL import Image
from skimage import img_as_float
from utils_face import fix_eyes, get_face_aligned
import logging

pet_parser = {
    1: 'cat',
    2: 'dog'
}

secret_key = os.environ["SECRET_KEY"]

def authorizeToken(token, secret_key=secret_key):
    if token == secret_key:
        return True
    else:
        return False

def parse_labels(labels, parser=pet_parser):
    return [parser[l] for l in labels]


def get_OD_results(img_np, det_model, box_thresh=0.8):
    logger = logging.getLogger(__name__ + ":OD")
    h, w, _ = img_np.shape
    input = img_np[np.newaxis, ...]

    detections = det_model.run(["detection_boxes", "detection_scores", "detection_classes"],
                               {'input_tensor': input.astype(np.uint8)})
    logger.debug("Got pet detections")
    boxes = detections[0][0]
    scores = detections[1][0]
    labels = detections[2][0]

    boxes = boxes[scores >= box_thresh]
    boxes = (boxes * [h, w, h, w]).astype(int)
    labels = labels[scores >= box_thresh]
    logger.debug("returning filtered pet detections")
    return boxes, labels


def preprocess4kpts(img_pil, boxes, size=(224, 224)):
    logger = logging.getLogger(__name__ + ":preprocessingKPTS")
    if len(boxes) > 0:
        logger.debug("found pets!")
        pets = list()
        for box in boxes:
            top, left, bottom, right = box
            pet_i = img_pil.crop((left, top, right, bottom))
            pet_i = pet_i.resize(size, resample=Image.BILINEAR)
            pet_i = np.array(pet_i)
            pet_i = pet_i.transpose(2, 0, 1)
            pet_i = pet_i[np.newaxis, ...]
            pets.append({'pet': pet_i, 'top': top, 'left': left, 'h': bottom-top,'w': right-left})
        logger.debug("processed all pets")
        return pets
    else:
        logger.debug("no pets found")
        return list()


def get_kpts(img_np, kpt_model, shape=(1, 3, 224, 224)):
    logger = logging.getLogger(__name__ + ":getKPTS")
    if img_np.shape == shape:
        logger.debug("correct shape")
        if img_np.dtype == np.uint8:
            model_input = img_as_float(img_np)
        else:
            model_input = img_np
        kpts = kpt_model.run(None, {'input': model_input.astype(np.float32)})[0][0]
        # normalization of kpts
        kpts = (kpts + 1) * (224 / 2)
        kpts = np.array([[kpts[i-1], kpts[i]] for i in range(1, 6, 2)])
        logger.debug("got kpts")
    else:
        logger.debug("incorrect shape, empty response")
        kpts = np.array([[]])
    return kpts

def postprocess4kpts(pets, kpts, size=(224, 224)):
    logger = logging.getLogger(__name__ + ":postprocessingKPTS")
    if len(pets) > 0:
        logger.debug("found pets!")
        new_kpts = list()
        for i, pet in enumerate(pets):
            top, left, h, w = pet['top'], pet['left'], pet['h'], pet['w']
            kpt_i = kpts[i] * [w / size[0], h / size[1]]
            kpt_i = kpt_i + [left, top]
            new_kpts.append(kpt_i)

        logger.debug("processed all pets")
        return new_kpts
    else:
        logger.debug("no pets found")
        return kpts

def preprocess4embedding(img_np, labels, pet_type, kpts, size=(224,224,3)):
    faces, labelss = list(), list()
    for i, kpt in enumerate(kpts):
        leye, reye, nose = fix_eyes(kpt[0], kpt[1], kpt[2], num_eyes=2)
        face_allgnd = get_face_aligned(img_np, leye, reye, nose, labels[i], SIZE=size)
        if labels[i] in pet_type:
            faces.append(face_allgnd)
            labelss.append(labels[i])
    return np.array(faces), labelss


def get_embedding(img_np, emb_model, shape=(1, 224, 224, 3)):
    logger = logging.getLogger(__name__ + ":getEmb")
    if img_np.shape == shape:
        logger.debug("correct shape")
        return emb_model.run(None, {'input_1': img_np.astype(np.float32)})[0]
    else:
        logger.debug("incorrect shape, empty response: " + str(img_np.shape))
        return np.empty((1,32))