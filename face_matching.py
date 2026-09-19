import random
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

random.seed(42)
np.random.seed(42)
lfw = fetch_lfw_people(
    min_faces_per_person=20,
    resize=0.5, #for faster processing
    color=False
)

images = lfw.images
labels = lfw.target
target_names = lfw.target_names

print("Number of images:", len(images))
print("Image dimensions:", images.shape[1:])
print("Number of subjects:", len(target_names))

sift=cv2.SIFT_create()


def extract_sift_features(images,sift):
    all_descriptors=[]

    for image in images:
        image=(image*255).astype(np.uint8)

        keypoints,descriptors=sift.detectAndCompute(image, None)
        #without the mask

        if descriptors is None:
            all_descriptors.append(None)
        else:
            all_descriptors.append(descriptors)

    return all_descriptors

