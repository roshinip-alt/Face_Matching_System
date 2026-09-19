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
    resize=0.5,
    color=False
)

images = lfw.images
labels = lfw.target
target_names = lfw.target_names

print("Number of images:", len(images))
print("Image dimensions:", images.shape[1:])
print("Number of subjects:", len(target_names))

sift = cv2.SIFT_create()


def extract_sift_features(images, sift):
    all_descriptors = []

    for image in images:
        image = (image * 255).astype(np.uint8)

        keypoints, descriptors = sift.detectAndCompute(image, None)

        if descriptors is None:
            all_descriptors.append(None)
        else:
            all_descriptors.append(descriptors)

    return all_descriptors


descriptors = extract_sift_features(images, sift)

print("SIFT feature extraction completed.")


person_to_images = {}

for i, label in enumerate(labels):
    if label not in person_to_images:
        person_to_images[label] = []

    person_to_images[label].append(i)


NUM_GENUINE_PAIRS = 250

genuine_pairs = []

while len(genuine_pairs) < NUM_GENUINE_PAIRS:

    label = random.choice(list(person_to_images.keys()))
    indices = person_to_images[label]

    if len(indices) < 2:
        continue

    idx1, idx2 = random.sample(indices, 2)

    if descriptors[idx1] is None or descriptors[idx2] is None:
        continue

    pair = tuple(sorted((idx1, idx2)))

    if pair not in genuine_pairs:
        genuine_pairs.append(pair)


print("Number of genuine pairs:", len(genuine_pairs))


NUM_IMPOSTOR_PAIRS = 250

impostor_pairs = []

all_indices = list(range(len(images)))

while len(impostor_pairs) < NUM_IMPOSTOR_PAIRS:

    idx1, idx2 = random.sample(all_indices, 2)

    if labels[idx1] != labels[idx2]:

        if descriptors[idx1] is not None and descriptors[idx2] is not None:

            pair = tuple(sorted((idx1, idx2)))

            if pair not in impostor_pairs:
                impostor_pairs.append(pair)


print("Number of impostor pairs:", len(impostor_pairs))


def calculate_sift_match_score(desc1, desc2):

    if desc1 is None or desc2 is None:
        return 0.0

    if len(desc1) < 2 or len(desc2) < 2:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    matches = matcher.knnMatch(
        desc1,
        desc2,
        k=2
    )

    good_matches = []

    for m_n in matches:

        if len(m_n) < 2:
            continue

        m, n = m_n

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return len(good_matches) / min(
        len(desc1),
        len(desc2)
    )


scores = []
true_labels = []

for idx1, idx2 in genuine_pairs:

    score = calculate_sift_match_score(
        descriptors[idx1],
        descriptors[idx2]
    )

    scores.append(score)
    true_labels.append(1)


for idx1, idx2 in impostor_pairs:

    score = calculate_sift_match_score(
        descriptors[idx1],
        descriptors[idx2]
    )

    scores.append(score)
    true_labels.append(0)


thresholds = sorted(set(scores))

best_threshold = 0
best_accuracy = 0

for threshold in thresholds:

    predictions = []

    for score in scores:

        if score >= threshold:
            predictions.append(1)
        else:
            predictions.append(0)

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold


predictions = []

for score in scores:

    if score >= best_threshold:
        predictions.append(1)
    else:
        predictions.append(0)


accuracy = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    zero_division=0
)

recall = recall_score(
    true_labels,
    predictions,
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    zero_division=0
)


idx1, idx2 = genuine_pairs[0]

genuine_score = calculate_sift_match_score(
    descriptors[idx1],
    descriptors[idx2]
)

print("\nExample Genuine Pair:")
print("Person:", target_names[labels[idx1]])
print("Score:", genuine_score)


idx1, idx2 = impostor_pairs[0]

impostor_score = calculate_sift_match_score(
    descriptors[idx1],
    descriptors[idx2]
)

print("\nExample Impostor Pair:")
print("Person 1:", target_names[labels[idx1]])
print("Person 2:", target_names[labels[idx2]])
print("Score:", impostor_score)


print("\nEvaluation Results:")
print("Number of pairs:", len(scores))
print("Best threshold:", best_threshold)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)