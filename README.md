# SIFT Face Matching

Face matching system using **SIFT** and the **LFW dataset**.

### Pipeline

```text
Face Images
→ SIFT Keypoints & Descriptors
→ BFMatcher
→ Lowe's Ratio Test
→ Matching Score
→ Match / Non-Match
```

SIFT extracts scale- and rotation-invariant local features from each face. Their 128-dimensional descriptors are matched using BFMatcher, and Lowe's ratio test filters unreliable matches.

The final decision is made using a normalized matching score and a learned threshold.

### Evaluation

* Accuracy
* Precision
* Recall
* F1-score

**Tech Stack:** Python, OpenCV, NumPy, Scikit-learn
