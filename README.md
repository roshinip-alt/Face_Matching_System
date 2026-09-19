# SIFT Face Matching

Face matching using **SIFT** and the **LFW dataset**.

### Pipeline

Face Images
→ SIFT Keypoints & Descriptors
→ BFMatcher
→ Lowe's Ratio Test
→ Matching Score
→ Threshold
→ Match / Non-Match

SIFT extracts scale- and rotation-invariant local features from each face. Their 128-dimensional descriptors are matched using BFMatcher, and Lowe's ratio test filters unreliable matches.

The final decision is made using a normalized matching score and a learned threshold.

### Evaluation

Tested on **500 face pairs**:

* 250 genuine pairs
* 250 impostor pairs

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 61.2% |
| Precision | 68.4% |
| Recall    | 41.6% |
| F1 Score  | 51.7% |

Best threshold: **0.05**

**Tech:** Python, OpenCV, NumPy, Scikit-learn
