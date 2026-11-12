# docker cp server:/app/best_model.pth ./best_model.pth
# MODEL_PATH=./best_model.pth python Communicator/intrusion_prediction.py

import torch.nn.functional as F
from typing import Tuple
import torch.nn as nn
import numpy as np
import torch
import os

MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")

class Net(nn.Module):
    def __init__(self, input_dim=4, num_classes=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)
    
model = Net()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# --- Normalization stats (must match training) ---
MEAN = np.array([27.041304, 61.775265, 969.56335,  3.2226298], dtype=np.float32)
STD  = np.array([8.085047,  24.433617, 105.70269,  2.5380642], dtype=np.float32)

torch.manual_seed(0)
np.random.seed(0)

def normalize(x: np.ndarray) -> np.ndarray:
    return (x - MEAN) / (STD + 1e-8)

# --- Predict intrusion probability with test‑time augmentation (TTA) ---
def predict_proba(x: np.ndarray, tta: int = 8, noise_scale: float = 0.01) -> float:
# Runs the model 16 times on the same input, each time adding tiny random noise
# Returns the average probability + the ± std deviation
# Real sensor readings always have small measurement noise anyway, so this simulates that uncertainty

    model.eval()
    probs = []
    with torch.no_grad():
        for _ in range(tta):
            # Add small gaussian noise in normalized space to emulate TTA
            aug = x + np.random.normal(scale=noise_scale, size=x.shape).astype(np.float32)
            inp = torch.tensor(aug).unsqueeze(0)  # shape (1, features)
            logits = model(inp)
            p = torch.softmax(logits, dim=1).cpu().numpy()[0, 1]
            probs.append(float(p))
    return float(np.mean(probs)), float(np.std(probs))

def feature_anomaly_score(x_raw: np.ndarray) -> float:
# Measures how far a sensor reading is from what is considered normal, using only the mean and standard deviation of the training data.

    z = (x_raw - MEAN) / (STD + 1e-8)   # Computes the z-score of each feature
    mahal = float(np.sum(z * z))  # Measures how far a sample is from the known normal distribution using z-scores
    # Convert to probabilistic-like score via scaled exponential
    scale = x_raw.size * 2.0
    score = 1.0 - np.exp(-mahal / scale)    # Converts that distance into a number between 0 and 1 using an exponential decay transformation
    return float(np.clip(score, 0.0, 1.0))

def combined_score(classifier_prob: float, feature_score: float, alpha: float = 0.7) -> float:
# This blends the two independent opinions together
# With alpha=0.75 the final score is 75% the neural network's probability and 25% the statistical anomaly score

    return alpha * classifier_prob + (1.0 - alpha) * feature_score

def detect_from_score(score: float, threshold: float = 0.5):
    return "INTRUSION 🚨" if score >= threshold else "NORMAL ✅"

# Optional utility: compute best threshold from labeled arrays (requires scikit-learn)
def optimal_threshold_from_labels(y_true: np.ndarray, scores: np.ndarray):
    try:
        from sklearn.metrics import roc_curve
    except Exception:
        raise RuntimeError("sklearn required for threshold calibration: pip install scikit-learn")
    fpr, tpr, thr = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    idx = np.argmax(j_scores)
    return float(thr[idx])

def random_sensor_sample(anomaly_prob: float = 0.1) -> Tuple[np.ndarray, int]:
    is_anomaly = np.random.rand() < anomaly_prob

    if is_anomaly:
        x_raw = np.array([
            np.random.uniform(40.0,  50.0),   # temperature
            np.random.uniform(90.0, 150.0),   # humidity
            np.random.uniform(600.0, 800.0),  # pressure
            np.random.uniform(7.0,  12.0),    # vibration
        ], dtype=np.float32)
        return x_raw, 1
    else:
        x_raw = np.array([
            np.random.uniform(15.0, 45.0),    # temperature
            np.random.uniform(20.0, 80.0),    # humidity
            np.random.uniform(800.0, 1100.0), # pressure
            np.random.uniform(0.0, 7.0),      # vibration
        ], dtype=np.float32)
        return x_raw, 0

normal_sample    = np.array([25.0,  55.0, 1000.0, 2.5], dtype=np.float32)
intrusion_sample = np.array([45.0, 120.0,  700.0, 9.5], dtype=np.float32)

normal_norm = normalize(normal_sample)  # Normalize
intrusion_norm = normalize(intrusion_sample)

normal_prob, normal_std = predict_proba(normal_norm, tta=16, noise_scale=0.01)  # Predict with TTA (Test-Time Augmentation)
intr_prob, intr_std = predict_proba(intrusion_norm, tta=16, noise_scale=0.01)
 
normal_feat_score = feature_anomaly_score(normal_sample)    # Feature anomaly scores   
intr_feat_score = feature_anomaly_score(intrusion_sample)

alpha = 0.75  # rely mostly on classifier, tune alpha on validation set
normal_comb = combined_score(normal_prob, normal_feat_score, alpha=alpha)
intr_comb = combined_score(intr_prob, intr_feat_score, alpha=alpha)
# Threshold: default 0.5, but should be calibrated on validation set using optimal_threshold_from_labels
THRESHOLD = 0.5


print("=== Normal sample ===")
print(f"classifier_prob={normal_prob:.4f} ±{normal_std:.4f}")
print(f"feature_score={normal_feat_score:.4f}")
print(f"combined_score={normal_comb:.4f} -> {detect_from_score(normal_comb, THRESHOLD)}\n")
print("=== Intrusion sample ===")
print(f"classifier_prob={intr_prob:.4f} ±{intr_std:.4f}")
print(f"feature_score={intr_feat_score:.4f}")
print(f"combined_score={intr_comb:.4f} -> {detect_from_score(intr_comb, THRESHOLD)}\n")


def range_based_truth(x_raw: np.ndarray) -> str:
    t, h, p, v = x_raw.tolist()  # <-- now unpacks 4 values
    is_intr = (40 <= t <= 50) and (90 <= h <= 150) and (600 <= p <= 800) and (7 <= v <= 12)
    is_norm = (15 <= t <= 35) and (30 <= h <= 80) and (900 <= p <= 1100) and (0 <= v <= 5)
    if is_intr:
        return "intrusion-by-ranges"
    if is_norm:
        return "normal-by-ranges"
    return "mixed/undefined-by-ranges"


print("=== Random samples ===")
for i in range(10):
    x_raw, true_label = random_sensor_sample(anomaly_prob=0.1)
    x_norm = normalize(x_raw)

    p, p_std = predict_proba(x_norm, tta=16, noise_scale=0.01)
    fs = feature_anomaly_score(x_raw)
    cs = combined_score(p, fs, alpha=alpha)

    pred = detect_from_score(cs, 0.5)

    print(f"\n--- Sample #{i} (true_label={true_label}) ---")
    print(f"x_raw={x_raw}")
    print(f"range_truth={range_based_truth(x_raw)}")
    print(f"classifier_prob={p:.4f} ±{p_std:.4f}")
    print(f"feature_score={fs:.4f}")
    print(f"combined_score={cs:.4f} -> {pred}")


print("\n=== Zoomed Boundary Testing (t=0.45 to t=0.55) ===")
print("Gradually blending all features from normal → intrusion\n")

for t in [round(x * 0.01, 2) for x in range(30, 50)]:
    temp      = 25.0   + t * (45.0  - 25.0)
    humidity  = 55.0   + t * (120.0 - 55.0)
    pressure  = 1000.0 + t * (700.0 - 1000.0)
    vibration = 2.5    + t * (9.5   - 2.5)

    x_raw = np.array([temp, humidity, pressure, vibration], dtype=np.float32)
    prob, _ = predict_proba(normalize(x_raw), tta=16, noise_scale=0.01)
    fs = feature_anomaly_score(x_raw)
    cs = combined_score(prob, fs, alpha=0.75)
    print(f"t={t:.2f} | temp={temp:.1f}, hum={humidity:.1f}, pres={pressure:.1f}, vib={vibration:.1f} → prob={prob:.4f}, combined={cs:.4f} → {detect_from_score(cs, THRESHOLD)}")

# ─────────────────────────────────────────────
# Precision & Recall
# ─────────────────────────────────────────────

print("\n=== Precision & Recall ===")

TP = 0  # Predicted intrusion, actually intrusion
FP = 0  # Predicted intrusion, actually normal
FN = 0  # Predicted normal,    actually intrusion
TN = 0  # Predicted normal,    actually normal

for _ in range(1000):
    x_raw, true_label = random_sensor_sample(anomaly_prob=0.1)
    prob, _ = predict_proba(normalize(x_raw), tta=16, noise_scale=0.01)
    fs = feature_anomaly_score(x_raw)
    cs = combined_score(prob, fs, alpha=0.75)
    predicted = 1 if cs >= THRESHOLD else 0

    if predicted == 1 and true_label == 1:
        TP += 1
    elif predicted == 1 and true_label == 0:
        FP += 1
    elif predicted == 0 and true_label == 1:
        FN += 1
    else:
        TN += 1

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1 Score  = {f1:.4f}")