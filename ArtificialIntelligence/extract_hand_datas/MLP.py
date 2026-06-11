import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np

# Configuration
DATASET_PATH = "hand_informs.json"
MLP_MODEL_PATH = "../hand_element_mode_87l.pth"


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0

def extract_features(lengths):
    palm_width = lengths.get("palm_width", 0.0)
    palm_height = lengths.get("palm_height", 0.0)

    index_length = lengths.get("index_length", 0.0)
    middle_length = lengths.get("middle_length", 0.0)
    ring_length = lengths.get("ring_length", 0.0)
    pinky_length = lengths.get("pinky_length", 0.0)

    finger_lengths = [index_length, middle_length, ring_length, pinky_length]
    max_finger = max(finger_lengths) if finger_lengths else 0.0
    avg_finger = sum(finger_lengths) / len(finger_lengths) if finger_lengths else 0.0

    return [
        safe_div(palm_width, palm_height),  # palm_aspect_ratio
        safe_div(avg_finger, palm_height),  # finger_length_ratio
        safe_div(index_length, ring_length),  # index_to_ring_ratio
        safe_div(index_length, max_finger),  # finger_profile.index
        safe_div(middle_length, max_finger),  # finger_profile.middle
        safe_div(ring_length, max_finger),  # finger_profile.ring
        safe_div(pinky_length, max_finger)  # finger_profile.little
    ]


class CompatibleMLP(nn.Module):
    def __init__(self, input_dim=7, num_classes=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing dataset file: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    person_features = {}
    for key, item in data.items():
        lengths = item.get("lengths", {})
        if not lengths or lengths.get("middle_length", 0) == 0 or lengths.get("palm_height", 0) == 0:
            continue

        feature_vector = extract_features(lengths)

        person_id = str(item["request_id"]).split("-")[0].strip().upper()

        if person_id not in person_features:
            person_features[person_id] = []
        person_features[person_id].append(feature_vector)

    unique_person_ids = sorted(list(person_features.keys()))
    person_avg_vectors = []
    for pid in unique_person_ids:
        avg_vector = np.mean(person_features[pid], axis=0)
        person_avg_vectors.append(avg_vector)

    person_avg_vectors = np.array(person_avg_vectors, dtype=np.float32)

    print(f"[INFO] Running K-Means exactly over {len(unique_person_ids)} unique persons...")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=15)
    person_labels = kmeans.fit_predict(person_avg_vectors)

    person_to_label = {pid: int(label) for pid, label in zip(unique_person_ids, person_labels)}

    X_list, Y_list = [], []
    for key, item in data.items():
        lengths = item.get("lengths", {})
        if not lengths or lengths.get("middle_length", 0) == 0 or lengths.get("palm_height", 0) == 0:
            continue

        person_id = str(item["request_id"]).split("-")[0].strip().upper()
        if person_id not in person_to_label:
            continue

        feature_vector = extract_features(lengths)
        X_list.append(feature_vector)
        Y_list.append(person_to_label[person_id])

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.int64)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # 모든 특징값을 균일한 스케일로 압축!

    print(f"[SUCCESS] Aligned dataset build completed. Total verified images: {len(X)}")

    counts = np.bincount(Y, minlength=5)
    print("\n" + "=" * 40)
    print(" Truly Cohesive Element Distribution")
    print("=" * 40)
    for class_idx in range(5):
        print(
            f"Cluster_{class_idx} | Total Imgs: {counts[class_idx]:<3} | Ratio: {counts[class_idx] / len(X) * 100:.1f}%")
    print("=" * 40 + "\n")

    # Train/Test 8:2 Split (사진 단위 분할로 안전하게 스케일 유지)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long)),
        batch_size=8, shuffle=True #8
    )

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    model = CompatibleMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)


    print("[INFO] Executing optimization backpropagation...")
    model.train()
    for epoch in range(250):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test_tensor), dim=1)
        total_correct = 0

        print("Evaluation Verification Status (Perfect Cohesion)")

        for class_idx in range(5):
            class_mask = (Y_test_tensor == class_idx)
            class_total = class_mask.sum().item()
            if class_total == 0: continue

            class_correct = ((preds == Y_test_tensor) & class_mask).sum().item()
            total_correct += class_correct
            print(
                f"Cluster_{class_idx} | Size: {class_total:<2} | Correct: {class_correct:<2} | Precision: {class_correct / class_total * 100:.1f}%")

        final_accuracy = (total_correct / Y_test_tensor.size(0)) * 100
        print("-" * 40)
        print(f"Total Combined Pipeline Accuracy: {final_accuracy:.2f}%")
        print("=" * 40 + "\n")

        torch.save(model.state_dict(), MLP_MODEL_PATH)
        print(f"[SUCCESS] Final robust model saved directly to: {MLP_MODEL_PATH}")
