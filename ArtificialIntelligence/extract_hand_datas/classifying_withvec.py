import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA

DATASET_PATH = "hand_informs.json"
MLP_MODEL_PATH = "../hand_element_model.pth"


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
    def __init__(self, input_dim=256, num_classes=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    # def __init__(self, input_dim=64, num_classes=5):
    #     super().__init__()
    #     self.network = nn.Sequential(
    #         nn.Linear(input_dim, 128),
    #         nn.ReLU(),
    #         nn.Dropout(0.2),
    #         nn.Linear(128, 64),
    #         nn.ReLU(),
    #         nn.Linear(64, 32),
    #         nn.ReLU(),
    #         nn.Linear(32, num_classes)
    #     )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    import random

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1 : grouping image per person
    person_features = {}
    for key, item in data.items():
        vector = item.get("vector", [])
        if not vector: continue

        person_id = str(item["request_id"]).split("-")[0].strip().upper()

        if person_id not in person_features:
            person_features[person_id] = []
        person_features[person_id].append(vector)

    # 2 : k means grouping
    # person_ids = sorted(list(person_features.keys()))
    # person_avg_vectors = []
    #
    # for p_id in person_ids:
    #     avg_vec = np.mean(person_features[p_id], axis=0)
    #     person_avg_vectors.append(avg_vec)
    #
    # person_avg_vectors = np.array(person_avg_vectors, dtype=np.float32)
    #
    # kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    # person_labels = kmeans.fit_predict(person_avg_vectors)
    #
    # person_to_label = {p_id: label for p_id, label in zip(person_ids, person_labels)}

    # 2 : constrained k-means grouping
    from k_means_constrained import KMeansConstrained

    unique_person_ids = sorted(list(person_features.keys()))
    person_avg_vectors = []
    for pid in unique_person_ids:
        avg_vector = np.mean(person_features[pid], axis=0)
        person_avg_vectors.append(avg_vector)
    person_avg_vectors = np.array(person_avg_vectors, dtype=np.float32)

    n_people = len(unique_person_ids)
    exact_size = n_people // 5

    print(f"Constrained KMeans (Target size per cluster: {exact_size})")
    kmeans = KMeansConstrained(
        n_clusters=5,
        size_min=exact_size,
        size_max=exact_size + 1,
        random_state=42
    )
    person_labels = kmeans.fit_predict(person_avg_vectors)

    person_to_label = {pid: int(label) for pid, label in zip(unique_person_ids, person_labels)}

    # 3 : labeling
    X_list, Y_list = [], []
    for key, item in data.items():
        lengths = item.get("lengths", {})
        if not lengths or lengths.get("middle_length", 0) == 0 or lengths.get("palm_height", 0) == 0:
            continue

        vector = item.get("vector", [])
        if not vector or len(vector) != 256:
            continue

        person_id = str(item["request_id"]).split("-")[0].strip().upper()
        if person_id not in person_to_label:
            continue

        # hand ratio
        feature_vector = extract_features(lengths)

        # hand line vec + hand ratio
        combined_vector = feature_vector + vector

        X_list.append(combined_vector)
        Y_list.append(person_to_label[person_id])

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.int64)

    # pca test
    # pca = PCA(n_components=64, random_state=42)
    # X = pca.fit_transform(X)
    # print(f"[INFO] PCA Transformation complete. New shape: {X.shape}")


    print(f"Data complete : Total verified images: {len(X)}")

    counts = np.bincount(Y, minlength=5)
    print("\n" + "=" * 40)
    print("Element Distribution")
    print("=" * 40)
    for class_idx in range(5):
        print(
            f"Cluster_{class_idx} | Total Imgs: {counts[class_idx]:<3} | Ratio: {counts[class_idx] / len(X) * 100:.1f}%")
    print("=" * 40 + "\n")

    # Train/Test 8:2 Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    #ensemble method

    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.metrics import accuracy_score
    #
    # print("Training Random Forest Ensemble Model...")
    # rf_model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=15)
    # rf_model.fit(X_train, Y_train)
    #
    # # eval
    # rf_preds = rf_model.predict(X_test)
    # rf_accuracy = accuracy_score(Y_test, rf_preds) * 100
    #
    # print("\n" + "=" * 40)
    # print(f"Random Forest Ensemble Accuracy: {rf_accuracy:.2f}%")
    # print("=" * 40 + "\n")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long)),
        batch_size=8, shuffle=True
    )
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    model = CompatibleMLP(input_dim=263, num_classes=5)
    # model = CompatibleMLP(input_dim=64, num_classes=5)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("model training")
    model.train()
    for epoch in range(300):
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

        print("\n" + "=" * 40)
        print("Evaluation Verification Status")
        print("=" * 40)
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

        # save model
        os.makedirs(os.path.dirname(MLP_MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MLP_MODEL_PATH)
        print(f"[SUCCESS] Final robust model saved directly to: {MLP_MODEL_PATH}")