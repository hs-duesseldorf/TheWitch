import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from charset_normalizer.cd import Counter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from k_means_constrained import KMeansConstrained
import pickle

DATASET_PATH = "hand_informs.json"
MODEL_SAVE_DIR = "../extract_hand_datas/hand_analysis_models"

""""
This is a script for training element classifying models!

Description of Models

1. Labeling with constrained K-Means
get a average of info from one person
divide average in 5 group for labeling dominant element
and divide again each group in 4 group for labeling weak element

2. Predict Element
Use Random Forest and XGBoost for predict dominant element
Use Random Forest and XGBoost and MLP for predict weak element
"""""

def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0

# change info to Absolute value to relative value
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
        safe_div(palm_width, palm_height),
        safe_div(avg_finger, palm_height),
        safe_div(index_length, ring_length),
        safe_div(index_length, max_finger),
        safe_div(middle_length, max_finger),
        safe_div(ring_length, max_finger),
        safe_div(pinky_length, max_finger)
    ]

#init mlp model
class SubWeakMLP(nn.Module):
    def __init__(self, input_dim=263, num_classes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x): return self.network(x)

if __name__ == "__main__":
    #set seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    ELEMENTS = ["holz", "feuer", "erde", "metall", "wasser"]

    #load dateset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    person_features = {}
    for key, item in data.items():
        #check exception
        vector = item.get("vector", [])
        if not vector or len(vector) != 256: continue
        lengths = item.get("lengths", {})
        if not lengths or lengths.get("middle_length", 0) == 0 or lengths.get("palm_height", 0) == 0: continue

        #gather every info from one person

        #add person_id if it is not in person_features
        person_id = str(item["request_id"]).split("-")[0].strip().upper()
        if person_id not in person_features:
            person_features[person_id] = []

        combined_vector = extract_features(lengths) + vector
        person_features[person_id].append(combined_vector)

    #sort by id
    unique_person_ids = sorted(list(person_features.keys()))

    #extract average from one person for labeling
    person_avg_vectors = np.array([np.mean(person_features[pid], axis=0) for pid in unique_person_ids],
                                  dtype=np.float32)

    # DOMINANT ELEMENT LABELING

    #define size of cluster
    n_people = len(unique_person_ids)
    dom_size = n_people // 5

    #define model
    kmeans_dom = KMeansConstrained(n_clusters=5, size_min=dom_size, size_max=dom_size + 1, random_state=42)
    # clustering
    dom_labels = kmeans_dom.fit_predict(person_avg_vectors)

    #add label to data
    person_to_dom = {pid: int(label) for pid, label in zip(unique_person_ids, dom_labels)}

    # WEAK ELEMENT LABELING
    person_to_weak_idx = {}
    for d_idx in range(5):
        #gather person in same dominant element
        sub_pids = [pid for pid in unique_person_ids if person_to_dom[pid] == d_idx]
        sub_vectors = np.array([np.mean(person_features[pid], axis=0) for pid in sub_pids], dtype=np.float32)

        #set size of cluster
        n_sub = len(sub_pids)
        weak_size = n_sub // 4

        #define model
        kmeans_weak = KMeansConstrained(n_clusters=4, size_min=weak_size, size_max=weak_size + 1, random_state=42)

        #clustering
        weak_labels = kmeans_weak.fit_predict(sub_vectors)

        # add label to data
        for pid, w_label in zip(sub_pids, weak_labels):
            person_to_weak_idx[pid] = int(w_label)

    #mapping each cluster to element so that dominant element and weak element are not a same element
    room_mapping = {d: [w for w in range(5) if d != w] for d in range(5)}

    print("labeling completed")

    #stratify: make sure that class labels are evenly distributed in train and test sets
    unique_Y_dom = [person_to_dom[pid] for pid in unique_person_ids]

    # test : train = 2 : 8
    train_pids, test_pids = train_test_split(unique_person_ids, test_size=0.2, random_state=42, stratify=unique_Y_dom)

    X_train_list, Y_dom_train, Y_weak_train = [], [], []
    X_test_list, Y_dom_test, Y_weak_test = [], [], []

    #make sure that one person in only test set or train set for valid evaluation
    for pid in unique_person_ids:
        d_label = person_to_dom[pid]
        w_label = person_to_weak_idx[pid]
        for feat in person_features[pid]:
            if pid in train_pids:
                X_train_list.append(feat)
                Y_dom_train.append(d_label)
                Y_weak_train.append(w_label)
            else:
                X_test_list.append(feat)
                Y_dom_test.append(d_label)
                Y_weak_test.append(w_label)

    X_train = np.array(X_train_list, dtype=np.float32)
    Y_dom_train = np.array(Y_dom_train, dtype=np.int64)
    Y_weak_train = np.array(Y_weak_train, dtype=np.int64)

    # check the ratio of element in test/train set
    train_counts = Counter(Y_dom_train)
    test_counts = Counter(Y_dom_test)
    for i, element in enumerate(ELEMENTS):
        print(f"[{element}] Train: {train_counts[i]} | Test : {test_counts[i]}")

    print("train, test set ready")

    # RF + XGBoost : dominant element training
    rf_model_s1 = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf_model_s1.fit(X_train, Y_dom_train)

    xgb_model_s1 = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_model_s1.fit(X_train, Y_dom_train)

    print("dominant element training completed")

    # MLP + rf + xgb : weak element training
    weak_mlp_models = {}
    weak_rf_models = {}
    weak_xgb_models = {}

    for d_idx in range(5):
        mask = (Y_dom_train == d_idx)
        X_sub, Y_sub = X_train[mask], Y_weak_train[mask]
        if len(X_sub) == 0: continue

        # mlp training
        weak_loader = DataLoader(
            TensorDataset(torch.tensor(X_sub, dtype=torch.float32), torch.tensor(Y_sub, dtype=torch.long)),
            batch_size=8, shuffle=True, drop_last=True)
        w_mlp = SubWeakMLP()
        w_criterion = nn.CrossEntropyLoss()
        w_optimizer = optim.Adam(w_mlp.parameters(), lr=0.001)

        w_mlp.train()
        for epoch in range(100):
            for bx, by in weak_loader:
                w_optimizer.zero_grad()
                w_criterion(w_mlp(bx), by).backward()
                w_optimizer.step()

        w_mlp.eval()
        weak_mlp_models[d_idx] = w_mlp

        # random forest training
        w_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
        w_rf.fit(X_sub, Y_sub)
        weak_rf_models[d_idx] = w_rf

        # XGBoost training
        w_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
        w_xgb.fit(X_sub, Y_sub)
        weak_xgb_models[d_idx] = w_xgb

    print("weak element training completed")

    # evaluation
    total_test_images = len(X_test_list)
    correct_dom = 0
    correct_weak = 0
    perfect_combo = 0

    for idx in range(total_test_images):
        feat = X_test_list[idx]
        true_d = Y_dom_test[idx]
        true_w = room_mapping[true_d][Y_weak_test[idx]]  # 실제 weak 원소

        feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        feat_np = np.array([feat], dtype=np.float32)

        with torch.no_grad():
            # dominant element
            p_dom = (rf_model_s1.predict_proba(feat_np)[0] + xgb_model_s1.predict_proba(feat_np)[0]) / 2.0
            pred_d = int(np.argmax(p_dom))

            # weak element
            mlp_logits = weak_mlp_models[pred_d](feat_t)

            mlp_probs = torch.softmax(mlp_logits, dim=1).squeeze(0).numpy()
            rf_probs = weak_rf_models[pred_d].predict_proba(feat_np)[0]
            xgb_probs = weak_xgb_models[pred_d].predict_proba(feat_np)[0]

            #voting
            final_weak_probs = (mlp_probs + rf_probs + xgb_probs) / 3.0

            pred_w_idx = int(np.argmax(final_weak_probs))
            pred_w = room_mapping[pred_d][pred_w_idx]

        # count score
        if pred_d == true_d:
            correct_dom += 1
        if pred_w == true_w:
            correct_weak += 1
        if pred_d == true_d and pred_w == true_w:
            perfect_combo += 1

    print(f"total test image set: {total_test_images}")
    print(f"dominant element accuracy : {correct_dom:<4} / {total_test_images} | {(correct_dom / total_test_images) * 100:.2f}%")
    print(f"weak element accuracy : {correct_weak:<4} / {total_test_images} | {(correct_weak / total_test_images) * 100:.2f}%")
    print(f"total accuracy : {perfect_combo:<4} / {total_test_images} | {(perfect_combo / total_test_images) * 100:.2f}%")

    #save models
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    # dominant element models
    with open("../extract_hand_datas/hand_analysis_models/rf_stage1.pkl", "wb") as f:
        pickle.dump(rf_model_s1, f)
    with open("../extract_hand_datas/hand_analysis_models/xgb_stage1.pkl", "wb") as f:
        pickle.dump(xgb_model_s1, f)

    print(f"dominant element models saved")

    # weak element models
    for d_idx in range(5):
        if d_idx in weak_mlp_models:
            # sub mlp
            mlp_path = os.path.join(MODEL_SAVE_DIR, f"sub_mlp_{d_idx}.pth")
            torch.save(weak_mlp_models[d_idx].state_dict(), mlp_path)

            # sub rf
            rf_path = os.path.join(MODEL_SAVE_DIR, f"sub_rf_{d_idx}.pkl")
            with open(rf_path, "wb") as f:
                pickle.dump(weak_rf_models[d_idx], f)

            # sub xgb
            xgb_path = os.path.join(MODEL_SAVE_DIR, f"sub_xgb_{d_idx}.pkl")
            with open(xgb_path, "wb") as f:
                pickle.dump(weak_xgb_models[d_idx], f)

    print(f"weak element models saved")


