import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from k_means_constrained import KMeansConstrained

DATASET_PATH = "hand_informs.json"


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
        safe_div(palm_width, palm_height),
        safe_div(avg_finger, palm_height),
        safe_div(index_length, ring_length),
        safe_div(index_length, max_finger),
        safe_div(middle_length, max_finger),
        safe_div(ring_length, max_finger),
        safe_div(pinky_length, max_finger)
    ]


class SubWeakMLP(nn.Module):
    def __init__(self, input_dim=263, num_classes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x): return self.network(x)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    ELEMENTS = ["holz", "feuer", "erde", "metall", "wasser"]

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    person_features = {}
    for key, item in data.items():
        vector = item.get("vector", [])
        if not vector or len(vector) != 256: continue
        lengths = item.get("lengths", {})
        if not lengths or lengths.get("middle_length", 0) == 0 or lengths.get("palm_height", 0) == 0: continue

        person_id = str(item["request_id"]).split("-")[0].strip().upper()
        if person_id not in person_features:
            person_features[person_id] = []

        combined_vector = extract_features(lengths) + vector
        person_features[person_id].append(combined_vector)

    unique_person_ids = sorted(list(person_features.keys()))
    person_avg_vectors = np.array([np.mean(person_features[pid], axis=0) for pid in unique_person_ids],
                                  dtype=np.float32)

    # 1단계 KMeans 대륙 배분
    n_people = len(unique_person_ids)
    dom_size = n_people // 5
    kmeans_dom = KMeansConstrained(n_clusters=5, size_min=dom_size, size_max=dom_size + 1, random_state=42)
    dom_labels = kmeans_dom.fit_predict(person_avg_vectors)
    person_to_dom = {pid: int(label) for pid, label in zip(unique_person_ids, dom_labels)}

    # 2단계 내부 방 4등분
    person_to_weak_idx = {}
    for d_idx in range(5):
        sub_pids = [pid for pid in unique_person_ids if person_to_dom[pid] == d_idx]
        sub_vectors = np.array([np.mean(person_features[pid], axis=0) for pid in sub_pids], dtype=np.float32)
        n_sub = len(sub_pids)
        weak_size = n_sub // 4
        kmeans_weak = KMeansConstrained(n_clusters=4, size_min=weak_size, size_max=weak_size + 1, random_state=42)
        weak_labels = kmeans_weak.fit_predict(sub_vectors)
        for pid, w_label in zip(sub_pids, weak_labels):
            person_to_weak_idx[pid] = int(w_label)

    room_mapping = {d: [w for w in range(5) if d != w] for d in range(5)}

    unique_Y_dom = [person_to_dom[pid] for pid in unique_person_ids]
    train_pids, test_pids = train_test_split(unique_person_ids, test_size=0.2, random_state=42, stratify=unique_Y_dom)

    X_train_list, Y_dom_train, Y_weak_train = [], [], []
    X_test_list, Y_dom_test, Y_weak_test = [], [], []

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

    # ==========================================
    # 🚀 1단계 학습: 2대장 트리 앙상블 (RF + XGBoost)
    # ==========================================
    print("Training Step 1: Stage-1 Dual-Tree Ensemble...")
    rf_model_s1 = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf_model_s1.fit(X_train, Y_dom_train)

    xgb_model_s1 = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb_model_s1.fit(X_train, Y_dom_train)

    # ==========================================
    # 🚀 2단계 학습: 대륙별 [트리 2대장 + MLP] 3종 세트 빌드
    # ==========================================
    weak_mlp_models = {}
    weak_rf_models = {}
    weak_xgb_models = {}

    print("Training Step 2: Stage-2 Full Ensemble Models (MLP + RF + XGBoost)...")
    for d_idx in range(5):
        mask = (Y_dom_train == d_idx)
        X_sub, Y_sub = X_train[mask], Y_weak_train[mask]
        if len(X_sub) == 0: continue

        # A. 2단계용 MLP 학습
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

        # B. 2단계용 Random Forest 학습
        w_rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
        w_rf.fit(X_sub, Y_sub)
        weak_rf_models[d_idx] = w_rf

        # C. 2단계용 XGBoost 학습
        w_xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
        w_xgb.fit(X_sub, Y_sub)
        weak_xgb_models[d_idx] = w_xgb

    # ============================================================
    # 📉 최종 검증 및 개별 이미지 단위 스코어 채점
    # ============================================================
    total_test_images = len(X_test_list)
    correct_dom_count = 0
    correct_weak_count = 0
    perfect_combination_count = 0

    for idx in range(total_test_images):
        feat = X_test_list[idx]
        true_d = Y_dom_test[idx]
        true_w_idx = Y_weak_test[idx]
        true_w = room_mapping[true_d][true_w_idx]

        feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        feat_np = np.array([feat], dtype=np.float32)

        with torch.no_grad():
            # [1단계 추론] 트리 2대장 확률 결합
            p_dom = (rf_model_s1.predict_proba(feat_np)[0] + xgb_model_s1.predict_proba(feat_np)[0]) / 2.0
            pred_d = int(np.argmax(p_dom))

            # [2단계 추론] 지정 대륙 안에서 MLP + RF + XGB 3대장 확률 결합 (Soft-Voting)
            if pred_d in weak_mlp_models:
                # MLP 확률 추출 (Softmax 적용)
                mlp_logits = weak_mlp_models[pred_d](feat_t)
                mlp_probs = torch.softmax(mlp_logits, dim=1).squeeze(0).numpy()

                # 트리 2대장 확률 추출
                rf_probs = weak_rf_models[pred_d].predict_proba(feat_np)[0]
                xgb_probs = weak_xgb_models[pred_d].predict_proba(feat_np)[0]

                # 3개 모델 확률 평균 결합
                final_weak_probs = (mlp_probs + rf_probs + xgb_probs) / 3.0
                pred_w_idx = int(np.argmax(final_weak_probs))
                pred_w = room_mapping[pred_d][pred_w_idx]
            else:
                pred_w = (pred_d + 1) % 5

        if pred_d == true_d: correct_dom_count += 1
        if pred_w == true_w: correct_weak_count += 1
        if pred_d == true_d and pred_w == true_w: perfect_combination_count += 1

    print("\n" + "=" * 75)
    print("  📊 [2단계 풀 앙상블 적용] 최종 스테이지별 스코어 보드")
    print("=" * 75)
    print(f"📋 전체 테스트 손 이미지 수        : {total_test_images}장")
    print("-" * 75)
    print(
        f"✅ [1단계 2대장 트리] 대륙 맞춘 개수 : {correct_dom_count:<4} / {total_test_images}장 | 정확도: {(correct_dom_count / total_test_images) * 100:.2f}%")
    print(
        f"✅ [2단계 트리플 앙상블] 원소 맞춘 개수 : {correct_weak_count:<4} / {total_test_images}장 | 정확도: {(correct_weak_count / total_test_images) * 100:.2f}%")
    print("-" * 75)
    print(
        f"🔥 [최종 조합] 사주방 완전 일치 개수 : {perfect_combination_count:<4} / {total_test_images}장 | 종합 정확도: {(perfect_combination_count / total_test_images) * 100:.2f}%")
    print("=" * 75 + "\n")

    import pickle

    rf_s1_path = "../hand_analysis_models/rf_stage1.pkl"
    xgb_s1_path = "../hand_analysis_models/xgb_stage1.pkl"

    with open(rf_s1_path, "wb") as f:
        pickle.dump(rf_model_s1, f)
    with open(xgb_s1_path, "wb") as f:
        pickle.dump(xgb_model_s1, f)

    # 2. 2단계 대륙별 서브 MLP 5개 저장 (torch.save state_dict 활용)
    saved_mlp_count = 0
    for d_idx in range(5):
        if d_idx in weak_mlp_models:
            mlp_path = f"sub_mlp_{d_idx}.pth"
            # 가중치 딕셔너리(state_dict)만 쏙 빼서 콤팩트하게 저장
            torch.save(weak_mlp_models[d_idx].state_dict(), mlp_path)
            print(f"✅ [Stage 2] 대륙 {d_idx}번 서브 MLP 저장 완료 ➔ {mlp_path}")
            saved_mlp_count += 1

    # ============================================================
    # 🔍 [확장 검증] 경계선(Border Hand) 분석 및 정확도 변동성 측정
    # ============================================================
    total_test_images = len(X_test_list)

    border_count = 0
    clear_count = 0

    # 그룹별 맞은 개수 카운트
    correct_dom_border = 0
    correct_dom_clear = 0
    correct_weak_border = 0
    correct_weak_clear = 0

    # 경계선으로 판정된 조합(예: holz-feuer) 분포 체크용
    border_pair_counts = {}

    for idx in range(total_test_images):
        feat = X_test_list[idx]
        true_d = Y_dom_test[idx]
        true_w_idx = Y_weak_test[idx]
        true_w = room_mapping[true_d][true_w_idx]

        feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        feat_np = np.array([feat], dtype=np.float32)

        with torch.no_grad():
            # 1단계 대륙 확률 추론
            p_dom = (rf_model_s1.predict_proba(feat_np)[0] + xgb_model_s1.predict_proba(feat_np)[0]) / 2.0

            # 확률 높은 순으로 정렬
            sorted_indices = np.argsort(p_dom)[::-1]
            pred_d = int(sorted_indices[0])
            second_d = int(sorted_indices[1])

            # 🚨 유저님이 설정한 경계선 조건 (확률 차이 5% 이하)
            prob_diff = p_dom[pred_d] - p_dom[second_d]
            is_border = prob_diff <= 0.05

            # 2단계 최종 원소 추론
            if pred_d in weak_mlp_models:
                mlp_logits = weak_mlp_models[pred_d](feat_t)
                mlp_probs = torch.softmax(mlp_logits, dim=1).squeeze(0).numpy()
                rf_probs = weak_rf_models[pred_d].predict_proba(feat_np)[0]
                xgb_probs = weak_xgb_models[pred_d].predict_proba(feat_np)[0]

                final_weak_probs = (mlp_probs + rf_probs + xgb_probs) / 3.0
                pred_w_idx = int(np.argmax(final_weak_probs))
                pred_w = room_mapping[pred_d][pred_w_idx]
            else:
                pred_w = (pred_d + 1) % 5

        # 통계 데이터 분류 누적
        if is_border:
            border_count += 1
            if pred_d == true_d: correct_dom_border += 1
            if pred_w == true_w: correct_weak_border += 1

            # 어떤 경계선 조합이 많이 나오는지 기록 (알파벳 순 정렬해서 커플링)
            pair = "-".join(sorted([ELEMENTS[pred_d], ELEMENTS[second_d]]))
            border_pair_counts[pair] = border_pair_counts.get(pair, 0) + 1
        else:
            clear_count += 1
            if pred_d == true_d: correct_dom_clear += 1
            if pred_w == true_w: correct_weak_clear += 1

    # --------------------------------------------------------
    # 📈 [경계선 분석 결과 리포트 출력]
    # --------------------------------------------------------
    print("\n" + "=" * 75)
    print("🔮 [경계선(Border Hand) 분석 및 시뮬레이션 결과]")
    print("=" * 75)
    print(f"전체 테스트 데이터 : {total_test_images}장")
    print(f"  - 애매한 경계선 손 (차이 <= 5%) : {border_count}장 ({safe_div(border_count, total_test_images) * 100:.1f}%)")
    print(f"  - 확신 전형의 손   (차이 >  5%) : {clear_count}장 ({safe_div(clear_count, total_test_images) * 100:.1f}%)")
    print("-" * 75)

    print("[1단계 대륙 정확도 비교]")
    print(
        f"  - 경계선 그룹 정확도 : {safe_div(correct_dom_border, border_count) * 100:.2f}% ({correct_dom_border}/{border_count})")
    print(
        f"  - 확신 전형 정확도   : {safe_div(correct_dom_clear, clear_count) * 100:.2f}% ({correct_dom_clear}/{clear_count})")
    print("\n[2단계 최종 원소 정확도 비교]")
    print(
        f"  - 경계선 그룹 정확도 : {safe_div(correct_weak_border, border_count) * 100:.2f}% ({correct_weak_border}/{border_count})")
    print(
        f"  - 확신 전형 정확도   : {safe_div(correct_weak_clear, clear_count) * 100:.2f}% ({correct_weak_clear}/{clear_count})")
    print("-" * 75)

    print("[자주 발생하는 경계선 조합 Top 5 Distribution]")
    sorted_pairs = sorted(border_pair_counts.items(), key=lambda x: x[1], reverse=True)
    for pair, count in sorted_pairs[:5]:
        print(f"  - {pair:<15} : {count:>2}번 발생 ({safe_div(count, border_count) * 100:.1f}%)")
    print("=" * 75 + "\n")