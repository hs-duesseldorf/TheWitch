import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from k_means_constrained import KMeansConstrained

DATASET_PATH = "hand_informs.json"
DOMINANT_MODEL_PATH = "../hand_dominant_model.pth"
WEAK_MODEL_PATH = "../hand_weak_model.pth"


# 대각선을 제외한 순수한 20개 조합의 (Dominant, Weak) 좌표 매핑 테이블
COMBO_20_MAP = [
    (0, 1), (0, 2), (0, 3), (0, 4),  # holz 시작 (대각선 0,0 제외)
    (1, 0),         (1, 2), (1, 3), (1, 4),  # feuer 시작 (대각선 1,1 제외)
    (2, 0), (2, 1),         (2, 3), (2, 4),  # erde 시작 (대각선 2,2 제외)
    (3, 0), (3, 1), (3, 2),         (3, 4),  # metall 시작 (대각선 3,3 제외)
    (4, 0), (4, 1), (4, 2), (4, 3)           # wasser 시작 (대각선 4,4 제외)
]

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
    def __init__(self, input_dim=263, num_classes=5):
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

    # 1 : 사람별 이미지 그룹화
    person_features = {}
    for key, item in data.items():
        vector = item.get("vector", [])
        if not vector: continue
        person_id = str(item["request_id"]).split("-")[0].strip().upper()
        if person_id not in person_features:
            person_features[person_id] = []
        person_features[person_id].append(vector)

    # 2 : 사람별 평균 풀 데이터셋 구축 (263차원 전체 활용)
    unique_person_ids = sorted(list(person_features.keys()))
    person_full_vectors = []

    for pid in unique_person_ids:
        avg_vec = np.mean(person_features[pid], axis=0)

        ratios = []
        for key, item in data.items():
            if str(item["request_id"]).split("-")[0].strip().upper() == pid:
                lengths = item.get("lengths", {})
                if lengths and lengths.get("middle_length", 0) != 0 and lengths.get("palm_height", 0) != 0:
                    ratios.append(extract_features(lengths))
        if not ratios:
            ratios = [[0.0] * 7]
        avg_ratio = np.mean(ratios, axis=0)

        # 비율 피처(7) + 임베딩(256)을 합쳐 263차원 풀 벡터 생성
        person_full_vectors.append(np.hstack([avg_ratio, avg_vec]))

    person_full_vectors = np.array(person_full_vectors, dtype=np.float32)  # (N_people, 263)

    # 🌟 [20개 정공법] 20개 유효 조합에 Constrained 균등 크기 제약 걸기
    n_people = len(unique_person_ids)
    exact_cell_size = n_people // 20  # 약 9명 (기준점)

    # 🌟 덜 Strict하게 제약 조건 완화 (버퍼 확장)
    # 최소 4명에서 최대 20명까지 유연하게 허용
    min_buffer = max(1, exact_cell_size - 5)  # 하한선을 4명으로 완화
    max_buffer = exact_cell_size + 11  # 상한선을 20명으로 완화

    print(f"20-Combo Matrix [Less Strict] KMeans (Range: {min_buffer} ~ {max_buffer} per cell)")

    kmeans_combo = KMeansConstrained(
        n_clusters=20,
        size_min=min_buffer,  # 덜 Strict한 하한선
        size_max=max_buffer,  # 덜 Strict한 상한선
        random_state=42
    )
    combo_labels = kmeans_combo.fit_predict(person_full_vectors)

    # 20진법 라벨(0~19)을 미리 정의한 20개 좌표계로 다이렉트 매핑
    person_to_label = {}
    person_to_weak_label = {}

    for pid, c_label in zip(unique_person_ids, combo_labels):
        # 0~19 번호를 1:1 매핑 테이블에서 좌표로 치환
        dom_idx, weak_idx = COMBO_20_MAP[int(c_label)]

        person_to_label[pid] = dom_idx
        person_to_weak_label[pid] = weak_idx

    # 3 : 이미지 데이터셋 생성 파트
    X_list, Y_dom_list, Y_weak_list = [], [], []
    for key, item in data.items():
        lengths = item.get("lengths", {})
        if not lengths or lengths.get("middle_length", 0) == 0 or lengths.get("palm_height", 0) == 0:
            continue
        vector = item.get("vector", [])
        if not vector or len(vector) != 256:
            continue

        person_id = str(item["request_id"]).split("-")[0].strip().upper()
        if person_id not in person_to_label or person_id not in person_to_weak_label:
            continue

        feature_vector = extract_features(lengths)
        combined_vector = feature_vector + vector

        X_list.append(combined_vector)
        Y_dom_list.append(person_to_label[person_id])
        Y_weak_list.append(person_to_weak_label[person_id])

    X = np.array(X_list, dtype=np.float32)
    Y_dom = np.array(Y_dom_list, dtype=np.int64)
    Y_weak = np.array(Y_weak_list, dtype=np.int64)

    print(f"Data complete : Total verified images: {len(X)}")

    # 4 : Train/Test Split (Stratify 보존)
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42, stratify=Y_dom)

    X_train, X_test = X[idx_train], X[idx_test]
    Y_dom_train, Y_dom_test = Y_dom[idx_train], Y_dom[idx_test]
    Y_weak_train, Y_weak_test = Y_weak[idx_train], Y_weak[idx_test]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_dom_train, dtype=torch.long),
        torch.tensor(Y_weak_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_dom_test_tensor = torch.tensor(Y_dom_test, dtype=torch.long)
    Y_weak_test_tensor = torch.tensor(Y_weak_test, dtype=torch.long)

    # 5 : MLP 신경망 빌드
    dom_model = CompatibleMLP(input_dim=263, num_classes=5)
    weak_model = CompatibleMLP(input_dim=263, num_classes=5)

    criterion = nn.CrossEntropyLoss()
    dom_optimizer = optim.Adam(dom_model.parameters(), lr=0.001)
    weak_optimizer = optim.Adam(weak_model.parameters(), lr=0.001)

    # 6 : 동시 모델 트레이닝
    print("Training Dominant and Weak models...")
    dom_model.train()
    weak_model.train()
    for epoch in range(300):
        for batch_x, batch_y_dom, batch_y_weak in train_loader:
            dom_optimizer.zero_grad()
            dom_loss = criterion(dom_model(batch_x), batch_y_dom)
            dom_loss.backward()
            dom_optimizer.step()

            weak_optimizer.zero_grad()
            weak_loss = criterion(weak_model(batch_x), batch_y_weak)
            weak_loss.backward()
            weak_optimizer.step()

    # 7 : 종합 검증 및 결과 행렬 도출
    dom_model.eval()
    weak_model.eval()
    with torch.no_grad():
        dom_preds = torch.argmax(dom_model(X_test_tensor), dim=1)
        weak_preds = torch.argmax(weak_model(X_test_tensor), dim=1)

        dom_accuracy = ((dom_preds == Y_dom_test_tensor).sum().item() / Y_dom_test_tensor.size(0)) * 100
        weak_accuracy = ((weak_preds == Y_weak_test_tensor).sum().item() / Y_weak_test_tensor.size(0)) * 100

        print("\n" + "=" * 60)
        print("  Dual Model Pipeline Result Verification")
        print("=" * 60)
        print(f"Dominant Model Accuracy : {dom_accuracy:.2f}%")
        print(f"Weak Model Accuracy     : {weak_accuracy:.2f}%")
        print("=" * 60 + "\n")

        # 7-1. 테스트 셋 매트릭스
        ELEMENTS = ["holz", "feuer", "erde", "metall", "wasser"]
        combination_matrix = np.zeros((5, 5), dtype=np.int64)
        for i in range(len(X_test_tensor)):
            combination_matrix[dom_preds[i].item()][weak_preds[i].item()] += 1

        print("=" * 60)
        print("  [FINAL TEST PREDICTIONS] 25 원소 조합 최종 균등 매트릭스")
        print("=" * 60)
        header = f"{'DOM \\ WEAK':<12}" + "".join([f"{ELEMENTS[j]:>10}" for j in range(5)])
        print(header)
        print("-" * 60)
        for d in range(5):
            row_str = f"{ELEMENTS[d]:<12}"
            for w in range(5):
                if d == w:
                    row_str += f"{'-':>10}"
                else:
                    count = combination_matrix[d][w]
                    pct = (count / len(X_test_tensor)) * 100
                    row_str += f"{count:>4}({pct:.1f}%)"
            print(row_str)
        print("=" * 60 + "\n")

        # 7-2. 일관성 검증 로그
        print("=" * 60)
        print("  동일 인물의 다중 이미지 기준 - 추론 일관성(Consistency) 검증")
        print("=" * 60)
        test_person_imgs = {}
        for idx in idx_test:
            item = data[list(data.keys())[idx]]
            pid = str(item["request_id"]).split("-")[0].strip().upper()
            if pid not in test_person_imgs:
                test_person_imgs[pid] = []
            test_person_imgs[pid].append(X[idx])

        checked = 0
        for pid, imgs in test_person_imgs.items():
            if len(imgs) < 2 or checked >= 3: continue
            print(f"👤 검증 대상 유저 [ {pid} ] (보유 이미지 수: {len(imgs)}장)")
            for img_idx, img_data in enumerate(imgs):
                test_x = torch.from_numpy(np.array([img_data])).float()
                pred_d = torch.argmax(dom_model(test_x)).item()
                pred_w = torch.argmax(weak_model(test_x)).item()
                print(f"   └ 📷 이미지 {img_idx + 1} -> 🟢 Dom: {ELEMENTS[pred_d]:<6} | 🔴 Weak: {ELEMENTS[pred_w]:<6}")
            print("-" * 60)
            checked += 1

        # 7-3. [FULL DATASET] 전체 데이터 기준 매트릭스
        print("\n" + "=" * 60)
        print("  [FULL DATASET] 전체 데이터 기준 25 원소 조합 매트릭스")
        print("=" * 60)
        X_all_tensor = torch.tensor(X, dtype=torch.float32)
        dom_all_preds = torch.argmax(dom_model(X_all_tensor), dim=1)
        weak_all_preds = torch.argmax(weak_model(X_all_tensor), dim=1)

        full_combination_matrix = np.zeros((5, 5), dtype=np.int64)
        for i in range(len(X_all_tensor)):
            full_combination_matrix[dom_all_preds[i].item()][weak_all_preds[i].item()] += 1

        header_all = f"{'DOM \\ WEAK':<12}" + "".join([f"{ELEMENTS[j]:>10}" for j in range(5)])
        print(header_all)
        print("-" * 60)
        for d in range(5):
            row_str = f"{ELEMENTS[d]:<12}"
            for w in range(5):
                if d == w:
                    row_str += f"{'-':>10}"
                else:
                    count = full_combination_matrix[d][w]
                    pct = (count / len(X_all_tensor)) * 100
                    row_str += f"{count:>4}({pct:.1f}%)"
            print(row_str)
        print("=" * 60 + "\n")

        # 8 : 저장
        os.makedirs(os.path.dirname(DOMINANT_MODEL_PATH), exist_ok=True)
        torch.save(dom_model.state_dict(), DOMINANT_MODEL_PATH)
        torch.save(weak_model.state_dict(), WEAK_MODEL_PATH)
        print(f"[SUCCESS] Dominant model saved to: {DOMINANT_MODEL_PATH}")
        print(f"[SUCCESS] Weak model saved to: {WEAK_MODEL_PATH}")

        # ============================================================
        # 🌟 [회의록 반영] KL-Divergence 분포 오차 정량 측정 파트
        # ============================================================
        print("\n" + "=" * 60)
        print("  📊 기획팀 회의록 기반: KL-Divergence 분포 검증")
        print("=" * 60)

        # 1. 기획팀의 Ideal Spread (20개 칸이 완벽하게 균등할 확률 = 칸당 5%)
        # 대각선을 제외한 20개 칸이므로 각각 1/20 = 0.05의 확률을 가집니다.
        P_ideal = np.full(20, 1.0 / 20.0)

        # 2. Actual Spread (현재 모델이 전체 데이터 세트에서 예측한 20개 칸의 실제 확률)
        # 25개 매트릭스에서 대각선을 제외한 유효 20개 칸의 카운트만 순서대로 추출합니다.
        actual_counts = []
        for d in range(5):
            for w in range(5):
                if d != w:
                    actual_counts.append(full_combination_matrix[d][w])

        actual_counts = np.array(actual_counts, dtype=np.float32)
        total_predictions = np.sum(actual_counts)

        # 0명이 나오는 칸이 있으면 로그 계산 시 무한대(inf)가 되므로
        # 아주 미세한 값(1e-7)을 더해주는 라플라스 스무딩(Laplace Smoothing) 적용
        Q_actual = (actual_counts + 1e-7) / (total_predictions + 1e-7 * 20)

        # 3. 🧪 KL-Divergence 공식 다이렉트 계산 (Formula: sum(P * log(P / Q)))
        # P(기획팀 이상향)를 기준으로 Q(현재 모델)가 얼마나 발산(Divergence)했는지 측정
        kl_div = np.sum(P_ideal * np.log(P_ideal / Q_actual))

        print(f"▶️ 기획팀 목표 분포 (Desired Spread) : 전 칸 균등 5.0%")
        print(f"▶️ 현재 모델의 KL-Divergence 스코어 : {kl_div:.4f}")

        # 4. 직관적인 직관 점수(오차율)로 환산 (0에 가까울수록 완벽한 황금 밸런스)
        if kl_div < 0.05:
            status = "🟢 [최상] 기획팀이 원하는 신의 황금 밸런스 데이터셋입니다."
        elif kl_div < 0.2:
            status = "🟡 [양호] 약간의 쏠림이 있으나 실제 서비스에 무방합니다."
        else:
            status = "🔴 [경고] 특정 조합 몰빵/전멸 심각! 데이터 추가 수집 필요."
        print(f"▶️ 종합 밸런스 판정: {status}")
        print("=" * 60 + "\n")