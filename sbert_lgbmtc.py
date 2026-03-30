"""
================================================================
SBERT-LGBMTC: Tự động ước lượng Story Point trong Agile
Tái hiện từ: Yalçıner et al., Applied Sciences 2024
================================================================

Cài đặt thư viện:
    pip install sentence-transformers lightgbm scikit-learn pandas numpy

Dữ liệu TAWOS:
    https://zenodo.org/record/6363556

Chạy thử nhanh với dữ liệu mẫu tổng hợp (không cần tải TAWOS):
    python sbert_lgbmtc.py --demo

Chạy với dữ liệu thật:
    python sbert_lgbmtc.py --data path/to/tawos.csv
================================================================
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import re
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════
# PHẦN 1: TẠO DỮ LIỆU MẪU (dùng khi không có TAWOS)
# ════════════════════════════════════════════════════════════

def generate_demo_data(n_issues=500, n_projects=3, seed=42):
    """
    Tạo dữ liệu mẫu mô phỏng cấu trúc TAWOS.
    Mỗi issue có: title, description, issue_type, components, story_points.
    """
    rng = np.random.default_rng(seed)

    issue_types = ["Bug", "Task", "Story", "Enhancement", "Sub-task"]
    components  = ["UI", "Backend", "Database", "API", "Auth", "Cache", "Logging"]
    sp_values   = [1, 2, 3, 5, 8, 13, 20]          # Fibonacci-like

    templates = [
        ("Fix {comp} crash on startup",
         "Application crashes when {comp} module initializes. Stack trace points to null pointer."),
        ("Implement {comp} pagination",
         "Add pagination support to {comp} endpoint. Requires cursor-based approach for large datasets."),
        ("Refactor {comp} service",
         "The {comp} service has grown too large. Split into smaller, testable units."),
        ("Add {comp} unit tests",
         "Coverage for {comp} is below 60%. Write unit tests for edge cases and error paths."),
        ("Update {comp} documentation",
         "API docs for {comp} are outdated. Update with new endpoints and parameter descriptions."),
        ("Optimize {comp} query performance",
         "The {comp} queries are slow. Analyze execution plan and add appropriate indexes."),
        ("Integrate {comp} with third-party SDK",
         "Connect {comp} module to external SDK. Handle auth tokens and retry logic."),
    ]

    rows = []
    for proj_id in range(n_projects):
        proj_name = f"PROJECT_{chr(65 + proj_id)}"  # PROJECT_A, B, C
        n          = n_issues // n_projects

        for i in range(n):
            comp  = rng.choice(components)
            tmpl  = templates[i % len(templates)]
            sp    = int(rng.choice(sp_values, p=[0.25, 0.25, 0.20, 0.15, 0.08, 0.05, 0.02]))

            # Kết hợp 1–3 components ngẫu nhiên
            comp_list = "|".join(rng.choice(components,
                                            size=int(rng.integers(1, 4)),
                                            replace=False).tolist())
            rows.append({
                "project_key": proj_name,
                "title":       tmpl[0].format(comp=comp),
                "description": tmpl[1].format(comp=comp),
                "issue_type":  rng.choice(issue_types),
                "components":  comp_list,
                "story_points": sp,
                "created_date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(i * 2)),
            })

    df = pd.DataFrame(rows)
    print(f"[Demo] Đã tạo {len(df)} issues từ {n_projects} dự án mẫu.")
    return df


# ════════════════════════════════════════════════════════════
# PHẦN 2: TIỀN XỬ LÝ VĂN BẢN
# ════════════════════════════════════════════════════════════

def preprocess_text(text: str) -> str:
    """
    Làm sạch văn bản theo bài báo:
    - Xóa URL và ký tự đặc biệt
    - GIỮ dấu câu, stop words, từ ngắn (không stemming/lemmatize)
    """
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)          # xóa URL
    text = re.sub(r"[^\w\s.,!?;:()\-]", " ", text)       # xóa ký tự lạ
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text_column(df: pd.DataFrame) -> pd.Series:
    """Ghép title + description thành một chuỗi văn bản."""
    return (
        df["title"].fillna("").apply(preprocess_text) + ". " +
        df["description"].fillna("").apply(preprocess_text)
    )


# ════════════════════════════════════════════════════════════
# PHẦN 3: TRÍCH XUẤT ĐẶC TRƯNG VỚI SBERT
# ════════════════════════════════════════════════════════════

_sbert_model = None   # cache để không load lại nhiều lần

def get_sbert_model(model_name: str = "all-mpnet-base-v2"):
    """Load SBERT model (cache sau lần đầu)."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[SBERT] Đang tải model '{model_name}' ...")
        _sbert_model = SentenceTransformer(model_name)
        print("[SBERT] Tải xong.")
    return _sbert_model


def sbert_encode(texts: list, batch_size: int = 64) -> np.ndarray:
    """Chuyển danh sách văn bản → ma trận embedding (n, 768)."""
    model = get_sbert_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,   # theo bài báo: cosine sim tính ngoài
    )
    return embeddings.astype(np.float32)


# ════════════════════════════════════════════════════════════
# PHẦN 4: ENCODE ĐẶC TRƯNG PHÂN LOẠI
# ════════════════════════════════════════════════════════════

def label_encode_issue_type(train_s: pd.Series, val_s: pd.Series):
    """Label Encoding cho issue_type (RQ3 trong bài báo)."""
    le = LabelEncoder()
    le.fit(train_s.fillna("Unknown"))

    def safe_encode(series):
        return series.fillna("Unknown").map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        ).values.reshape(-1, 1)

    return safe_encode(train_s), safe_encode(val_s)


def onehot_encode_components(train_s: pd.Series, val_s: pd.Series):
    """One-Hot Encoding cho components (phân tách bởi '|')."""
    # Thu thập tất cả giá trị từ tập train
    all_comps = set()
    for val in train_s.fillna(""):
        for c in str(val).split("|"):
            c = c.strip()
            if c:
                all_comps.add(c)
    all_comps = sorted(all_comps)
    comp_idx  = {c: i for i, c in enumerate(all_comps)}

    def encode(series):
        mat = np.zeros((len(series), len(all_comps)), dtype=np.float32)
        for row_i, val in enumerate(series.fillna("")):
            for c in str(val).split("|"):
                c = c.strip()
                if c in comp_idx:
                    mat[row_i, comp_idx[c]] = 1.0
        return mat

    return encode(train_s), encode(val_s)


# ════════════════════════════════════════════════════════════
# PHẦN 5: METRICS ĐÁNH GIÁ
# ════════════════════════════════════════════════════════════

def compute_random_mae(y_true: np.ndarray, n_runs: int = 1000, seed: int = 0) -> float:
    """Tính MAE của random baseline (1000 lần, theo công thức SA trong bài báo)."""
    rng = np.random.default_rng(seed)
    maes = []
    for _ in range(n_runs):
        pred = rng.choice(y_true, size=len(y_true), replace=True)
        maes.append(mean_absolute_error(y_true, pred))
    return float(np.mean(maes))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             random_mae: float | None = None) -> dict:
    """
    Tính MAE, MdAE, SA.
    SA = (1 - MAE_model / MAE_random) * 100
    """
    mae  = mean_absolute_error(y_true, y_pred)
    mdae = float(np.median(np.abs(y_true - y_pred)))
    if random_mae is None:
        random_mae = compute_random_mae(y_true)
    sa = (1.0 - mae / random_mae) * 100.0
    return {"MAE": round(mae, 4), "MdAE": round(mdae, 4),
            "SA": round(sa, 2), "random_mae": round(random_mae, 4)}


# ════════════════════════════════════════════════════════════
# PHẦN 6: MÔ HÌNH LIGHTGBM (SBERT-LGBMTC)
# ════════════════════════════════════════════════════════════

def build_lgbm():
    """
    Tạo LGBMRegressor với hyperparameters manual tuning
    (phù hợp với thiết lập bài báo).
    """
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators      = 500,
        learning_rate     = 0.05,
        num_leaves        = 63,
        max_depth         = -1,
        min_child_samples = 20,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )


def train_and_evaluate(X_train, y_train, X_val, y_val,
                       random_mae: float | None = None):
    """Huấn luyện LightGBM và trả về (model, y_pred, metrics)."""
    from lightgbm import early_stopping, log_evaluation
    model = build_lgbm()
    model.fit(
        X_train, y_train,
        eval_set   = [(X_val, y_val)],
        callbacks  = [early_stopping(50, verbose=False),
                      log_evaluation(-1)],
    )
    y_pred  = model.predict(X_val)
    metrics = evaluate(y_val, y_pred, random_mae)
    return model, y_pred, metrics


# ════════════════════════════════════════════════════════════
# PHẦN 7: PIPELINE CHÍNH
# ════════════════════════════════════════════════════════════

def run_project(proj_df: pd.DataFrame, sbert_batch: int = 64) -> dict:
    """
    Chạy pipeline SBERT-LGBMTC cho một project:
    1. Chia 80/20 theo thứ tự thời gian
    2. Encode văn bản bằng SBERT
    3. Encode issue_type (Label) + components (One-Hot)
    4. Huấn luyện LightGBM
    5. Đánh giá và trả kết quả
    """
    proj_df = proj_df.sort_values("created_date").reset_index(drop=True)
    split   = int(len(proj_df) * 0.8)
    train   = proj_df.iloc[:split]
    val     = proj_df.iloc[split:]

    if len(val) < 5:
        return None   # Không đủ dữ liệu validation

    # ── Văn bản ──
    train_texts = build_text_column(train).tolist()
    val_texts   = build_text_column(val).tolist()
    all_emb     = sbert_encode(train_texts + val_texts, batch_size=sbert_batch)
    X_train_emb = all_emb[:len(train_texts)]
    X_val_emb   = all_emb[len(train_texts):]

    # ── Phân loại ──
    train_it, val_it     = label_encode_issue_type(train["issue_type"], val["issue_type"])
    train_comp, val_comp = onehot_encode_components(train["components"], val["components"])

    # ── Ghép đặc trưng ──
    X_train = np.hstack([X_train_emb, train_it, train_comp])
    X_val   = np.hstack([X_val_emb,   val_it,   val_comp])

    y_train = train["story_points"].values.astype(float)
    y_val   = val["story_points"].values.astype(float)

    # ── Random baseline ──
    random_mae = compute_random_mae(y_val)

    # ── Huấn luyện ──
    _, y_pred, metrics = train_and_evaluate(X_train, y_train, X_val, y_val, random_mae)
    metrics["n_train"] = len(train)
    metrics["n_val"]   = len(val)
    return metrics


def run_all_projects(df: pd.DataFrame, project_col: str = "project_key") -> pd.DataFrame:
    """Chạy pipeline cho từng project và tổng hợp kết quả."""
    projects = df[project_col].unique()
    records  = []

    for proj in projects:
        print(f"\n{'─'*55}")
        print(f" Project: {proj}")
        proj_df = df[df[project_col] == proj].copy()
        result  = run_project(proj_df)
        if result is None:
            print("  ⚠ Bỏ qua: dữ liệu validation quá nhỏ.")
            continue
        result["project"] = proj
        records.append(result)
        print(f"  MAE={result['MAE']:.3f}  MdAE={result['MdAE']:.3f}  SA={result['SA']:.1f}%"
              f"  (train={result['n_train']}, val={result['n_val']})")

    results_df = pd.DataFrame(records).set_index("project")
    return results_df


def print_summary(results_df: pd.DataFrame):
    """In tổng kết kết quả toàn bộ."""
    print(f"\n{'═'*55}")
    print(" KẾT QUẢ TỔNG HỢP")
    print(f"{'═'*55}")
    print(f"  Số project  : {len(results_df)}")
    print(f"  MAE TB      : {results_df['MAE'].mean():.3f}  ± {results_df['MAE'].std():.3f}")
    print(f"  MdAE TB     : {results_df['MdAE'].mean():.3f}  ± {results_df['MdAE'].std():.3f}")
    print(f"  SA TB       : {results_df['SA'].mean():.2f}%")
    pos_sa = (results_df["SA"] > 0).sum()
    print(f"  SA > 0      : {pos_sa}/{len(results_df)} dự án (vượt random baseline)")
    print(f"\n Bài báo gốc : MAE=2.15, MdAE=1.85, SA=93%")
    print(f"{'═'*55}")
    print("\nChi tiết theo project:")
    print(results_df[["MAE","MdAE","SA","n_train","n_val"]].to_string())


# ════════════════════════════════════════════════════════════
# PHẦN 8: ENTRY POINT
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SBERT-LGBMTC Story Point Estimator")
    parser.add_argument("--demo",    action="store_true",
                        help="Chạy với dữ liệu mẫu tổng hợp (không cần TAWOS)")
    parser.add_argument("--data",    type=str, default=None,
                        help="Đường dẫn tới file CSV TAWOS")
    parser.add_argument("--project", type=str, default="project_key",
                        help="Tên cột project (mặc định: project_key)")
    parser.add_argument("--batch",   type=int, default=64,
                        help="Batch size khi encode SBERT (mặc định: 64)")
    args = parser.parse_args()

    # ── Tải dữ liệu ──
    if args.demo or args.data is None:
        print("[Chế độ Demo] Dùng dữ liệu mẫu tổng hợp.")
        df = generate_demo_data()
    else:
        print(f"[Dữ liệu] Đang tải: {args.data}")
        df = pd.read_csv(args.data)
        print(f"  → {len(df)} issues, {df[args.project].nunique()} projects")

    # ── Chạy pipeline ──
    results = run_all_projects(df, project_col=args.project)

    # ── In kết quả ──
    print_summary(results)

    # ── Lưu kết quả ──
    out_path = "results_sbert_lgbmtc.csv"
    results.to_csv(out_path)
    print(f"\nĐã lưu kết quả → {out_path}")


if __name__ == "__main__":
    main()
