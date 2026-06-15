import numpy as np
import pandas as pd
import re
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import normalize
import scipy.sparse as sp

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

warnings.filterwarnings("ignore")

# KONFIGURASI
SAMPLE_SIZE = 10_000
N_TOPICS = 20
MAX_FEATURES = 5_000
MAX_ITER_LDA = 50
RANDOM_STATE = 42
N_SPLITS = 5
DATASET_PATH = "training.1600000.processed.noemoticon.csv"
ALPHA_FK = 1.0

print("=== KONFIGURASI ===")
print(f"SAMPLE_SIZE  = {SAMPLE_SIZE:,}")
print(f"N_TOPICS     = {N_TOPICS}")
print(f"MAX_FEATURES = {MAX_FEATURES}")
print(f"MAX_ITER_LDA = {MAX_ITER_LDA}")
print(f"ALPHA_FK     = {ALPHA_FK}")

# 1. DATASET
def load_dataset(path, sample_size=None):
    print("=== LOAD DATASET ===")
    cols = ["polarity","id","date","query","user","text"]
    df = pd.read_csv(path, encoding="latin-1", header=None, names=cols)
    df = df[df["polarity"].isin([0, 4])].copy()
    df["label"] = (df["polarity"] == 4).astype(int)

    if sample_size and sample_size < len(df):
        n_each = sample_size // 2
        df_neg = df[df["label"]==0].sample(n=n_each, random_state=RANDOM_STATE)
        df_pos = df[df["label"]==1].sample(n=n_each, random_state=RANDOM_STATE)
        df = pd.concat([df_neg, df_pos], ignore_index=True)
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Total   : {len(df):,}")
    print(f"Positif : {df['label'].sum():,}")
    print(f"Negatif : {(df['label']==0).sum():,}")
    return df

# 2. PREPROCESSING
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"@\w+",           "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]",       "", text)
    return re.sub(r"\s+", " ", text).strip()

# 3. TOPIC MODEL (LDA sebagai PLSA)
class TopicModel:
    def __init__(self):
        self.lda = LatentDirichletAllocation(
            n_components=N_TOPICS,
            max_iter=MAX_ITER_LDA,
            learning_method='batch',
            random_state=RANDOM_STATE,
            doc_topic_prior=0.1,
            topic_word_prior=0.01,
            n_jobs=1,
            verbose=0
        )
        self.pw_z = None

    def fit(self, X):
        self.lda.fit(X)
        self.pw_z = (self.lda.components_ / self.lda.components_.sum(axis=1, keepdims=True)).astype(np.float32)
        return self

    def get_pz_d(self, X):
        pz_d = self.lda.transform(X).astype(np.float32)
        pz_d /= pz_d.sum(axis=1, keepdims=True) + 1e-10
        return pz_d

# 4. FISHER KERNEL
def compute_fisher_phi(topic_model, X_count, pz_d, fisher_info_ref=None):
    X_arr = (np.array(X_count.todense(), dtype=np.float32)
             if hasattr(X_count, 'todense')
             else np.array(X_count, dtype=np.float32))
    n_docs = X_arr.shape[0]
    K = N_TOPICS
    BATCH = 300

    phi    = np.zeros((n_docs, K), dtype=np.float32)
    sq_sum = np.zeros(K, dtype=np.float32)

    for s in range(0, n_docs, BATCH):
        e = min(s + BATCH, n_docs)
        Xb = X_arr[s:e]
        pzd_b = pz_d[s:e]

        num  = pzd_b[:, np.newaxis, :] * topic_model.pw_z.T[np.newaxis, :, :]
        denom = num.sum(axis=2, keepdims=True) + 1e-10
        pzdw = num / denom

        # Fisher score 
        fs = np.einsum('bv,bvk->bk', Xb, pzdw) / (pzd_b + 1e-10) 
        phi[s:e] = fs
        if fisher_info_ref is None:
            sq_sum += (fs ** 2).sum(axis=0)

    # Fisher information
    fi = sq_sum / n_docs + 1e-10 if fisher_info_ref is None else fisher_info_ref

    # Natural gradient 
    phi = phi / fi[np.newaxis, :]
    phi = normalize(phi, norm='l2')
    return phi, fi

def dot_kernel(A, B):
    K = A @ B.T
    if sp.issparse(K):
        K = K.toarray()
    return np.array(K, dtype=np.float64)

# 5. TIGA METODE CLASSIFIER
class HIST_SVM:
    def __init__(self, **kw):
        self.tfidf = TfidfVectorizer(
            max_features=MAX_FEATURES, sublinear_tf=True,
            min_df=2, stop_words='english', ngram_range=(1, 2))
        self.clf = LinearSVC(C=1.0, max_iter=3000, random_state=RANDOM_STATE)

    def fit(self, texts, labels):
        X = self.tfidf.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        return self.clf.predict(self.tfidf.transform(texts))

class PLSA_SVM:
    def __init__(self, **kw):
        self.count = CountVectorizer(
            max_features=MAX_FEATURES, min_df=2, stop_words='english')
        self.tm  = TopicModel()
        self.clf = LinearSVC(C=1.0, max_iter=3000, random_state=RANDOM_STATE)

    def fit(self, texts, labels):
        X = self.count.fit_transform(texts)
        self.tm.fit(X)
        Z_train = self.tm.get_pz_d(X)
        self.clf.fit(Z_train, labels)
        return self

    def predict(self, texts):
        X = self.count.transform(texts)
        Z_test = self.tm.get_pz_d(X)
        return self.clf.predict(Z_test)


class FK_SVM:
    def __init__(self, **kw):
        self.count = CountVectorizer(
            max_features=MAX_FEATURES, min_df=2, stop_words='english')
        self.tfidf = TfidfVectorizer(
            max_features=MAX_FEATURES, sublinear_tf=True, min_df=2, stop_words='english', ngram_range=(1, 2))
        self.tm = TopicModel()
        self.clf = SVC(kernel="precomputed", C=1.0, random_state=RANDOM_STATE, class_weight='balanced')
        self.phi_train = None
        self.fi_train = None
        self.Xtf_train = None

    def fit(self, texts, labels):
        # Count matrix → LDA
        X_count = self.count.fit_transform(texts)
        self.tm.fit(X_count)
        pz_d_tr = self.tm.get_pz_d(X_count)

        # Fisher phi training
        self.phi_train, self.fi_train = compute_fisher_phi(
            self.tm, X_count, pz_d_tr, fisher_info_ref=None)

        # TF-IDF training
        self.Xtf_train = normalize(self.tfidf.fit_transform(texts), norm='l2')

        # Hybrid kernel training: K_hat = K_tfidf + alpha * K_fisher
        K_tf = dot_kernel(self.Xtf_train, self.Xtf_train)
        K_fk = self.phi_train @ self.phi_train.T
        K_train = K_tf + ALPHA_FK * K_fk

        self.clf.fit(K_train, labels)
        return self

    def predict(self, texts):
        X_count = self.count.transform(texts)
        pz_d_te = self.tm.get_pz_d(X_count)

        # Fisher phi test 
        phi_test, _ = compute_fisher_phi(
            self.tm, X_count, pz_d_te, fisher_info_ref=self.fi_train)

        X_tfidf_te = normalize(self.tfidf.transform(texts), norm='l2')

        # Hybrid kernel test vs train
        K_tf   = dot_kernel(X_tfidf_te, self.Xtf_train)
        K_fk   = phi_test @ self.phi_train.T
        K_test  = K_tf + ALPHA_FK * K_fk

        return self.clf.predict(K_test)

# 6. EVALUASI
def evaluate(ModelClass, texts_tr, texts_te, y_tr, y_te):
    model  = ModelClass()
    model.fit(texts_tr, y_tr)
    y_pred = model.predict(texts_te)
    acc = accuracy_score(y_te, y_pred) * 100
    rec = recall_score(y_te, y_pred, average="binary", zero_division=0) * 100
    return acc, rec

# 7. EXPERIMENT 1
def experiment1(texts, labels):
    print("=== EXPERIMENT 1: 5 Fold Cross Validation ===")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    texts = np.array(texts)
    labels = np.array(labels)
    res = {m: {"acc": [], "rec": []} for m in ["HIST-SVM","PLSA-SVM","FK-SVM"]}
    MAP = [("HIST-SVM", HIST_SVM), ("PLSA-SVM", PLSA_SVM), ("FK-SVM", FK_SVM)]

    for fold, (tr_idx, te_idx) in enumerate(skf.split(texts, labels)):
        print(f"\nRound {fold+1}/{N_SPLITS} ===")
        tr_t, te_t = texts[tr_idx], texts[te_idx]
        tr_l, te_l = labels[tr_idx], labels[te_idx]
        for name, Cls in MAP:
            t0 = time.time()
            acc, rec = evaluate(Cls, tr_t, te_t, tr_l, te_l)
            res[name]["acc"].append(acc)
            res[name]["rec"].append(rec)
            print(f"{name:10s}  Acc={acc:.2f}%  Recall={rec:.2f}%  ({time.time()-t0:.1f}s)")
    return res

# 8. EXPERIMENT 2
def experiment2(texts, labels):
    print("=== EXPERIMENT 2: Training Sample Percentages ===")

    pcts = [0.30, 0.40, 0.50, 0.60, 0.70]
    plbls = ["30%","40%","50%","60%","70%"]
    texts = np.array(texts)
    labels = np.array(labels)

    X_pool, X_test, y_pool, y_test = train_test_split(texts, labels, test_size=0.30, random_state=RANDOM_STATE, stratify=labels)

    res = {m: {"acc": [], "rec": []} for m in ["HIST-SVM","PLSA-SVM","FK-SVM"]}
    MAP = [("HIST-SVM", HIST_SVM), ("PLSA-SVM", PLSA_SVM), ("FK-SVM", FK_SVM)]

    for pct in pcts:
        print(f"\nTraining: {int(pct*100)}% ===")
        n = max(int(len(X_pool) * pct / 0.70), 300)
        n = min(n, len(X_pool))
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_pool), size=n, replace=False)
        X_tr, y_tr = X_pool[idx], y_pool[idx]
        for name, Cls in MAP:
            t0 = time.time()
            acc, rec = evaluate(Cls, X_tr, X_test, y_tr, y_test)
            res[name]["acc"].append(acc)
            res[name]["rec"].append(rec)
            print(f"{name:10s}  Acc={acc:.2f}%  Recall={rec:.2f}%  ({time.time()-t0:.1f}s)")
    return res, plbls

# 9. PRINT TABEL
def _fmt_table(header, rows):
    if HAS_TABULATE:
        return tabulate(rows, headers=header, tablefmt="grid")
    cw = [max(len(str(r[i])) for r in [header]+rows) for i in range(len(header))]
    sep = "+-" + "-+-".join("-"*w for w in cw) + "-+"
    rs = lambda r: "| " + " | ".join(str(r[i]).ljust(cw[i]) for i in range(len(r))) + " |"
    return "\n".join([sep, rs(header), sep] + [rs(r) for r in rows] + [sep])

def _print_table(title, res, row_labels, col0):
    for lbl, key in [("(a) Precision (%)","acc"), ("(b) Recall Rate (%)","rec")]:
        print(f"\n  {lbl}")
        rows = [[row_labels[i],
                 f"{res['HIST-SVM'][key][i]:.2f}%",
                 f"{res['PLSA-SVM'][key][i]:.2f}%",
                 f"{res['FK-SVM'][key][i]:.2f}%"]
                for i in range(len(row_labels))]
        avgs = [np.mean(res[m][key]) for m in ["HIST-SVM","PLSA-SVM","FK-SVM"]]
        rows.append(["Average", f"{avgs[0]:.2f}%", f"{avgs[1]:.2f}%", f"{avgs[2]:.2f}%"])
        print(_fmt_table([col0,"HIST-SVM","PLSA-SVM","FK-SVM"], rows))

def print_table1(res):
    _print_table("TABLE 1 : 5-Fold Cross Validation",
                 res, [f"Round {i+1}" for i in range(N_SPLITS)], "Round")

def print_table2(res, pl):
    _print_table("TABLE 2 : Training Sample Percentages", res, pl, "Train%")

# 10. PLOT GRAFIK
COLORS  = {"HIST-SVM": "#1f77b4", "PLSA-SVM": "#ff7f0e", "FK-SVM": "#7f7f7f"}
METHODS = ["HIST-SVM", "PLSA-SVM", "FK-SVM"]

def _bar_chart(ax, x_labels, data_dict, title):
    x = np.arange(len(x_labels))
    w = 0.25
    all_v = [v for m in METHODS for v in data_dict[m]]
    lo = max(0, min(all_v) - 8)
    hi = min(100, max(all_v) + 7)

    for i, m in enumerate(METHODS):
        bars = ax.bar(x + i*w, data_dict[m], w, label=m, color=COLORS[m], edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, data_dict[m]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=6.5)

    ax.set_title(title, fontsize=10)
    ax.set_xticks(x + w)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylim(lo, hi)
    ax.set_ylabel("Score (%)")
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.35, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _save_fig(res, x_labels, fname, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    _bar_chart(axes[0], x_labels, {m: res[m]["acc"] for m in METHODS}, "(a) Precision")
    _bar_chart(axes[1], x_labels, {m: res[m]["rec"] for m in METHODS}, "(b) Recall Rate")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {fname}")

def plot_figure1(res):
    _save_fig(res, [f"Round{i+1}" for i in range(N_SPLITS)],
              "figure1_experiment1.png",
              "Figure 1. Multi-Round CV: FK-SVM vs HIST-SVM vs PLSA-SVM")

def plot_figure2(res, pl):
    _save_fig(res, pl,
              "figure2_experiment2.png",
              "Figure 2. Training % Comparison: FK-SVM vs HIST-SVM vs PLSA-SVM")



# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    try:
        df = load_dataset(DATASET_PATH, sample_size=SAMPLE_SIZE)
    except FileNotFoundError:
        print(f"\n[ERROR] File tidak ditemukan: {DATASET_PATH}")
        return

    print("\n=== PREPROCESSING ===")
    t0 = time.time()
    df["clean"] = df["text"].apply(preprocess_text)
    df = df[df["clean"].str.len() > 5].reset_index(drop=True)
    texts  = df["clean"].tolist()
    labels = df["label"].tolist()
    print(f"Selesai {time.time()-t0:.1f} detik")

    res1 = experiment1(texts, labels)
    print_table1(res1)
    plot_figure1(res1)

    res2, pl = experiment2(texts, labels)
    print_table2(res2, pl)
    plot_figure2(res2, pl)

if __name__ == "__main__":
    main()