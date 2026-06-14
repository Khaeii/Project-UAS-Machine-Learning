import numpy as np
import pandas as pd
import re
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

warnings.filterwarnings("ignore")

# KONFIGURASI 
SAMPLE_SIZE   = 20_000
N_TOPICS      = 20 
MAX_FEATURES  = 5_000
MAX_ITER_EM   = 50 
RANDOM_STATE  = 42
N_SPLITS      = 5 
DATASET_PATH  = "training.1600000.processed.noemoticon.csv"


# 1. LOAD & PREPROCESSING
def load_dataset(path, sample_size=None):
    print(f"\n{'='*60}")
    print("1. LOADING DATASET")
    print(f"{'='*60}")
    cols = ["polarity","id","date","query","user","text"]
    df = pd.read_csv(path, encoding="latin-1", header=None, names=cols)
    df = df[df["polarity"].isin([0, 4])].copy()
    df["label"] = (df["polarity"] == 4).astype(int)

    if sample_size and sample_size < len(df):
        # Fix: stratified sampling tanpa groupby (kompatibel semua versi pandas)
        n_each = sample_size // 2
        df_neg = df[df["label"] == 0].sample(n=n_each, random_state=RANDOM_STATE)
        df_pos = df[df["label"] == 1].sample(n=n_each, random_state=RANDOM_STATE)
        df = pd.concat([df_neg, df_pos], ignore_index=True)
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle

    print(f"  Total records : {len(df):,}")
    print(f"  Positif       : {df['label'].sum():,}")
    print(f"  Negatif       : {(df['label']==0).sum():,}")
    return df


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 2. PLSA - MEMORY OPTIMIZED
class PLSA:
    """
    PLSA dengan EM algorithm - versi hemat memori.
    Tidak menyimpan matriks (D,V,K) secara penuh.
    """
    def __init__(self, n_topics=10, max_iter=20, random_state=42):
        self.n_topics    = n_topics
        self.max_iter    = max_iter
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        X_arr = np.array(X.todense(), dtype=np.float32) if hasattr(X, 'todense') \
                else np.array(X, dtype=np.float32)
        n_docs, n_words = X_arr.shape
        K = self.n_topics

        # Inisialisasi
        self.pz_d = np.random.dirichlet(np.ones(K), size=n_docs).astype(np.float32)
        self.pw_z = np.random.dirichlet(np.ones(n_words), size=K).astype(np.float32)

        # BATCH SIZE untuk hemat RAM: proses per-batch agar (batch,V,K) kecil
        BATCH = min(500, n_docs)

        prev_ll = -np.inf
        for it in range(self.max_iter):
            pz_d_new = np.zeros_like(self.pz_d)
            pw_z_new = np.zeros_like(self.pw_z)

            for start in range(0, n_docs, BATCH):
                end   = min(start + BATCH, n_docs)
                Xb    = X_arr[start:end]          # (B, V)
                pzd_b = self.pz_d[start:end]      # (B, K)

                # E-step: P(z|d,w) shape (B,V,K)
                num   = pzd_b[:, np.newaxis, :] * self.pw_z.T[np.newaxis, :, :]
                denom = num.sum(axis=2, keepdims=True) + 1e-10
                pz_dw = num / denom

                # M-step accumulate
                weighted = Xb[:, :, np.newaxis] * pz_dw     # (B,V,K)
                pz_d_new[start:end] = weighted.sum(axis=1)   # (B,K)
                pw_z_new += weighted.sum(axis=0).T           # (K,V)

            # Normalize
            self.pz_d = pz_d_new / (pz_d_new.sum(axis=1, keepdims=True) + 1e-10)
            self.pw_z = pw_z_new / (pw_z_new.sum(axis=1, keepdims=True) + 1e-10)

            # Log-likelihood
            ll = 0.0
            for start in range(0, n_docs, BATCH):
                end  = min(start + BATCH, n_docs)
                Xb   = X_arr[start:end]
                pdw  = (self.pz_d[start:end, np.newaxis, :] *
                        self.pw_z.T[np.newaxis, :, :]).sum(axis=2)
                ll  += (Xb * np.log(pdw + 1e-10)).sum()

            if (it + 1) % 5 == 0:
                print(f"   EM iter {it+1:2d}/{self.max_iter}  LL={ll:.1f}")

            if abs(ll - prev_ll) < 0.5:
                print(f"   Converged at iter {it+1}")
                break
            prev_ll = ll

        del X_arr
        return self

    def transform(self, X):
        return self.pz_d

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def infer(self, X, n_iter=15):
        """Infer P(z|d) untuk data baru (test set)."""
        X_arr = np.array(X.todense(), dtype=np.float32) if hasattr(X, 'todense') \
                else np.array(X, dtype=np.float32)
        n_docs = X_arr.shape[0]
        K = self.n_topics
        BATCH = min(500, n_docs)

        pz_d = np.random.RandomState(self.random_state).dirichlet(
            np.ones(K), size=n_docs).astype(np.float32)

        for _ in range(n_iter):
            pz_d_new = np.zeros_like(pz_d)
            for start in range(0, n_docs, BATCH):
                end  = min(start + BATCH, n_docs)
                Xb   = X_arr[start:end]
                pzd_b = pz_d[start:end]
                num  = pzd_b[:, np.newaxis, :] * self.pw_z.T[np.newaxis, :, :]
                denom = num.sum(axis=2, keepdims=True) + 1e-10
                pz_dw = num / denom
                pz_d_new[start:end] = (Xb[:, :, np.newaxis] * pz_dw).sum(axis=1)
            pz_d = pz_d_new / (pz_d_new.sum(axis=1, keepdims=True) + 1e-10)

        return pz_d



# 3. FISHER KERNEL - MEMORY OPTIMIZED
def compute_fisher_phi(plsa_model, X):
    """
    Hitung Fisher feature vector φ(X) = I^{-1} * Ux
    secara batch untuk hemat RAM.
    """
    X_arr = np.array(X.todense(), dtype=np.float32) if hasattr(X, 'todense') \
            else np.array(X, dtype=np.float32)
    n_docs = X_arr.shape[0]
    K = plsa_model.n_topics
    BATCH = min(500, n_docs)

    phi = np.zeros((n_docs, K), dtype=np.float32)

    # Untuk Fisher info matrix (diagonal approximation)
    fisher_scores_sq_sum = np.zeros(K, dtype=np.float32)

    for start in range(0, n_docs, BATCH):
        end   = min(start + BATCH, n_docs)
        Xb    = X_arr[start:end]
        pzd_b = plsa_model.pz_d[start:end]

        # E-step P(z|d,w)
        num   = pzd_b[:, np.newaxis, :] * plsa_model.pw_z.T[np.newaxis, :, :]
        denom = num.sum(axis=2, keepdims=True) + 1e-10
        pz_dw = num / denom

        # Fisher score: Ux = sum_w n(d,w)*P(z|d,w) / P(z|d)
        fs = np.einsum('bv,bvk->bk', Xb, pz_dw) / (pzd_b + 1e-10)  # (B,K)
        phi[start:end] = fs
        fisher_scores_sq_sum += (fs**2).sum(axis=0)

    # Fisher information I ≈ diag(E[Ux^2])  (Eq. 3)
    fisher_info = fisher_scores_sq_sum / n_docs + 1e-10

    # Natural gradient φ = I^{-1} * Ux  (Eq. 12)
    phi = phi / fisher_info[np.newaxis, :]
    return phi   # (D, K)


# Fisher kernel: K(Xi,Xj) = φ(Xi)^T φ(Xj)  (Eq. 14 & 16)
def fisher_kernel_matrix(phi_a, phi_b):
    return phi_a @ phi_b.T  # (Da, Db)

# 4. TIGA METODE
class HIST_SVM:
    """TF-IDF histogram + LinearSVC  (Section 3.4)"""
    def __init__(self, **kw):
        self.vec = TfidfVectorizer(max_features=MAX_FEATURES,
                                   sublinear_tf=True, min_df=2)
        self.clf = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, labels)
        return self

    def predict(self, texts):
        return self.clf.predict(self.vec.transform(texts))


class PLSA_SVM:
    """PLSA Z-vector + LinearSVC  (Section 3.4)"""
    def __init__(self, **kw):
        self.vec  = CountVectorizer(max_features=MAX_FEATURES, min_df=2)
        self.plsa = PLSA(n_topics=N_TOPICS, max_iter=MAX_ITER_EM,
                         random_state=RANDOM_STATE)
        self.clf  = LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        Z = self.plsa.fit_transform(X)
        self.clf.fit(Z, labels)
        return self

    def predict(self, texts):
        X = self.vec.transform(texts)
        Z = self.plsa.infer(X)
        return self.clf.predict(Z)


class FK_SVM:
    """Fisher Kernel (PLSA) + SVM precomputed  (metode utama jurnal)"""
    def __init__(self, **kw):
        self.vec       = CountVectorizer(max_features=MAX_FEATURES, min_df=2)
        self.plsa      = PLSA(n_topics=N_TOPICS, max_iter=MAX_ITER_EM, random_state=RANDOM_STATE)
        self.clf       = SVC(kernel="precomputed", C=1.0, random_state=RANDOM_STATE)
        self.phi_train = None

    def fit(self, texts, labels):
        X = self.vec.fit_transform(texts)
        self.plsa.fit(X)
        self.phi_train = compute_fisher_phi(self.plsa, X)   # (D_tr, K)
        K_tr           = fisher_kernel_matrix(self.phi_train, self.phi_train)
        self.clf.fit(K_tr, labels)
        return self

    def predict(self, texts):
        X = self.vec.transform(texts)
        phi_test = self._infer_phi(X)
        K_te     = fisher_kernel_matrix(phi_test, self.phi_train)
        return self.clf.predict(K_te)

    def _infer_phi(self, X):
        """Fisher phi untuk test docs."""
        pz_d = self.plsa.infer(X)
        # Simpan pz_d sementara
        orig = self.plsa.pz_d
        self.plsa.pz_d = pz_d
        phi = compute_fisher_phi(self.plsa, X)
        self.plsa.pz_d = orig
        return phi

# 5. EVALUASI
def evaluate(ModelClass, texts_tr, texts_te, y_tr, y_te):
    model = ModelClass()
    model.fit(texts_tr, y_tr)
    y_pred = model.predict(texts_te)
    acc = accuracy_score(y_te, y_pred) * 100
    rec = recall_score(y_te, y_pred, average="binary", zero_division=0) * 100
    return acc, rec


# 6. EXPERIMENT 1 – 5-Fold CV
def experiment1(texts, labels):
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: 5-Fold Cross Validation (Table 1 & Figure 2)")
    print(f"{'='*60}")

    skf    = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RANDOM_STATE)
    texts  = np.array(texts)
    labels = np.array(labels)

    res = {m: {"acc":[], "rec":[]} for m in ["HIST-SVM","PLSA-SVM","FK-SVM"]}
    MAP = {"HIST-SVM": HIST_SVM, "PLSA-SVM": PLSA_SVM, "FK-SVM": FK_SVM}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n  ── Round {fold+1}/{N_SPLITS} ──")
        tr_t, te_t = texts[tr_idx], texts[te_idx]
        tr_l, te_l = labels[tr_idx], labels[te_idx]

        for name, Cls in MAP.items():
            t0 = time.time()
            acc, rec = evaluate(Cls, tr_t, te_t, tr_l, te_l)
            res[name]["acc"].append(acc)
            res[name]["rec"].append(rec)
            print(f"    {name:10s}  Acc={acc:.2f}%  Recall={rec:.2f}%"
                  f"  ({time.time()-t0:.1f}s)")
    return res

# 7. EXPERIMENT 2 – Variasi % Training
def experiment2(texts, labels):
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Training Percentages (Table 2 & Figure 3)")
    print(f"{'='*60}")

    percentages = [0.30, 0.40, 0.50, 0.60, 0.70]
    pct_labels  = ["30%","40%","50%","60%","70%"]
    texts  = np.array(texts)
    labels = np.array(labels)

    X_pool, X_test, y_pool, y_test = train_test_split(
        texts, labels, test_size=0.30,
        random_state=RANDOM_STATE, stratify=labels)

    res = {m: {"acc":[], "rec":[]} for m in ["HIST-SVM","PLSA-SVM","FK-SVM"]}
    MAP = {"HIST-SVM": HIST_SVM, "PLSA-SVM": PLSA_SVM, "FK-SVM": FK_SVM}

    for pct in percentages:
        print(f"\n  ── Training proportion: {int(pct*100)}% ──")
        n_train = max(int(len(X_pool) * pct / 0.70), 200)
        n_train = min(n_train, len(X_pool))
        rng     = np.random.RandomState(RANDOM_STATE)
        idx     = rng.choice(len(X_pool), size=n_train, replace=False)
        X_tr, y_tr = X_pool[idx], y_pool[idx]

        for name, Cls in MAP.items():
            t0 = time.time()
            acc, rec = evaluate(Cls, X_tr, X_test, y_tr, y_test)
            res[name]["acc"].append(acc)
            res[name]["rec"].append(rec)
            print(f"    {name:10s}  Acc={acc:.2f}%  Recall={rec:.2f}%"
                  f"  ({time.time()-t0:.1f}s)")

    return res, pct_labels

# 8. PRINT TABEL
def _fmt_table(header, rows):
    if HAS_TABULATE:
        return tabulate(rows, headers=header, tablefmt="grid")
    # Fallback tanpa tabulate
    col_w = [max(len(str(r[i])) for r in [header]+rows) for i in range(len(header))]
    sep   = "+-" + "-+-".join("-"*w for w in col_w) + "-+"
    def row_str(r):
        return "| " + " | ".join(str(r[i]).ljust(col_w[i]) for i in range(len(r))) + " |"
    lines = [sep, row_str(header), sep] + [row_str(r) for r in rows] + [sep]
    return "\n".join(lines)

def print_table1(res):
    print(f"\n{'='*65}")
    print("TABLE 1 – 5-Fold Cross Validation Results")
    print(f"{'='*65}")
    rounds = [f"Round {i+1}" for i in range(N_SPLITS)]
    for label, key in [("(a) Precision (%)", "acc"), ("(b) Recall Rate (%)", "rec")]:
        print(f"\n  {label}")
        header = ["Round", "HIST-SVM", "PLSA-SVM", "FK-SVM"]
        rows   = [[r,
                   f"{res['HIST-SVM'][key][i]:.2f}%",
                   f"{res['PLSA-SVM'][key][i]:.2f}%",
                   f"{res['FK-SVM'][key][i]:.2f}%"]
                  for i, r in enumerate(rounds)]
        avgs   = [np.mean(res[m][key]) for m in ["HIST-SVM","PLSA-SVM","FK-SVM"]]
        rows.append(["Average",
                     f"{avgs[0]:.2f}%", f"{avgs[1]:.2f}%", f"{avgs[2]:.2f}%"])
        print(_fmt_table(header, rows))

def print_table2(res, pct_labels):
    print(f"\n{'='*65}")
    print("TABLE 2 – Different Training Sample Percentages")
    print(f"{'='*65}")
    for label, key in [("(a) Precision (%)", "acc"), ("(b) Recall Rate (%)", "rec")]:
        print(f"\n  {label}")
        header = ["Train%", "HIST-SVM", "PLSA-SVM", "FK-SVM"]
        rows   = [[p,
                   f"{res['HIST-SVM'][key][i]:.2f}%",
                   f"{res['PLSA-SVM'][key][i]:.2f}%",
                   f"{res['FK-SVM'][key][i]:.2f}%"]
                  for i, p in enumerate(pct_labels)]
        avgs   = [np.mean(res[m][key]) for m in ["HIST-SVM","PLSA-SVM","FK-SVM"]]
        rows.append(["Average",
                     f"{avgs[0]:.2f}%", f"{avgs[1]:.2f}%", f"{avgs[2]:.2f}%"])
        print(_fmt_table(header, rows))

# 9. PLOT GRAFIK
COLORS = {"HIST-SVM":"#1f77b4", "PLSA-SVM":"#ff7f0e", "FK-SVM":"#7f7f7f"}
METHODS = ["HIST-SVM","PLSA-SVM","FK-SVM"]

def _bar_chart(ax, x_labels, data_dict, title, ylim_lo=60):
    x     = np.arange(len(x_labels))
    width = 0.25
    for i, m in enumerate(METHODS):
        ax.bar(x + i*width, data_dict[m], width, label=m,
               color=COLORS[m], edgecolor='white', linewidth=0.5)
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(ylim_lo, 100)
    ax.set_ylabel("Score (%)")
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.35)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))


def plot_figure2(res):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 2. Multi-Round Cross-Validation: FK-SVM vs HIST-SVM vs PLSA-SVM",
                 fontsize=12, fontweight='bold')
    rounds = [f"Round{i+1}" for i in range(N_SPLITS)]
    _bar_chart(axes[0], rounds,
               {m: res[m]["acc"] for m in METHODS}, "(a) Precision", ylim_lo=60)
    _bar_chart(axes[1], rounds,
               {m: res[m]["rec"] for m in METHODS}, "(b) Recall Rate", ylim_lo=60)
    plt.tight_layout()
    plt.savefig("figure2_experiment1.png", dpi=150, bbox_inches='tight')
    print("  Saved: figure2_experiment1.png")
    plt.show()


def plot_figure3(res, pct_labels):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 3. Training Sample % Comparison: FK-SVM vs HIST-SVM vs PLSA-SVM",
                 fontsize=12, fontweight='bold')
    _bar_chart(axes[0], pct_labels,
               {m: res[m]["acc"] for m in METHODS}, "(a) Precision", ylim_lo=50)
    _bar_chart(axes[1], pct_labels,
               {m: res[m]["rec"] for m in METHODS}, "(b) Recall Rate", ylim_lo=50)
    plt.tight_layout()
    plt.savefig("figure3_experiment2.png", dpi=150, bbox_inches='tight')
    print("  Saved: figure3_experiment2.png")
    plt.show()


def plot_flowchart():
    """Figure 1 - sesuai jurnal"""
    fig, ax = plt.subplots(figsize=(5, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 17); ax.axis('off')
    ax.set_title("Figure 1. FK-SVM Sentiment Analysis Flow",
                 fontsize=11, fontweight='bold')

    def box(x, y, w, h, txt, color="#D6E4F0", fs=9):
        r = FancyBboxPatch((x-w/2, y-h/2), w, h,
                           boxstyle="round,pad=0.15", lw=1,
                           edgecolor='#2c3e50', facecolor=color)
        ax.add_patch(r)
        ax.text(x, y, txt, ha='center', va='center',
                fontsize=fs, fontweight='bold', multialignment='center')

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

    box(3.5,16,4,0.85,"Training Data","#AED6F1")
    box(7.5,16,4,0.85,"Test Data","#AED6F1")
    box(3.5,14.3,4,0.85,"Word Segmentation","#D5F5E3")
    box(7.5,14.3,4,0.85,"Word Segmentation","#D5F5E3")
    box(3.5,12.5,4,0.85,"Training PLSA","#FAD7A0")
    box(5.5,10.7,7,0.85,"Topic Feature","#F9E79F")
    box(2.5,8.8,3.5,0.85,"PLSA Model","#FAD7A0")
    box(7,8.8,4,0.85,"Fisher Kernel (FK)","#D2B4DE")
    ax.annotate("", xy=(5,8.8), xytext=(4.3,8.8),
                arrowprops=dict(arrowstyle="->",color="#7f7f7f",lw=1.2,linestyle="dashed"))
    ax.text(4.65,9.05,"derive",fontsize=7,color='gray',ha='center')
    box(5.5,7,5,0.85,"SVM Classifier\n(FK-SVM)","#E8DAEF")
    box(5.5,5.2,5,0.85,"Result Evaluation","#FDEDEC")

    arr(3.5,15.58,3.5,14.73); arr(7.5,15.58,7.5,14.73)
    arr(3.5,13.88,3.5,12.93); arr(3.5,12.08,4.2,11.13)
    arr(7.5,13.88,6.8,11.13)
    arr(2.5,10.25,2.5,9.23);  arr(7,10.25,7,9.23)
    arr(5.5,10.28,5.5,7.43);  arr(5.5,6.58,5.5,5.63)

    plt.tight_layout()
    plt.savefig("figure1_flowchart.png", dpi=150, bbox_inches='tight')
    print("  Saved: figure1_flowchart.png")
    plt.show()

# MAIN
def main():
    print("="*60)
    print(" FK-SVM Sentiment Analysis (Memory-Optimized)")
    print(" Based on: Han et al., Appl. Sci. 2020, 10, 1125")
    print("="*60)

    df = load_dataset(DATASET_PATH, sample_size=SAMPLE_SIZE)

    print("\n2. PREPROCESSING...")
    t0 = time.time()
    df["clean"] = df["text"].apply(preprocess_text)
    df = df[df["clean"].str.len() > 5].reset_index(drop=True)
    texts  = df["clean"].tolist()
    labels = df["label"].tolist()
    print(f"   Selesai ({time.time()-t0:.1f}s). {len(texts):,} dokumen.")

    print("\n  Plotting Figure 1 (flowchart)...")
    plot_flowchart()

    res1 = experiment1(texts, labels)
    print_table1(res1)
    plot_figure2(res1)

    res2, pct_lbl = experiment2(texts, labels)
    print_table2(res2, pct_lbl)
    plot_figure3(res2, pct_lbl)

    print("\n" + "="*60)
    print("SELESAI! File output:")
    print("  figure1_flowchart.png")
    print("  figure2_experiment1.png")
    print("  figure3_experiment2.png")
    print("="*60)


if __name__ == "__main__":
    main()