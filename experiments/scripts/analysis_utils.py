from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple


def map_wildguard_label_to_binary(label: Any) -> Optional[int]:
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in {"harmful", "unsafe", "1", "true", "yes"}:
        return 1
    if s in {"unharmful", "safe", "0", "false", "no"}:
        return 0
    return None


@dataclass
class BinaryMetrics:
    n: int
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> BinaryMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    tp = tn = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1

    return BinaryMetrics(n=len(y_true), tp=tp, tn=tn, fp=fp, fn=fn)


def save_confusion_matrix_plot(cm: Tuple[int, int, int, int], out_path: Path, title: str) -> None:
    tp, tn, fp, fn = cm
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return

    data = [[tn, fp], [fn, tp]]
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    sns.heatmap(data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_xticklabels(["safe(0)", "unsafe(1)"])
    ax.set_yticklabels(["safe(0)", "unsafe(1)"], rotation=0)
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_threat_category_bar(categories: Iterable[Any], out_path: Path, title: str, top_k: int = 20) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    # Flatten list-like categories
    flat: List[str] = []
    for c in categories:
        if c is None:
            continue
        if isinstance(c, list):
            flat.extend([str(x) for x in c if x is not None])
        else:
            flat.append(str(c))

    counts = Counter(flat)
    most = counts.most_common(top_k)
    if not most:
        return

    labels = [k for k, _ in most]
    values = [v for _, v in most]

    fig = plt.figure(figsize=(8, 3.5))
    ax = fig.add_subplot(111)
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
