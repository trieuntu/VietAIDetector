"""Shared provenance, text checks and frozen evaluation settings."""

import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

SEED = 20261108
THRESHOLD = 0.9279661178588868
CONFIGURATIONS = ((650, 150), (600, 0))
GENERATORS = ("claude-sonnet-4.6", "gemini-3.7-flash", "gpt-5.6-luna")
TOPICS = {
    "kinh_te": ("kinh tế", "kinh doanh", "tài chính - kinh doanh"),
    "van_hoa": ("văn hóa", "văn hoá", "văn hóa - giải trí"),
    "chinh_tri": ("chính trị", "ngoại giao"),
    "xa_hoi": ("xã hội", "giáo dục", "sức khỏe", "đời sống"),
    "the_thao": ("thể thao", "bóng đá", "bóng đá việt nam", "ngoại hạng anh",
                 "bóng đá thế giới", "bóng đá anh", "la liga", "bundesliga",
                 "serie a", "ligue 1", "tennis", "v-league", "world cup 2022",
                 "hậu trường bóng đá", "bóng đá & cuộc sống"),
}
LENGTH_BINS = ((2048, 2560), (2560, 4096), (4096, 8192))
CODE_FILES = (
    "preprocessing/normalizer.py", "preprocessing/text_utils.py",
    "processing/chunker.py", "core/scorer.py", "core/metrics.py",
)
BOILERPLATE = re.compile(
    r"giấy phép (?:thiết lập|số)|bản quyền thuộc về|"
    r"trang thông tin điện tử tổng hợp|chịu trách nhiệm nội dung|"
    r"tổng biên tập\s*:|thông tin tòa soạn|<\s*(?:script|iframe|html)\b|https?://|>>>",
    re.IGNORECASE,
)


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty manifest")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def text_normalizer(repo):
    sys.path.insert(0, str(Path(repo).resolve()))
    from preprocessing.normalizer import TextNormalizer
    from preprocessing.text_utils import DEHYPHEN_RE
    normalizer = TextNormalizer()

    def clean(text):
        return "\n".join(normalizer.normalize(p) for p in
                         normalizer.filter_paragraphs(DEHYPHEN_RE.sub(r"\1\2", text)))
    return clean


def length_bin(n):
    for lower, upper in LENGTH_BINS:
        if lower <= n < upper:
            return f"[{lower},{upper})"
    raise ValueError(f"Document length outside frozen range: {n}")


def shingles(text):
    words = re.findall(r"\w+", canonical(text))
    return {hashlib.blake2b(" ".join(words[i:i + 5]).encode(), digest_size=8).digest()
            for i in range(len(words) - 4)}


class OverlapIndex:
    def __init__(self, documents):
        self.sets = [shingles(d) for d in documents]
        self.postings = defaultdict(list)
        for i, values in enumerate(self.sets):
            for value in values:
                self.postings[value].append(i)

    def matches(self, text):
        values = shingles(text)
        hits = Counter(i for value in values for i in self.postings.get(value, ()))
        result = []
        for i, intersection in hits.items():
            smaller = min(len(values), len(self.sets[i]))
            if smaller < 25:
                continue
            containment = intersection / smaller
            jaccard = intersection / (len(values) + len(self.sets[i]) - intersection)
            if containment >= 0.8 or jaccard >= 0.8:
                result.append((i, round(jaccard, 8), round(containment, 8)))
        return result


def reference_texts(row):
    for key, value in row.items():
        if isinstance(value, str) and (key in ("text", "user_prompt") or "generated_text" in key):
            yield key, value


def metrics(rows):
    from sklearn.metrics import roc_auc_score
    counts = Counter((r["label"], r["prediction"]) for r in rows)
    tn, fp, fn, tp = (counts[k] for k in ((0, 0), (0, 1), (1, 0), (1, 1)))
    result = dict(n=len(rows), human=tn + fp, ai=tp + fn, tn=tn, fp=fp, fn=fn, tp=tp,
                  ai_recall=tp / (tp + fn) if tp + fn else None,
                  human_fpr=fp / (tn + fp) if tn + fp else None)
    for name, successes, total in (("ai_recall", tp, tp + fn), ("human_fpr", fp, tn + fp)):
        result[f"{name}_ci95_low"], result[f"{name}_ci95_high"] = wilson(successes, total)
    result.update(macro_f1=None, auroc_vote=None, auroc_mean=None)
    if tn + fp and tp + fn:
        result["macro_f1"] = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2
        for key in ("vote", "mean"):
            result[f"auroc_{key}"] = float(roc_auc_score(
                [r["label"] for r in rows], [r[f"score_{key}"] for r in rows]))
    return result


def wilson(successes, total):
    if not total:
        return None, None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)
