"""
Student Feedback Sentiment Analyzer 
Classifies course-feedback comments into Positive / Neutral / Negative,
compares several models, evaluates them honestly (including on phrasing the
model never saw), and performs aspect-based sentiment analysis (ABSA):
"labs were great but deadlines were tight" -> labs: Positive, assignments: Negative.

Usage:
    python train_and_evaluate.py                          # train, compare, evaluate (synthetic data)
    python train_and_evaluate.py --data feedback.csv      # use your own CSV (columns: text,label)
    python train_and_evaluate.py --save model.joblib      # save the best model
    python train_and_evaluate.py --load model.joblib --predict "Labs were great but exams were unfair"
    python train_and_evaluate.py --load model.joblib --interactive
    python train_and_evaluate.py --tune                   # grid-search Logistic Regression first
"""

import argparse
import json
import random
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
LABELS = ["Negative", "Neutral", "Positive"]
LOW_CONFIDENCE = 0.50  # below this max-probability, a prediction is flagged

# --------------------------------------------------------------------------
# 1. TEXT PREPROCESSING (with negation handling)
# --------------------------------------------------------------------------
CONTRACTIONS = {"can't": "cannot", "won't": "will not", "shan't": "shall not"}
NEGATORS = {"not", "no", "never", "cannot", "hardly", "without"}
CLAUSE_BREAKS = {"but", "although", "however", "though", "yet", "whereas"}
NEG_SCOPE = 2  # how many content words a negator flips ("not very helpful" -> not_helpful)


def expand_contractions(text: str) -> str:
    text = text.replace("\u2019", "'")  # curly apostrophe
    for k, v in CONTRACTIONS.items():
        text = re.sub(rf"\b{k}\b", v, text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"\b(it|that|there|what)'s\b", r"\1 is", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'ve\b", " have", text)
    text = re.sub(r"'ll\b", " will", text)
    text = re.sub(r"'m\b", " am", text)
    return text


def _base(text: str) -> str:
    text = expand_contractions(text.lower())
    return re.sub(r"http\S+", " ", text)


def normalize(text: str) -> str:
    """Lowercase, expand contractions, keep letters only (used for char n-grams)."""
    return " ".join(re.findall(r"[a-z]+", _base(text)))


def mark_negation(text: str) -> str:
    """'was not very helpful' -> 'was not very not_helpful' (used for word n-grams).

    The negation scope ends at punctuation or a contrast word such as 'but'.
    Stop words don't consume scope, so 'not very helpful' still flips 'helpful'.
    """
    out, scope = [], 0
    for tok in re.findall(r"[a-z]+|[.,;:!?]", _base(text)):
        if tok in ".,;:!?" or tok in CLAUSE_BREAKS:
            scope = 0
            if tok not in ".,;:!?":
                out.append(tok)
            continue
        if tok in NEGATORS:
            out.append(tok)
            scope = NEG_SCOPE
        elif scope > 0 and tok not in ENGLISH_STOP_WORDS:
            out.append("not_" + tok)
            scope -= 1
        else:
            out.append(tok)
    return " ".join(out)


# --------------------------------------------------------------------------
# 2. DATA: HANDWRITTEN GOLD SET + TEMPLATE-BASED SYNTHETIC DATA
# --------------------------------------------------------------------------
# NOTE: the handwritten set is used ONLY for testing (never for training), so it
# acts as a small independent gold set.
HANDWRITTEN = [
    ("The lab demonstrations were very helpful and made concepts clear.", "Positive"),
    ("Professor explains difficult topics with great examples. Loved this course!", "Positive"),
    ("Excellent course, I learned a lot and the instructor was always approachable.", "Positive"),
    ("The group projects were engaging and taught me real teamwork skills.", "Positive"),
    ("Lecture slides were well organised and the pace was perfect.", "Positive"),
    ("The assignment deadlines were far too tight and there was no feedback.", "Negative"),
    ("Lectures were boring and the professor just read from the slides.", "Negative"),
    ("Exam questions had nothing to do with what was taught. Very unfair.", "Negative"),
    ("The course was disorganised and the instructor rarely replied to emails.", "Negative"),
    ("Too much content crammed into too little time, I felt lost most weeks.", "Negative"),
    ("The lab demonstrations were very helpful, but the assignment deadlines were too tight.", "Neutral"),
    ("Good content overall, although the pace was too fast at times.", "Neutral"),
    ("The course was okay. Some lectures were useful, others were not.", "Neutral"),
    ("Average course. Nothing special but nothing terrible either.", "Neutral"),
    ("The instructor is knowledgeable but the slides could be improved.", "Neutral"),
    ("The course covers the syllabus. Classes are held twice a week.", "Neutral"),
]

ASPECTS = {
    "lectures": {
        "Positive": ["the lectures were clear and engaging", "lectures were well structured and easy to follow",
                     "I enjoyed every lecture", "the lecture examples were really helpful"],
        "Negative": ["the lectures were boring and hard to follow", "lectures felt disorganised and rushed",
                     "I could not stay focused during the lectures", "the lectures were just slides read aloud"],
        "Neutral": ["the lectures were held twice a week", "lectures covered the topics in the syllabus",
                    "the lectures were average", "lectures were fine, nothing special"],
    },
    "labs": {
        "Positive": ["the lab sessions were very helpful", "hands-on labs made the concepts click",
                     "the lab demonstrations were excellent", "lab work was practical and well guided"],
        "Negative": ["the lab sessions were confusing and poorly supervised", "lab equipment often did not work",
                     "the labs were a waste of time", "lab instructions were unclear"],
        "Neutral": ["the labs were scheduled on Fridays", "lab sessions were okay", "the lab covered basic exercises",
                    "labs were neither good nor bad"],
    },
    "assignments": {
        "Positive": ["the assignments helped me understand the material", "assignments were fair and well designed",
                     "I got useful feedback on every assignment", "the assignment difficulty was just right"],
        "Negative": ["the assignment deadlines were too tight", "assignments were unclear and overloaded",
                     "we got no feedback on assignments", "the workload from assignments was overwhelming"],
        "Neutral": ["there were four assignments in total", "assignments were submitted online",
                    "the assignments were average", "assignment difficulty varied"],
    },
    "instructor": {
        "Positive": ["the professor was approachable and supportive", "the instructor explains very well",
                     "the teacher was passionate and knowledgeable", "the professor answered every question patiently"],
        "Negative": ["the professor was unapproachable and dismissive", "the instructor rarely replied to questions",
                     "the teacher was often late and unprepared", "the professor explained things poorly"],
        "Neutral": ["the instructor was fine", "the professor held office hours on Tuesdays",
                    "the teacher was knowledgeable but quiet", "the professor was okay"],
    },
    "exams": {
        "Positive": ["the exams were fair and matched the syllabus", "exam preparation was well supported",
                     "the exam format was clear and reasonable"],
        "Negative": ["the exams were unfair and too difficult", "exam questions were not covered in class",
                     "the exam time was far too short"],
        "Neutral": ["there was a midterm and a final", "the exams were moderately difficult",
                    "exams were held in the main hall"],
    },
    "materials": {
        "Positive": ["the course materials were excellent", "the slides and notes were very clear",
                     "the reading list was useful and relevant"],
        "Negative": ["the course materials were outdated", "the slides were confusing and full of errors",
                     "the reading list was irrelevant"],
        "Neutral": ["the materials were shared on the portal", "the slides were average",
                    "the reading list had ten books"],
    },
    "pace": {
        "Positive": ["the pace of the course was perfect", "the course moved at a comfortable speed"],
        "Negative": ["the pace was far too fast", "too much content was crammed into each class"],
        "Neutral": ["the pace was moderate", "the course pace varied by week"],
    },
}

OPENERS = ["", "Overall, ", "Honestly, ", "In my opinion, ", "To be fair, "]
CLOSERS = [".", ".", "!", " overall.", " this semester."]
MIXERS = [", but ", ", although ", ", however ", ". On the other hand, "]


def split_templates(holdout_frac=0.25):
    """Split every aspect/sentiment phrase list into train phrases and held-out phrases.

    Held-out phrases are NEVER used to generate training data, so scoring on them
    measures generalisation to unseen wording instead of template memorisation.
    """
    train, held = {}, {}
    for aspect, by_sent in ASPECTS.items():
        train[aspect], held[aspect] = {}, {}
        for sent, phrases in by_sent.items():
            n_hold = max(1, round(len(phrases) * holdout_frac))
            train[aspect][sent] = phrases[:-n_hold]
            held[aspect][sent] = phrases[-n_hold:]
    return train, held


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


def generate_synthetic(pool, n_per_class=300, seed=SEED):
    rng = random.Random(seed)
    aspects = list(pool)
    rows, seen = [], set()

    def pick(sent):
        return rng.choice(pool[rng.choice(aspects)][sent])

    for label in LABELS:
        count, attempts = 0, 0
        while count < n_per_class and attempts < n_per_class * 200:
            attempts += 1
            if label in ("Positive", "Negative"):
                c1 = pick(label)
                text = rng.choice(OPENERS) + c1
                if rng.random() < 0.5:
                    c2 = pick(label)
                    if c2 != c1:
                        text += " and " + c2
            elif rng.random() < 0.5:  # mixed opinion -> Neutral
                pos, neg = pick("Positive"), pick("Negative")
                first, second = (pos, neg) if rng.random() < 0.5 else (neg, pos)
                text = rng.choice(OPENERS) + first + rng.choice(MIXERS) + second
            else:
                c1 = pick("Neutral")
                text = rng.choice(OPENERS) + c1
                if rng.random() < 0.4:
                    c2 = pick("Neutral")
                    if c2 != c1:
                        text += " and " + c2
            text = _cap(text.strip()) + rng.choice(CLOSERS)
            if text not in seen:
                seen.add(text)
                rows.append((text, label))
                count += 1
    return rows


def _frame(rows):
    df = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates("text")
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_datasets(args):
    """Returns (X_train, y_train, {test_name: (X, y)}, full_df_for_theme_extraction)."""
    if args.data:
        df = pd.read_csv(args.data)
        if not {"text", "label"} <= set(df.columns):
            raise SystemExit("CSV must have 'text' and 'label' columns.")
        df = df.dropna(subset=["text", "label"]).drop_duplicates("text")
        df["label"] = df["label"].astype(str).str.strip().str.title()
        bad = set(df["label"]) - set(LABELS)
        if bad:
            raise SystemExit(f"Unknown labels {bad}; expected {LABELS}.")
        X_tr, X_te, y_tr, y_te = train_test_split(
            df.text, df.label, test_size=0.2, stratify=df.label, random_state=SEED)
        return X_tr, y_tr, {"holdout": (X_te, y_te)}, df

    train_pool, held_pool = split_templates()
    df = _frame(generate_synthetic(train_pool, args.n_per_class))
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.text, df.label, test_size=0.2, stratify=df.label, random_state=SEED)
    unseen = _frame(generate_synthetic(held_pool, max(60, args.n_per_class // 3), SEED + 1))
    gold = _frame(HANDWRITTEN)
    tests = {
        "in_distribution": (X_te, y_te),          # same phrase pool as training (optimistic)
        "unseen_phrasing": (unseen.text, unseen.label),  # phrases never seen in training
        "handwritten": (gold.text, gold.label),   # independent human-written gold set
    }
    return X_tr, y_tr, tests, df


# --------------------------------------------------------------------------
# 3. MODELS
# --------------------------------------------------------------------------
def get_feature_extractor():
    """Word n-grams (negation-aware) + character n-grams (robust to typos)."""
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(preprocessor=mark_negation, ngram_range=(1, 2),
                                       min_df=2, sublinear_tf=True)),
        ("char_tfidf", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb",
                                       ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])


def _lr():
    return LogisticRegression(max_iter=1000, C=2.0, class_weight="balanced", random_state=SEED)


def _calibrated_svc():
    # LinearSVC has no predict_proba; calibration gives real probabilities
    # (needed for confidence scores and soft voting).
    return CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced", random_state=SEED), cv=3)


def build_models():
    return {
        "Logistic Regression": Pipeline([("features", get_feature_extractor()), ("clf", _lr())]),
        "Linear SVC (calibrated)": Pipeline([("features", get_feature_extractor()), ("clf", _calibrated_svc())]),
        "Complement NB": Pipeline([
            ("features", TfidfVectorizer(preprocessor=mark_negation, ngram_range=(1, 2),
                                         min_df=2, sublinear_tf=True)),
            ("clf", ComplementNB(alpha=0.5)),
        ]),
        # Soft voting averages probabilities, so 3 voters can't produce the
        # 1-1 ties the old 2-voter hard-voting ensemble could.
        "Ensemble (soft voting)": Pipeline([
            ("features", get_feature_extractor()),
            ("clf", VotingClassifier(
                estimators=[("lr", _lr()), ("svc", _calibrated_svc()),
                            ("cnb", ComplementNB(alpha=0.5))],
                voting="soft")),
        ]),
    }


def tune_logreg(model, X, y):
    grid = {
        "features__word_tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "clf__C": [0.5, 2.0, 8.0],
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(model, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
    gs.fit(X, y)
    print(f"  Best LR params: {gs.best_params_} (CV macro-F1 {gs.best_score_:.3f})")
    return gs.best_estimator_


def compare_models(models, X, y, folds=5):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    results = {}
    for name, model in models.items():
        s = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        results[name] = (s.mean(), s.std())
        print(f"  {name:<26} CV macro-F1 = {s.mean():.3f} (+/- {s.std():.3f})")
    return results


# --------------------------------------------------------------------------
# 4. EVALUATION
# --------------------------------------------------------------------------
def evaluate(model, X, y, verbose=False, title=""):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro", labels=LABELS)
    if verbose:
        print(f"\n--- {title} ---")
        print(classification_report(y, pred, labels=LABELS, digits=3, zero_division=0))
        cm = pd.DataFrame(confusion_matrix(y, pred, labels=LABELS),
                          index=[f"true_{l}" for l in LABELS],
                          columns=[f"pred_{l}" for l in LABELS])
        print(cm)
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}


def top_model_features(model, k=8):
    """Most influential word features per class (Logistic Regression only)."""
    feats = model.named_steps["features"]
    names = np.array(feats.get_feature_names_out())
    keep = np.array([n.startswith("word_tfidf__") for n in names])
    clf = model.named_steps["clf"]
    print("\n=== Most Influential Words (Logistic Regression) ===")
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i][keep]
        words = [w.replace("word_tfidf__", "") for w in names[keep][np.argsort(coefs)[::-1][:k]]]
        print(f"{cls:<9}", ", ".join(words))


EXTRA_STOP = {"course", "overall", "honestly", "opinion", "fair", "semester", "hand", "really", "just"}
KEEP = {"not", "no", "too", "very"}
STOP = (set(ENGLISH_STOP_WORDS) - KEEP) | EXTRA_STOP


def print_themes(df, k=8):
    print("\n=== Most Frequent Phrases Per Class ===")
    for label in LABELS:
        texts = df.loc[df.label == label, "text"]
        cv = CountVectorizer(preprocessor=normalize, ngram_range=(1, 2),
                             stop_words=list(STOP), min_df=3)
        counts = cv.fit_transform(texts).sum(axis=0).A1
        terms = cv.get_feature_names_out()
        top = [(terms[i], int(counts[i])) for i in np.argsort(counts)[::-1][:k]]
        print(f"{label:<9}", ", ".join(f"{t} ({c})" for t, c in top))


# --------------------------------------------------------------------------
# 5. ASPECT-BASED SENTIMENT ANALYSIS (clause-level)
# --------------------------------------------------------------------------
ASPECT_KEYWORDS = {
    "lectures": {"lecture", "class"},
    "labs": {"lab", "practical"},
    "assignments": {"assignment", "deadline", "workload", "homework", "project"},
    "instructor": {"professor", "instructor", "teacher", "lecturer"},
    "exams": {"exam", "midterm", "final", "test", "quiz"},
    "materials": {"slide", "material", "reading", "note", "textbook"},
    "pace": {"pace", "fast", "speed", "rushed", "crammed"},
}


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def detect_aspects(text):
    words = {_stem(w) for w in normalize(text).split()}
    return [a for a, kws in ASPECT_KEYWORDS.items() if words & kws]


def split_clauses(text):
    """Split into opinion clauses on sentence ends / contrast words, then on 'and'
    when both halves mention an aspect ("lectures were dull and labs were great")."""
    parts = re.split(r"[.;!?]+|\s*,?\s*\b(?:but|although|however|though|whereas|yet)\b\s*,?",
                     text, flags=re.I)
    clauses = []
    for p in (p.strip() for p in parts if p and p.strip()):
        halves = [h.strip() for h in re.split(r"\s+and\s+", p, flags=re.I) if h.strip()]
        if len(halves) > 1 and all(detect_aspects(h) for h in halves):
            clauses.extend(halves)
        else:
            clauses.append(p)
    return clauses


def predict_with_confidence(model, text):
    proba = model.predict_proba([text])[0]
    i = int(np.argmax(proba))
    return model.classes_[i], float(proba[i])


def analyze(model, text):
    label, conf = predict_with_confidence(model, text)
    per_aspect = {}
    for clause in split_clauses(text):
        for a in detect_aspects(clause):
            c_label, c_conf = predict_with_confidence(model, clause)
            if a not in per_aspect or c_conf > per_aspect[a]["confidence"]:
                per_aspect[a] = {"sentiment": c_label, "confidence": round(c_conf, 3)}
    sentiments = {v["sentiment"] for v in per_aspect.values()}
    return {
        "overall": label,
        "confidence": round(conf, 3),
        "low_confidence": conf < LOW_CONFIDENCE,
        "mixed": {"Positive", "Negative"} <= sentiments,
        "aspects": per_aspect,
    }


def format_analysis(r):
    flag = "  (low confidence)" if r["low_confidence"] else ""
    mixed = "  [mixed opinions]" if r["mixed"] else ""
    lines = [f"  Overall: {r['overall']} ({r['confidence']:.0%}){flag}{mixed}"]
    for a, v in r["aspects"].items():
        lines.append(f"    - {a:<12} {v['sentiment']:<9} ({v['confidence']:.0%})")
    if not r["aspects"]:
        lines.append("    - (no specific aspect detected)")
    return "\n".join(lines)


def interactive_loop(model):
    print("\n[Interactive Mode] Type a comment (or 'q' to exit).")
    while True:
        text = input("> ").strip()
        if text.lower() in {"q", "quit", "exit", ""}:
            break
        print(format_analysis(analyze(model, text)))


# --------------------------------------------------------------------------
# 6. MAIN
# --------------------------------------------------------------------------
SAMPLES = [
    "The lab demonstrations were very helpful, but the assignment deadlines were too tight.",
    "Best professor I have had, truly inspiring!",
    "The exams were unfair and the slides were outdated.",
    "Lectures are held on Monday and Wednesday.",
    "The lectures weren't engaging at all and the labs were not helpful.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="CSV with 'text' and 'label' columns (Positive/Neutral/Negative)")
    ap.add_argument("--n-per-class", type=int, default=300, help="synthetic comments per class")
    ap.add_argument("--tune", action="store_true", help="grid-search Logistic Regression first")
    ap.add_argument("--save", help="save the best model to this .joblib path")
    ap.add_argument("--load", help="load a saved model instead of training")
    ap.add_argument("--predict", help="analyze one comment and exit")
    ap.add_argument("--metrics-out", help="write evaluation metrics to this JSON path")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    # ---- inference-only path -------------------------------------------
    if args.load:
        bundle = joblib.load(args.load)
        model = bundle["model"]
        print(f"Loaded model: {bundle['name']}")
        if args.predict:
            print(format_analysis(analyze(model, args.predict)))
        if args.interactive or not args.predict:
            interactive_loop(model)
        return

    # ---- training path --------------------------------------------------
    X_train, y_train, tests, df = build_datasets(args)
    print(f"Training comments: {len(X_train)} | classes: {y_train.value_counts().to_dict()}")
    print(f"Test sets: {', '.join(f'{k} (n={len(v[0])})' for k, v in tests.items())}\n")

    models = build_models()
    if args.tune:
        print("Tuning Logistic Regression...")
        models["Logistic Regression"] = tune_logreg(models["Logistic Regression"], X_train, y_train)

    print("Cross-validation on training data:")
    cv_scores = compare_models(models, X_train, y_train)
    best_name = max(cv_scores, key=lambda n: cv_scores[n][0])
    print(f"\nSelected by CV macro-F1: {best_name}")
    if not args.data:
        print("  (CV on synthetic data is optimistic: folds share phrase templates."
              " Trust 'unseen_phrasing' and 'handwritten' more.)")

    summary, metrics = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics[name] = {t: evaluate(model, X, y) for t, (X, y) in tests.items()}
        summary[name] = {t: m["macro_f1"] for t, m in metrics[name].items()}
        summary[name]["cv"] = round(cv_scores[name][0], 3)
    print("\n=== Macro-F1 by model and test set ===")
    print(pd.DataFrame(summary).T.to_string())

    best = models[best_name]
    for t, (X, y) in tests.items():
        evaluate(best, X, y, verbose=True, title=f"{best_name} on '{t}'")

    top_model_features(models["Logistic Regression"])
    print_themes(df)

    print("\n=== Aspect-Based Sample Predictions ===")
    for s in SAMPLES:
        print(f"\n\"{s}\"\n{format_analysis(analyze(best, s))}")

    if args.save:
        joblib.dump({"model": best, "name": best_name, "labels": LABELS}, args.save)
        print(f"\nSaved best model -> {args.save}")
    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump({"cv_macro_f1": {n: round(v[0], 4) for n, v in cv_scores.items()},
                       "selected": best_name, "test_metrics": metrics}, f, indent=2)
        print(f"Saved metrics -> {args.metrics_out}")

    if args.interactive:
        interactive_loop(best)


if __name__ == "__main__":
    main()