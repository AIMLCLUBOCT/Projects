# Student Feedback Sentiment Analyzer (v2)

Classifies course-feedback comments as **Positive / Neutral / Negative**, compares
several models, evaluates them honestly, and tells you **which aspect** of the course
each opinion is about (lectures, labs, assignments, instructor, exams, materials, pace).

```
"The lab demonstrations were very helpful, but the assignment deadlines were too tight."
  Overall: Neutral (66%)  [mixed opinions]
    - labs         Positive  (94%)
    - assignments  Negative  (90%)
```

## Quick start

```bash
pip install -r requirements.txt

python train_and_evaluate.py                       # train + compare + evaluate on synthetic data
python train_and_evaluate.py --data feedback.csv   # train on your own data
python train_and_evaluate.py --tune                # grid-search Logistic Regression first
python train_and_evaluate.py --save model.joblib --metrics-out metrics.json

python train_and_evaluate.py --load model.joblib --predict "Labs were great but exams were unfair"
python train_and_evaluate.py --load model.joblib --interactive
```

> Load saved models by running **this same script** (the model references its preprocessing
> functions, so it can only be unpickled where they are defined).

### Using your own data

CSV with two columns:

| text | label |
|---|---|
| The labs were really useful | Positive |
| Deadlines were too tight | Negative |
| Classes are twice a week | Neutral |

Labels are case-insensitive. With `--data`, the script does a stratified 80/20 split.
**Real data is strongly recommended**: the built-in synthetic data is for demos only.

## What changed from v1

| Area | v1 | v2 | Why |
|---|---|---|---|
| Evaluation | Random split of template data (scores ~100%) | 3 test sets: in-distribution, **unseen phrasing**, independent **handwritten** gold set | Template data leaks across splits, so the old score was misleading |
| Handwritten data | Mixed into training (and used again as demo samples) | Test-only | No train/test contamination |
| Negation | Contractions expanded only | Negation marking: `not helpful` becomes `not_helpful` (scope ends at punctuation / "but") | Bag-of-words can't tell "helpful" from "not helpful" |
| Ensemble | Hard voting with 2 voters (ties possible) | Soft voting with 3 voters (LR + calibrated SVC + Complement NB) | Averages probabilities, no ties |
| Probabilities | LinearSVC gave labels only | SVC calibrated; every model has `predict_proba` | Confidence scores and low-confidence flags |
| Aspects | Keyword list, no sentiment per aspect | **Clause-level ABSA**: splits on "but/although/however/and", classifies each clause | Mixed comments get per-aspect sentiment |
| Aspect matching | Exact words only | Light plural stemming, more keywords | Fewer misses (`labs`, `slides`, `deadline`) |
| Tuning | None | Optional `--tune` (GridSearchCV on n-grams and C) | Better hyperparameters |
| Interpretability | Frequency counts | Plus top-weighted words per class from Logistic Regression | Shows what the model actually learned |
| Persistence | Retrain every run | `--save` / `--load`, `--predict`, `--metrics-out` (JSON) | Reuse and track results |
| Data input | Synthetic only | `--data your.csv` | Real-world use |

## Architecture

```
text
 |-- preprocess: lowercase, expand contractions, strip URLs
 |      |-- word path: + negation marking
 |      '-- char path: letters only
 v
FeatureUnion
 |-- TF-IDF word 1-2 grams (negation-aware)
 '-- TF-IDF char_wb 3-5 grams (robust to typos)
 v
Classifier: LogisticRegression | calibrated LinearSVC | ComplementNB | soft-voting ensemble
 v
Model selection by 5-fold CV macro-F1  ->  evaluate on 3 test sets
 v
ABSA: split text into clauses -> classify each clause -> map to aspects -> overall + per-aspect result
```

## Reading the results

Example run (default settings):

| Model | in_distribution | unseen_phrasing | handwritten |
|---|---|---|---|
| Logistic Regression | 1.00 | 0.79 | 0.94 |
| Linear SVC (calibrated) | 1.00 | 0.80 | 0.94 |
| Complement NB | 0.91 | 0.79 | 0.88 |
| Ensemble (soft voting) | 1.00 | 0.80 | 0.94 |

- **in_distribution** is optimistic: test sentences are built from the same phrases as training.
- **unseen_phrasing** uses phrases never seen in training. This is the number that reflects generalization.
- **handwritten** is only 16 sentences, so it is noisy. Treat it as a sanity check.
- CV on synthetic data is also optimistic for the same reason (folds share templates), which
  is why several models tie at 1.000. On real data, CV is trustworthy.

## Known limitations

- **Vocabulary is limited by the synthetic data.** Words never seen in training (e.g. "great", "dull")
  carry no signal, so predictions on them are weak and often low-confidence. Training on real
  comments is the single biggest improvement you can make.
- ABSA uses keyword aspect detection and a clause classifier trained on whole comments, so
  it is a heuristic. Sarcasm and long multi-sentence comments will be misread.
- Low-confidence predictions (< 50%) are flagged; consider routing those to manual review.

## Next steps

1. Collect 500+ real, labeled comments and train with `--data`.
2. Add per-aspect labels to your data so aspect sentiment can be trained directly.
3. Try a pretrained transformer (e.g. DistilBERT / `sentence-transformers` embeddings + Logistic Regression) as an extra model in `build_models()`.