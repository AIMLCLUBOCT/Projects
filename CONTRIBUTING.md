# Contributing Projects to AIML Club OCT

We welcome project contributions from students of Oriental College of Technology and the global open-source community!

---

## Submission Criteria

Before submitting a project, ensure it satisfies these minimum quality standards:

1. **Clear Directory Structure:**
   ```text
   Projects/[level]/[project-name]/
   ├── README.md              # Project documentation using our standard template
   ├── requirements.txt       # Strict, working dependency specifications
   ├── .gitignore             # Must exclude virtual environments and heavy model weights
   ├── src/                   # Clean, modular Python source code
   └── notebooks/             # (Optional) Cleaned exploratory Jupyter notebooks
   ```
2. **Reproducibility:** Anyone should be able to clone your directory, run `pip install -r requirements.txt`, and execute your training or inference script without missing file errors.
3. **No Large Binaries or Datasets in Git:** Do not commit multi-gigabyte dataset files or heavy model checkpoints (`.pt`, `.ckpt`, `.h5`) directly to Git. Provide download scripts or link to public Kaggle / Hugging Face datasets.
4. **Zero Plagiarism & No Stolen Code:** If you build on existing open-source models or repositories, give explicit credit in your README.
5. **No Secrets or Personal Data:** Verify that `.env` files, API keys, or private student information are NOT present.

---

## 🚀 Fast-Track: Solve a "Good First Issue"
Don't have a project yet? You can still make your first open-source contribution today by solving one of our pre-seeded beginner issues:
- 📌 [**Issue #1: Add interactive Streamlit UI to student-performance-predictor**](https://github.com/AIMLCLUBOCT/Projects/issues/1)
- 📌 [**Issue #2: Add real-time webcam inference loop to safety-helmet-detector**](https://github.com/AIMLCLUBOCT/Projects/issues/2)

---

## 🛠️ Step-by-Step Submission Guide

```bash
# 1. Fork the repo on GitHub, then clone your fork locally:
git clone https://github.com/[YOUR-USERNAME]/Projects.git
cd Projects

# 2. Create a clean feature branch:
git checkout -b feat/my-new-contribution

# 3. Add your code, project folder, or bugfix:
# Follow the structure in templates/PROJECT_TEMPLATE.md

# 4. Stage, commit, and push:
git add .
git commit -m "feat: add [project-name] starter implementation"
git push origin feat/my-new-contribution

# 5. Open your Pull Request on GitHub:
# Navigate to https://github.com/AIMLCLUBOCT/Projects and click "Compare & pull request"
```

Once submitted, club maintainers will review your PR, suggest any optimizations, and merge your work into the official repository! 🌟
