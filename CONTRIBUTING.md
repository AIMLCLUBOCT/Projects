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

## Submission Steps

1. Fork the `Projects` repository.
2. Create a branch: `git checkout -b feat/add-[project-name]`.
3. Add your project under the appropriate directory (`beginner/`, `intermediate/`, `advanced/`, or `research/`).
4. Update the main [`Projects/README.md`](./README.md) table to add your project card.
5. Commit and push your changes.
6. Open a Pull Request referencing your issue or project proposal.
