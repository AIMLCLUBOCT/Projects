<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&custom_color_list=0052CC,8A2BE2,00F5FF&height=230&section=header&text=AIML%20Projects%20Hub&fontSize=46&fontColor=ffffff&animation=fadeIn" alt="Projects Hub Header" width="100%"/>

# 🚀 AIML Club OCT Projects Hub

<img src="https://readme-typing-svg.demolab.com?font=Outfit&weight=600&size=20&duration=3000&pause=1000&color=58A6FF&center=true&vCenter=true&width=750&lines=Production+AI%2FML+Starters+%E2%80%A2+Computer+Vision+%E2%80%A2+NLP+%E2%80%A2+RAG;Zero-Dependency+Execution+Modes+%E2%80%A2+Open+Source+Contributions;Student+Showcases+%E2%80%A2+Applied+Research+%E2%80%A2+Full-Stack+AI" alt="Typing Tagline"/>

<br/><br/>

[![Live Web Portal](https://img.shields.io/badge/Web_Portal-aimlcluboct.github.io-00F5FF?style=for-the-badge&logo=githubpages&logoColor=black)](https://aimlcluboct.github.io/#projects)
[![Live Activities](https://img.shields.io/badge/Live_Activities-Student_Radar-FF6B6B?style=for-the-badge&logo=rss)](https://aimlcluboct.github.io/#activities)
[![AI & Machine Learning Club](https://img.shields.io/badge/AI_%26_ML_Club-OCT_Bhopal-0052CC?style=for-the-badge&logo=googlechrome&logoColor=white)](https://aimlcluboct.in)
[![Projects: Active](https://img.shields.io/badge/Status-Active_Development-brightgreen.svg?style=for-the-badge)](#)
[![Roadmap Board](https://img.shields.io/badge/Roadmap-Live_Kanban_Board-blueviolet.svg?style=for-the-badge&logo=github)](https://github.com/orgs/AIMLCLUBOCT/projects/2)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-orange.svg?style=for-the-badge)](./CONTRIBUTING.md)

</div>

---

> [!IMPORTANT]
> **📢 Live Student Notice & Activity Board:** All active student project sprints, hackathons, and repository releases are broadcast live on our **[Live Activities Radar on aimlcluboct.github.io/#activities ↗](https://aimlcluboct.github.io/#activities)**.

## Overview

The **Projects** repository serves as the central hub for practical software engineering and machine learning implementations developed within the **AI & Machine Learning Club (AIML Club OCT)**, **Oriental College of Technology, Bhopal**.

We emphasize **production readiness**, **clean code architecture**, **reproducible environments**, and **open-source collaboration**. Every project in this repository includes complete source code, dependency specifications, evaluation metrics, and documentation.

---

## 🧭 Project Directory & Levels

Browse projects by difficulty level:

- 🟢 [**Beginner Projects (`beginner/`)**](./beginner/): Tabular ML models, exploratory data analysis apps, and fundamental predictive pipelines designed for 1st/2nd year students.
- 🟡 [**Intermediate Projects (`intermediate/`)**](./intermediate/): Computer vision pipelines, NLP classifiers, deep learning architectures, and FastAPI serving layers.
- 🔴 [**Advanced Projects (`advanced/`)**](./advanced/): Full-stack generative AI applications, agentic workflows, RAG systems, and containerized microservices.
- 🟣 [**Research & Experimental (`research/`)**](./research/): Academic paper replications, benchmark validations, and ablation studies.
- 🧩 [**Project Templates & Submission Guide (`templates/`)**](./templates/): Standard blueprints for submitting your project.

---

## 🌟 Featured Club Initiatives & Projects

### 1. AIML Club OCT Digital Ecosystem & Voice Portal
- **Problem:** Students lack a unified, real-time portal to discover AI workshops, access resources, and submit anonymous feedback or event suggestions to club leadership.
- **Solution:** A modern, distributed web ecosystem featuring real-time updates, automated link redirection, and an interactive feedback portal.
- **Domain:** Web Engineering & Community Automation
- **Tech Stack:** Next.js, React, Tailwind CSS, Vercel, REST APIs
- **Difficulty:** Intermediate
- **Status:** 🟢 Live / Production
- **Live Portals:** [aimlcluboct.in](https://aimlcluboct.in) \| [voice.aimlcluboct.in](https://voice.aimlcluboct.in) \| [social.aimlcluboct.in](https://social.aimlcluboct.in)
- **Documentation:** [Ecosystem Architecture](./advanced/aiml-digital-ecosystem/)

### 2. Tabular Student Performance & Risk Predictor
- **Problem:** Identifying students who require early academic interventions before semester examinations.
- **Solution:** Machine learning pipeline using scikit-learn ensemble models (Random Forests, Logistic Regression) with pure-Python zero-dependency fallback and feature importance ranking.
- **Domain:** Predictive Modeling & Educational Data Mining
- **Tech Stack:** Python, Pandas, Scikit-Learn, Streamlit
- **Difficulty:** Beginner
- **Status:** 🟢 Implemented Starter
- **Code & Docs:** [`beginner/student-performance-predictor`](./beginner/student-performance-predictor/)

### 3. Phishing & Spam Message Classifier
- **Problem:** Detecting fraudulent text, urgent banking scams, and phishing attempts targeted at university students and staff.
- **Solution:** Natural Language Processing (NLP) pipeline comparing Multinomial Naive Bayes and Logistic Regression with TF-IDF n-gram vectorization and pure-Python zero-dependency fallback.
- **Domain:** Natural Language Processing & Cyber AI
- **Tech Stack:** Python, Scikit-Learn, TF-IDF Vectorizer, Multinomial Naive Bayes
- **Difficulty:** Beginner
- **Status:** 🟢 Implemented Starter
- **Code & Docs:** [`beginner/phishing-spam-detector`](./beginner/phishing-spam-detector/)

### 4. Real-Time Safety Equipment & Helmet Detection
- **Problem:** Ensuring workplace and two-wheeler safety compliance on campus grounds.
- **Solution:** Real-time computer vision pipeline utilizing YOLO object detection to identify helmets and safety gear from video streams, with educational IoU simulation.
- **Domain:** Computer Vision & Edge AI
- **Tech Stack:** Python, OpenCV, Ultralytics YOLO, PyTorch
- **Difficulty:** Intermediate
- **Status:** 🟢 Implemented Starter
- **Code & Docs:** [`intermediate/safety-helmet-detector`](./intermediate/safety-helmet-detector/)

### 5. Campus Ordinance & Syllabus RAG Assistant
- **Problem:** Navigating dense 100+ page university ordinances, grading criteria, and semester course catalogs is time-consuming for students.
- **Solution:** Retrieval-Augmented Generation (RAG) assistant that indexes official college PDFs and ordinances into a vector search index to provide verified answers with citations.
- **Domain:** Generative AI & Natural Language Processing
- **Tech Stack:** Python, Cosine Similarity Vector Index, LangChain / ChromaDB
- **Difficulty:** Advanced
- **Status:** 🟢 Implemented Starter
- **Code & Docs:** [`advanced/campus-rag-assistant`](./advanced/campus-rag-assistant/)

---

## 📋 Standard Project Card Format

When submitting or documenting projects, contributors must use this standard specification:

```markdown
### [Project Name]
- **Problem:** [1-sentence problem statement]
- **Solution:** [1-2 sentences explaining the technical solution]
- **Domain:** [e.g. Computer Vision / NLP / Tabular ML / GenAI / MLOps]
- **Tech Stack:** [e.g. Python, PyTorch, OpenCV, FastAPI]
- **Difficulty:** [Beginner / Intermediate / Advanced / Research]
- **Status:** [Proposed / In Development / Completed / Maintained]
- **Repository:** [Direct link to project directory or submodule]
- **Demo:** [Live URL or Colab link if applicable]
- **Documentation:** [Link to local README.md]
```

---

## 🛠️ How to Submit Your Project

Have you built an interesting AI/ML project during a college hackathon, club workshop, or self-study? We would love to feature it!

1. Check our [**Contributing Guide**](./CONTRIBUTING.md).
2. Use the [**Project Blueprint Template**](./templates/PROJECT_TEMPLATE.md).
3. Ensure your project directory contains a clean `README.md`, `requirements.txt`, and clear setup instructions.
4. Open a Pull Request for review by the technical maintainers.

### 🎯 Active Starter Tasks (Good First Issues):
Looking for something concrete to work on? Grab one of our open contributor tasks:
- 🚀 **[Issue #7: Streamlit Web UI for Phishing & Spam Detector](https://github.com/AIMLCLUBOCT/Projects/issues/7)** (Beginner NLP & Web)
- 📊 **[Issue #8: Student Feedback Sentiment Analyzer](https://github.com/AIMLCLUBOCT/Projects/issues/8)** (Beginner Text Analytics & EDA)
- 📈 **[Issue #8 in learning_resources: Decision Boundary Visualization Tool](https://github.com/AIMLCLUBOCT/learning_resources/issues/8)** (Matplotlib & Classification)

> 💬 **Want feedback first?** You can share your project demo, link, or prototype directly in our [**Student Project Showcase Discussion Panel**](https://github.com/AIMLCLUBOCT/Projects/discussions/3) or [**Club-wide Showcase**](https://github.com/AIMLCLUBOCT/learning_resources/discussions/9) to get feedback from club seniors!
>
> 🌐 **Frontier Engineering Discussion:** Brainstorm architectures on our [**Autonomous Multi-Agent Swarms & Local Edge AI Forum ↗**](https://github.com/AIMLCLUBOCT/Projects/discussions/4)!
>
> 💼 **Career & Portfolio Advice:** Learn how to present your GitHub projects on resumes in our [**Career & Portfolio Hub ↗**](https://github.com/AIMLCLUBOCT/learning_resources/discussions/10)!

---

## 🌐 Connect with the Community

- **Official Website:** [aimlcluboct.in](https://aimlcluboct.in)
- **Digital Hub & Socials:** [social.aimlcluboct.in](https://social.aimlcluboct.in)
- **Share Ideas & Feedback:** [voice.aimlcluboct.in](https://voice.aimlcluboct.in)
- **Email:** [aimlcluboct@gmail.com](mailto:aimlcluboct@gmail.com)

<br/>
<div align="center">
<sub>© 2026 AI & Machine Learning Club – Oriental College of Technology, Bhopal.</sub><br/><br/>
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&custom_color_list=0052CC,8A2BE2,00F5FF&height=100&section=footer" width="100%"/>
</div>