# SkillMatch

SkillMatch is a semantic job search system built on LinkedIn job postings.

The system allows users to describe their desired job in natural language (skills, experience, technologies and preferences) and retrieves relevant vacancies even when the wording of the query does not exactly match the job title.

For example, the query:

> Junior data analyst with Python, SQL and Tableau, preferably remote

may retrieve positions such as:

- Junior Data Analyst
- Business Intelligence Analyst
- Reporting Analyst
- Analytics Associate
- Data Specialist

---

## Project Goal

This project compares lexical and dense retrieval methods for semantic job search.

The goal is to evaluate different retrieval approaches in terms of:

- retrieval quality;
- search latency;
- memory consumption;
- index size.

---

## Retrieval Methods

The project implements:

- **BM25** — lexical baseline;
- **Dense Retrieval** using Sentence Transformers;
- **FAISS IndexFlatIP** — exact vector search;
- **FAISS HNSW** — approximate nearest neighbour search;
- **Hybrid Retrieval (RRF)** *(optional)*.

---

## Dataset

Dataset:

**1.3M LinkedIn Jobs & Skills (2024)**

The original dataset contains approximately **1.3 million job postings**.

Experiments are performed on a representative sample of **500,000 job postings**.

Required files:

```text
linkedin_job_postings.csv
job_summary.csv
job_skills.csv
```

The files are joined using the common key:

```text
job_link
```

---

## Data Setup

The original dataset is **not included** in this repository because of its size.

Each team member should download it independently from Kaggle:

https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024

After extracting the archive, place the required files into:

```text
data/raw/
├── linkedin_job_postings.csv
├── job_summary.csv
└── job_skills.csv
```

Large files such as datasets, embeddings and FAISS indexes remain local and are excluded from Git.

---

## Repository Structure

```text
SkillMatch/
│
├── app/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── notebooks/
├── qrels/
├── results/
├── src/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Evaluation

Retrieval quality is evaluated using:

- nDCG@10
- MRR@10
- Precision@10
- ANN Recall@10

Efficiency is evaluated using:

- latency;
- embedding generation time;
- index size;
- RAM usage.

---

## Technology Stack

- Python
- Pandas
- NumPy
- Sentence Transformers
- FAISS
- Rank-BM25
- PyTorch
- Streamlit

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rslwqr/SkillMatch.git
cd SkillMatch
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Getting Started

1. Download the dataset from Kaggle.
2. Place the required CSV files into `data/raw/`.
3. Run the preprocessing pipeline.
4. Build the retrieval indexes.
5. Evaluate the retrieval methods.
6. Launch the Streamlit demo.