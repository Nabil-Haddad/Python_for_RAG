# RAG Lessons

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)
![Gensim](https://img.shields.io/badge/Gensim-GloVe-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF%20loading-red)
![asyncio](https://img.shields.io/badge/asyncio-concurrency-lightgrey)

A hands-on, step-by-step path to learning **Retrieval-Augmented Generation (RAG)** with Python. Each lesson is a small, runnable script with the explanation written next to the code, so you can read it, run it, and change it.

The lessons are written by a learner, for learners. If something is unclear or wrong, please open an issue.

## What is RAG?

A language model only knows what it saw during training. RAG gives it a way to look things up first:

```
Documents → chunks → embeddings → vector search → relevant context → LLM answer
```

Every step hands data to the next one, so this course starts with the building blocks (data structures, loading files, async, logging) and then moves to embeddings and retrieval.

## Who is this for?

- You know basic Python (functions, classes, lists, dicts).
- You want to understand how a RAG pipeline works, not only call a library.
- No machine learning background is needed.

## Repository layout

```
RAG_lessons/
├── Foundations/        Python skills every RAG pipeline needs
├── skills/
│   └── embedding/      How text becomes vectors
├── data/
│   └── processed/      Sample chunks used by the lessons
├── requirements.txt
└── .env.example        Template for your secrets
```

## Lessons

### Foundations

Read them in order. The first lines of each file list what it covers.

| # | File | Topic |
|---|------|-------|
| 1 | [Lesson01.py](Foundations/Lesson01.py) | Virtual environments, secrets with `.env`, config validation |
| 2 | [Lesson02.py](Foundations/Lesson02.py) | Dataclasses, Pydantic models, list comprehensions, dict patterns |
| 3 | [Lesson03.py](Foundations/Lesson03.py) | File I/O, loading PDFs and text, chunking with overlap, JSON |
| 4 | [Lesson04.py](Foundations/Lesson04.py) | Async Python: `asyncio.gather`, semaphores, retries |
| 5 | [Lesson05.py](Foundations/Lesson05.py) | Type hints, structured logging, context managers |

### Skills: embeddings

| File | Topic |
|------|-------|
| [embedding.md](skills/embedding/embedding.md) | Static vs contextual embeddings, how static ones are trained, dot product vs cosine similarity (read this first) |
| [static_embedding.py](skills/embedding/static_embedding.py) | Load GloVe vectors, inspect them, nearest words, `king - man + woman`, 2D plots |
| [vectors_similarity.py](skills/embedding/vectors_similarity.py) | A small plot showing the difference between dot product and cosine similarity |

## Getting started

You need Python 3.12 or newer (some scripts use f-string syntax from 3.12).

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd RAG_lessons

# 2. Create and activate a virtual environment
python -m venv .ragenv
source .ragenv/bin/activate        # Windows: .ragenv\Scripts\activate

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Create your own .env file from the template
cp .env.example .env
```

The embedding scripts need a few more packages than the foundations lessons:

```bash
pip install gensim numpy matplotlib scikit-learn
```

## Running a lesson

Run the scripts from the repository root, so relative paths such as `data/processed/` work:

```bash
python Foundations/Lesson02.py
python skills/embedding/static_embedding.py
```

The first run of `static_embedding.py` downloads the GloVe model (about 66 MB) into `~/gensim-data/`. Later runs reuse the cached copy.

## How to use this material

1. Read the comments at the top of the file: they say why the topic matters and what you will learn.
2. Run the script and compare the output with the code.
3. Change something (a value, a word, a chunk size) and run it again. Most of the learning happens here.
4. Read the matching `.md` notes when there is one.

## Contributing

Corrections, typo fixes and new lesson ideas are welcome. Open an issue or a pull request.
