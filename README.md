# AQG-RAG-Pipeline (Last Updated: Spring 2025)

A retrieval-augmented generation (RAG) pipeline for **Automated Question Generation (AQG)** over
college-level textbook content. The system indexes OpenStax chapter text, retrieves topically
relevant passages, and generates grounded multiple-choice questions with distractors, a labeled
correct answer, and a rationale.

Built as part of the **[Algoverse AI Research](https://algoverseairesearch.org/)** program.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sabharwalainesh/AQG-RAG-Pipeline/blob/main/Automated_Question_Generation.ipynb)
[![Status: on hold](https://img.shields.io/badge/status-on%20hold-orange)](#project-status)

---

> [!IMPORTANT]
> **This project is unfinished and currently on hold.** The generation pipeline works end to end,
> but the evaluation stage — grading generated questions against Bloom's taxonomy — was never
> completed. See [Project status](#project-status) for details. Results here should be read as a
> working prototype, not as validated findings.

---

## Motivation

Hand-writing assessment items is the slowest part of building courseware. Prior work on automated
question generation largely fine-tunes seq2seq models on question-generation corpora, which ties
question quality to the training distribution and makes it expensive to target a new subject.

This project asks a different question: **how far does retrieval get you without fine-tuning?**
Instead of training a generator, the pipeline grounds a general-purpose LLM in retrieved textbook
passages at inference time. Swapping the corpus swaps the subject matter — no retraining required.

## How it works

```
OpenStax chapters (data.json)
        │
        ▼
  Field concatenation ──── bname + intro + chapter_text + summary
        │
        ▼
  Chunking ─────────────── RecursiveCharacterTextSplitter (tiktoken)
                           chunk_size=500 tokens, overlap=50
        │
        ▼
  Embedding + Index ────── OpenAI embeddings → Chroma ("chapter-texts")
        │
        ▼
  Retrieval ────────────── query: "<book> ... focusing on <topic>"
                           top-3 chunks passed as context
        │
        ▼
  Generation ───────────── gpt-4o-mini, temperature=0
                           JSON-constrained MCQ prompt
        │
        ▼
  Dedup + Serialize ────── exact-match filter on question text
                           → generated_mcq_questions_with_topics_no_dupes.json
```

The generation prompt constrains the model to emit **valid JSON only** and explicitly instructs it
not to copy context verbatim, which keeps items from degenerating into cloze deletions of the
source text.

## Repository structure

| Path | Description |
| --- | --- |
| `Automated_Question_Generation.ipynb` | End-to-end pipeline: install, index, retrieve, generate, save |
| `data.json` | Source corpus — 192 chapters across 12 OpenStax textbooks (~20 MB) |
| `generated_mcq_questions_with_topics_no_dupes.json` | Sample generated output from a pipeline run |

## Corpus

`data.json` is a list of chapter records with the following schema:

| Field | Type | Description |
| --- | --- | --- |
| `bname` | `str` | Book identifier, e.g. `biology`, `business_ethics` |
| `chapter` | `int` | Chapter number within the book |
| `intro` | `str` | Chapter outline and opening narrative |
| `chapter_text` | `str` | Full chapter body |
| `summary` | `str` | Section-by-section chapter summary |
| `questions` | `list` | Gold end-of-chapter questions with choices and answers |
| `keyterm` | `str` | Key-term glossary (often empty) |

Book coverage:

| Book | Chapters |
| --- | --- |
| biology | 39 |
| anatomy_and_physiology | 23 |
| u.s._history | 22 |
| microbiology | 20 |
| introduction_to_sociology | 15 |
| american_government | 15 |
| principles_of_accounting, vol. 1 (financial) | 13 |
| principles_of_accounting, vol. 2 (managerial) | 12 |
| psychology | 12 |
| business_law_i_essentials | 11 |
| business_ethics | 9 |
| introduction_to_intellectual_property | 1 |

The `questions` field is not consumed by the current pipeline. It is retained as a gold reference
set for downstream evaluation.

## Output format

```json
[
  {
    "subtopic": "business_ethics",
    "topics": [
      {
        "topic_name": "finance",
        "multiple_choice_questions": [
          {
            "question": "...",
            "options": ["...", "...", "...", "..."],
            "correct_answer": "...",
            "explanation": "..."
          }
        ]
      }
    ]
  }
]
```

Output is grouped by book (`subtopic`), then by the high-level lens the question was framed
through (`topic_name`).

## Setup

The notebook is written for Google Colab with Drive mounted, but runs anywhere with a Python 3.10+
environment.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

Pinned versions matter here — LangChain 0.0.327 predates the `langchain-community` split, so
imports like `from langchain.vectorstores import Chroma` will break on newer releases.

### Running in Colab

1. Open the notebook via the badge above.
2. Run the install cell, then the `os._exit(00)` cell. This hard-restarts the runtime so the pinned
   `pydantic` version is actually loaded — Colab ships a conflicting version at boot, and without
   the restart Chroma fails on import.
3. Provide your OpenAI key when prompted, then run the pipeline cell.

### Running locally

Point `dataset_path` at `data.json` and drop the `drive.mount()` and `google.colab` imports.

## Configuration

| Parameter | Value | Notes |
| --- | --- | --- |
| `chunk_size` | 500 tokens | Tiktoken-based, not character-based |
| `chunk_overlap` | 50 tokens | Must be `< chunk_size` (asserted) |
| Retriever `k` | 4 retrieved, top 3 used | Only the first 3 chunks reach the prompt |
| Generation model | `gpt-4o-mini` | |
| `temperature` | 0 | See limitations |
| Questions per (book, topic) | 10 target | Capped at 50 sampling attempts |

## Project status

**On hold. The pipeline is complete; the evaluation is not.**

The research design had two halves. The first — retrieve textbook context and generate grounded
multiple-choice items — is implemented and runnable, and is what this repository contains. The
second half was the part that would have made it a result rather than a demo.

### What was planned

Generated questions were to be graded against **Bloom's taxonomy**, classifying each item by
cognitive level (recall, comprehension, application, analysis, evaluation, synthesis). The
hypothesis under test was that retrieval grounding shifts the distribution of generated items
upward — that passing chapter context to the model produces fewer low-level recall questions and
more items requiring application or analysis, relative to an ungrounded baseline.

That claim is only meaningful if the grading is credible. Bloom's-level classification of
assessment items is a judgment call that requires subject-matter expertise: distinguishing a
genuine application question from a recall question wearing application vocabulary is exactly the
distinction a non-expert gets wrong. Using the same class of model to both generate and grade the
items would have been circular.

### Why it stopped

The project could not secure professors or graders with the scholarly expertise in these subject
areas to evaluate the generated questions. Without qualified human annotators, there was no
defensible way to score the output, and no evaluation infrastructure or budget to substitute for
them. Rather than publish scores from unqualified grading, the research was paused.

### What would restart it

- Access to subject-matter experts in the corpus domains (biology, accounting, sociology, U.S.
  history, and so on) willing to annotate a sample of generated items by Bloom's level
- An annotation protocol with a written rubric and at least two independent raters per item, so
  inter-rater agreement can be reported alongside the results
- A baseline arm — the same model generating without retrieved context — to compare the Bloom's
  distribution against, since the grounded numbers mean nothing in isolation
- The fixes listed below, so the sample being graded is actually diverse enough to be worth grading

Anyone with relevant expertise who wants to pick this up is welcome to open an issue.

## Known limitations and next steps

These are open items, not solved problems — flagged here so results are read with the right caveats.

- **Determinism suppresses yield.** `temperature=0` plus a fixed retrieval query means repeated
  sampling for the same `(book, topic)` pair returns near-identical questions. Because dedup is an
  exact string match, most attempts are discarded. The committed sample output contains 2–8 items
  per cell rather than the target 10, and the attempt cap is what terminates the loop. Raising
  temperature, or diversifying the retrieval query per attempt, is the obvious fix.
- **Exact-match dedup is too weak.** Two questions can be semantically identical and lexically
  distinct. Embedding-similarity dedup with a threshold would be a stricter filter.
- **The retrieval query mixes two axes.** Querying for a `bname` and a topic in the same string
  can retrieve passages relevant to one but not the other. Metadata filtering on `bname` with
  semantic search restricted to the topic would separate them cleanly.
- **No chunk metadata.** `Document` objects are built without a metadata dict, so retrieved chunks
  cannot be traced back to their source book or chapter. This blocks both filtered retrieval and
  provenance checking of generated items.
- **No quantitative evaluation.** This is the blocking gap described under
  [Project status](#project-status). The `questions` field in `data.json` provides gold
  end-of-chapter items for the same chapters, so an automatic comparison on answerability and
  distractor plausibility is feasible without human raters — but the Bloom's-level analysis, which
  was the actual research question, is not.
- **The committed sample predates the current notebook.** `generated_mcq_questions_with_topics_no_dupes.json`
  was produced with an earlier topic list (`finance`, `health`, `social`, `politics`, `technology`);
  the notebook now specifies subject-area topics. Regenerate to reproduce.

## Acknowledgments

### Hugging Face — EduQG

The corpus in this repository is derived from the **EduQG** dataset released by Amir Hadifar,
Semere Kiros Bitew, Johannes Deleu, Chris Develder, and Thomas Demeester. Their work assembled and
cleaned the OpenStax chapter/question pairs that this pipeline retrieves over, and it remains the
reference point for question-generation work on this corpus.

- Model: [`hadifar/openstax_qg_agno`](https://huggingface.co/hadifar/openstax_qg_agno) — the
  fine-tuned seq2seq question generator released alongside the dataset, and the natural
  fine-tuning baseline against which a retrieval-only approach should be measured
- Model: [`hadifar/tqa_qg_agno`](https://huggingface.co/hadifar/tqa_qg_agno) — companion model
  trained on TQA
- Dataset mirror: [`voidful/EduQG`](https://huggingface.co/datasets/voidful/EduQG)
- Source repository: [hadifar/question-generation](https://github.com/hadifar/question-generation)

```bibtex
@article{hadifar2023eduqg,
  title   = {EduQG: A Multi-Format Multiple-Choice Dataset for the Educational Domain},
  author  = {Hadifar, Amir and Bitew, Semere Kiros and Deleu, Johannes and
             Develder, Chris and Demeester, Thomas},
  journal = {IEEE Access},
  volume  = {11},
  pages   = {20885--20896},
  year    = {2023},
  eprint  = {arXiv:2210.06104}
}
```

### OpenStax

Underlying textbook content is published by [OpenStax](https://openstax.org/), Rice University,
under Creative Commons licensing. Please observe OpenStax attribution requirements when
redistributing derived text.

### Algoverse AI Research

This project was completed under the mentorship of the
[Algoverse AI Research](https://algoverseairesearch.org/) program.

### Tooling

[LangChain](https://github.com/langchain-ai/langchain) · [Chroma](https://github.com/chroma-core/chroma) · [OpenAI API](https://platform.openai.com/)

## License

Code in this repository is available under the MIT License. Note that this does **not** extend to
`data.json`, which inherits the licensing of the EduQG dataset and the underlying OpenStax content.
