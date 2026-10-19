# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking (Catan)

**Course**: Large Model Architectures — Assignment 1  
**Author**: Rohit Rohon Sabat  
**Repository**: [https://github.com/rohitrohon/LMA-Assignment-1](https://github.com/rohitrohon/LMA-Assignment-1)

---

## Overview

This repository implements, evaluates, and contrasts two Retrieval-Augmented Generation (RAG) workflows over the official 16-page Catan base game rulebook:

1. **Pipeline 1 (Baseline Retrieval)**:
   - Chunking: 180 words with a 40-word sliding window overlap (73 chunks).
   - Bi-Encoder: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
   - Indexing: FAISS `IndexFlatIP` (exact cosine similarity / inner product).
   - Generation: Top 10 candidate retrieval &rarr; Top 3 chunks passed to Gemini API.

2. **Pipeline 2 (Advanced Re-Ranking)**:
   - Candidates: Top 10 retrieved from Pipeline 1.
   - Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` (joint self-attention across `(query, chunk)` tokens).
   - Re-sorting: Re-scores and re-ranks all 10 candidates; Top 3 passed to Gemini API using an identical prompt.

---

## Key Results & Findings

### 1. Measured Rank Shifts
Evaluating target rule chunks using verified chunk IDs demonstrates dramatic promotions for complex query types:

| Query Key | Query Category | Target Rule Chunk | Baseline Rank | Re-Ranked Rank | Rank Shift |
|---|---|---|:---:|:---:|:---:|
| `trade_negation` | Negation / Constraint | Chunk 10 (p. 4) | #8 | #1 | **+7** |
| `road_broken` | Exception / Corner Case | Chunk 33 (p. 9) | #5 | #2 | **+3** |
| `bigger_settlement` | Keyword Trap / Synonym | Chunk 13 (p. 5) | #7 | #2 | **+5** |
| `bank_trade_synonym` | Keyword Trap / Synonym | Chunk 34 (p. 9) | #4 | #1 | **+3** |
| `win_other_turn` | Control (Turn Condition) | Chunk 26 (p. 7) | #1 | #1 | 0 |
| `knight_army` | Control (Exact Keywords) | Chunk 16 (p. 5) | #2 | #1 | +1 |
| `custom_query` (7-roll) | Complex Scenario / Math | Chunk 44 / 47 (p. 11–12) | #2 | #1 | +1 |

### 2. Qualitative Answer Improvements
- **`road_broken`**: Baseline Answer 1 misstates that the player *"must give the special card to another player"*. Re-ranked Answer 2 accurately states that road segments split, continuous lengths are re-measured, and the incumbent card holder keeps the card if still tied.
- **`bigger_settlement`**: Baseline Answer 1 omitted the upgrade cost (3 ore, 2 grain) and the double production benefit. Re-ranked Answer 2 is complete and precise.
- **`bank_trade_synonym`**: Baseline Answer 1 omitted harbor exchange rates. Re-ranked Answer 2 provides the full 4:1, 3:1, and 2:1 schedules.
- **`trade_negation`**: Both answers correctly identified the prohibition because the 40-word sliding window overlap included the restriction sentence in Chunk 9.

### 3. Latency & Scaling Profile
- **Retrieval Overhead**: Re-ranking takes **66.4 ms** for 10 candidate pairs, increasing retrieval from **5.90 ms** to **72.33 ms** (~12.2x).
- **End-to-End Impact**: Across 4.67 s Gemini generation time, re-ranking adds **less than 1.5%** to total latency.
- **Scaling Limit**: Cross-Encoder scales as $\mathcal{O}(N)$ transformer forward passes per query (projecting to ~28 min at 100k chunks and ~4.7 hours at 1M chunks), whereas FAISS exact vector search over 100,000 vectors runs in **6.2 ms** (~274,000x faster). This highlights the necessity of the two-stage RAG cascade.

---

## Repository Structure

```
.
├── rag_baseline_vs_rerank_catan_results.ipynb   # Executed notebook with all outputs and tables
├── rag_baseline_vs_rerank_catan.ipynb           # Clean solution notebook
├── RAG_Assignment_Report.md                     # Full comprehensive assignment report
├── catan_rules.pdf                              # Official Catan base rules document
├── RAG_LMA_Assignment_v1.pdf                    # Assignment problem statement
└── README.md                                    # Project overview
```

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rohitrohon/LMA-Assignment-1.git
   cd LMA-Assignment-1
   ```

2. **Install dependencies**:
   ```bash
   pip install google-genai pypdf sentence-transformers faiss-cpu pandas numpy requests
   ```

3. **Set your Gemini API Key**:
   Create a `.env` file in the root directory:
   ```bash
   echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
   ```
   *(Alternatively, the notebook will prompt for your key via `getpass` or read from Colab `userdata`)*.

4. **Launch Jupyter**:
   ```bash
   jupyter notebook rag_baseline_vs_rerank_catan_results.ipynb
   ```
   Run all cells sequentially via **Kernel &rarr; Restart & Run All**.
