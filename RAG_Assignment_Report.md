# RAG Assignment Report: Baseline Retrieval vs. Re-Ranking

**Author**: Rohit Rohon Sabat  
**GitHub Repository**: [https://github.com/rohitrohon/LMA-Assignment-1](https://github.com/rohitrohon/LMA-Assignment-1)  
**Deliverables**: Jupyter Notebook (`rag_baseline_vs_rerank_catan_results.ipynb`) + Report  

---

## 1. Setup & Methodology

- **Corpus**: Official 16-page Catan base game rulebook. Using a fixed sliding window of 180 words with a 40-word overlap, the document was partitioned into 73 chunks.
- **Bi-Encoder (Pipeline 1 Baseline)**: `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional unit-normalized embeddings, indexed using FAISS `IndexFlatIP` (exact inner product / cosine similarity). Initial retrieval retrieves the Top 10 chunks, and the Top 3 are passed to the generator.
- **Cross-Encoder (Pipeline 2 Re-Ranking)**: `cross-encoder/ms-marco-MiniLM-L-6-v2`. The Top 10 baseline candidates are paired with the query as `(query, chunk)` and scored in a single forward pass. Candidates are re-sorted by cross-encoder score, and the Top 3 re-ranked chunks are passed to the generator.
- **LLM**: Gemini API (`gemini-2.5-flash` with `gemini-3.8-flash` fallback), `temperature=0`, using an identical prompt across both pipelines.

---

## 2. Architecture Comparison

The fundamental difference between a Bi-Encoder and a Cross-Encoder lies in **how and when token interactions occur**:

### Bi-Encoder (Independent Encodings & Late Interaction)
A Bi-Encoder encodes the query and each document passage completely independently:
$$\vec{u} = \text{Encoder}(q), \quad \vec{v}_i = \text{Encoder}(d_i)$$
Relevance is scored by the geometric similarity (dot product or cosine angle) between two fixed-size vectors in $\mathbb{R}^{384}$. 

- **Advantage**: Because $\vec{v}_i$ does not depend on the query, all chunk embeddings can be computed once offline and indexed in a vector database (like FAISS). At query time, we only embed the query once and perform matrix-vector multiplication.
- **Limitation**: Squeezing an entire 180-word passage into a single 384-dimensional vector causes an information bottleneck. Subtle semantic qualifiers—such as negation ("not allowed"), temporal limits ("only during your turn"), or corner-case exceptions—are blurred into the dominant topical theme.

### Cross-Encoder (Joint Early Cross-Attention)
A Cross-Encoder concatenates the query and the chunk into a single sequence:
$$\text{Input} = \texttt{[CLS]} \, q \, \texttt{[SEP]} \, d \, \texttt{[SEP]}$$
It processes this unified sequence through all 6 transformer layers simultaneously.

- **Advantage**: Every single token in the query attends directly to every single token in the chunk across all attention heads. The model can evaluate whether the passage actually answers the question, distinguishing fine-grained constraints from surface-level keyword overlap. It outputs a calibrated relevance score directly.
- **Limitation**: Nothing can be precomputed offline. Because the joint representation depends on both $q$ and $d$, evaluating $N$ chunks requires $N$ separate forward passes through the transformer at query time.

---

## 3. Qualitative & Rank Shift Analysis

To evaluate retrieval precision, target chunks containing the authoritative rule were identified by manual inspection of the rulebook text. Rather than relying on loose regex pattern matching, fixed chunk IDs were assigned to verify exactly where the true answer ranked before and after re-ranking.

### Rank Shift Summary

| Query Key | Query Category | Target Rule Chunk | Baseline Rank | Re-Ranked Rank | Rank Shift | Target in LLM Top-3? |
|---|---|---|:---:|:---:|:---:|:---:|
| `trade_negation` | Negation / Constraint | Chunk 10 (p. 4) | #8 | #1 | **+7** | No &rarr; **Yes** |
| `road_broken` | Exception / Corner Case | Chunk 33 (p. 9) | #5 | #2 | **+3** | No &rarr; **Yes** |
| `bigger_settlement` | Keyword Trap / Synonym | Chunk 13 (p. 5) | #7 | #2 | **+5** | No &rarr; **Yes** |
| `bank_trade_synonym` | Keyword Trap / Synonym | Chunk 34 (p. 9) | #4 | #1 | **+3** | No &rarr; **Yes** |
| `win_other_turn` | Control (Turn Condition) | Chunk 26 (p. 7) | #1 | #1 | 0 | Yes &rarr; Yes |
| `knight_army` | Control (Exact Keywords) | Chunk 16 (p. 5) | #2 | #1 | +1 | Yes &rarr; Yes |
| `custom_query` (7-roll) | Complex Scenario / Math | Chunk 44 / 47 (p. 11–12) | #2 | #1 | +1 | Yes &rarr; Yes |

---

### Detailed Failure Query Analysis & Answer Comparisons

#### 1. Exception / Corner Case: `road_broken`
- **Query**: *"Does my longest road still count if an opponent builds a settlement in the middle of it?"*
- **Target**: Chunk 33 (p. 9, Almanac: "Longest Road"): explains that building a settlement on an unoccupied intersection breaks the road into two segments, requiring players to re-measure continuous segments, and details the tie-breaking rules.
- **Retrieval Behavior**:
  - **Baseline**: Top 3 were Chunks 32, 12, 48. Chunk 33 was buried at **Rank #5**. The Bi-Encoder rewarded Chunk 32 (general definition of Longest Road) and Chunk 12 (basic road building costs) because of lexical overlap with "road".
  - **Re-ranked**: Chunk 33 was promoted to **Rank #2** (CE score: +3.55). Re-ranked Top 3: Chunks 32, 33, 12.
- **LLM Answer Comparison**:
  - **Answer 1 (Baseline)**: *Misstatement & Distortion*. Stated that the road splits, but claimed the player *"must give the special card to another player in this scenario."* This misinterprets the condition under which the card changes hands (it only changes hands if another player has a strictly longer road; it does not automatically transfer to the opponent who built the settlement).
  - **Answer 2 (Re-ranked)**: *Accurate & Comprehensive*. Correctly explained that the road is split into two separate segments, that the player must re-evaluate their longest continuous road, and that they lose the card only if they no longer hold the longest road. It also captured the nuance from Chunk 33 that if the incumbent remains tied for longest road after the break, they retain the card.

---

#### 2. Keyword Trap / Synonym: `bigger_settlement`
- **Query**: *"How do I make my settlement bigger and what extra do I get from it?"*
- **Target**: Chunk 13 (p. 5, Game Rules: "City"): explicitly states that a settlement is upgraded to a city for 3 ore and 2 grain, replaces the piece, and produces 2 resource cards instead of 1.
- **Retrieval Behavior**:
  - **Baseline**: Top 3 were Chunks 53, 71, 4. The word "settlement" acted as a distractor, pulling in setup rules, the distance rule, and general tactical advice. Chunk 13 was buried at **Rank #7** (cosine 0.453).
  - **Re-ranked**: Chunk 13 surged to **Rank #2** (CE score: -1.77). Re-ranked Top 3: Chunks 71, 13, 47.
  - *Note on Cross-Encoder scores*: For this query, all Cross-Encoder scores were negative (the top score was -1.09), indicating lower general model confidence due to vocabulary mismatch ("bigger" vs. "upgrade/city"). Despite low absolute logit scores, the relative ranking correctly pushed the true rule into the top 2.
- **LLM Answer Comparison**:
  - **Answer 1 (Baseline)**: *Severe Omission*. Stated: *"You make your settlement bigger by upgrading it to a city. Each city is worth 2 victory points. To upgrade, you need to acquire resources."* It failed to mention the cost (3 ore, 2 grain) and omitted the primary benefit (producing 2 resource cards instead of 1).
  - **Answer 2 (Re-ranked)**: *Complete and Exact*. Specified that upgrading requires 3 Ore and 2 Grain, replaces the settlement piece, awards 2 victory points, and produces 2 resource cards from adjacent hexes instead of 1.

---

#### 3. Keyword Trap / Synonym: `bank_trade_synonym`
- **Query**: *"Can I swap my resource cards for something I need without involving other players?"*
- **Target**: Chunk 34 (p. 9, Almanac: "Maritime Trade"): details maritime trading with the supply/bank and the specific harbor rates (4:1, 3:1, 2:1).
- **Retrieval Behavior**:
  - **Baseline**: Top 3 were Chunks 9, 40, 38. The phrase "trade/swap cards with players" heavily favored domestic trading passages. Chunk 34 sat at **Rank #4**.
  - **Re-ranked**: Chunk 34 was elevated to **Rank #1** (CE score: +5.31). Re-ranked Top 3: Chunks 34, 9, 38.
- **LLM Answer Comparison**:
  - **Answer 1 (Baseline)**: *Incomplete*. Confirmed that you can trade without other players via maritime trade, quoting: *"You can also trade without the other players!"*, but provided zero exchange rates or mechanics.
  - **Answer 2 (Re-ranked)**: *Precise*. Explained the standard 4:1 bank exchange rate, the 3:1 generic harbor rate, and the 2:1 special harbor rate.

---

#### 4. Negation & The Chunk Overlap Phenomenon: `trade_negation`
- **Query**: *"Can two players who are not taking their turn trade resources with each other?"*
- **Target**: Chunk 10 (p. 4, Game Rules: "Domestic Trade"): contains the explicit restriction: *"Important: Players may only trade with the player whose turn it is. The other players may not trade among themselves."*
- **Retrieval Behavior**:
  - **Baseline**: Top candidate was Chunk 9 (p. 4, general trade overview). Chunk 10 was buried at **Rank #8** (cosine 0.435). The Bi-Encoder latched onto the permissive words "trade", "resources", and "players".
  - **Re-ranked**: Chunk 10 moved from **Rank #8 &rarr; Rank #1** (CE score: +6.83, a massive +7 promotion).
- **LLM Answer Comparison**:
  - Both Answer 1 and Answer 2 answered correctly: *"No, two players who are not taking their turn cannot trade resources with each other."*
  - **Why did the baseline succeed despite poor retrieval?**  
    Because of the sliding-window chunking (180 words, 40-word overlap), the last 40 words of Chunk 9 are identical to the first 40 words of Chunk 10. The sentence *"Players may only trade with the player whose turn it is. The other players may not trade among themselves"* happened to fall squarely within those shared 40 words.
  - **Key Takeaway**: Chunk overlap can inadvertently mask retrieval failures by providing the required clause inside an irrelevant chunk. This demonstrates why automated metrics can be deceptive and why inspecting retrieved chunks by hand is necessary.

---

#### 5. Control Cases
- **`win_other_turn`**: Chunk 26 was ranked #1 by both pipelines (cosine: 0.588, CE score: +8.40). Top 3 were identical, and both answers correctly stated that a player can only win on their own turn.
- **`knight_army`**: Chunk 16 was ranked #2 by baseline and #1 by re-ranking. Both pipelines included the chunk in their Top 3, and both correctly stated 3 knights are required.
- **Custom Scenario (Rolling 7 with 15 cards: 13 resource, 2 dev cards)**: Chunk 44/47 was ranked #2 by baseline and #1 by re-ranking. Both models correctly identified that only resource cards count (13 cards), and that half rounded down must be discarded (6 resource cards).

---

## 4. Latency Profiling

Retrieval and generation latencies were benchmarked across 5 repeated runs over 5 distinct queries following a warm-up execution. Timings were measured on an Apple Silicon M-series machine (using `mps` device acceleration for PyTorch).

### Runtime Breakdown

| Pipeline Stage | Baseline Pipeline | Re-Ranking Pipeline | Overhead / Ratio |
|---|:---:|:---:|:---:|
| Query Embedding (`all-MiniLM-L6-v2`) | 5.87 ms | 5.87 ms | 1.0x |
| FAISS Index Search (`IndexFlatIP`, 73 chunks) | 0.03 ms | 0.03 ms | 1.0x |
| Cross-Encoder Scoring (10 candidate pairs) | *Not used* | 66.43 ms | +66.43 ms |
| **Total Retrieval Latency** | **5.90 ms** | **72.33 ms** | **~12.2x** |
| **LLM Generation Latency (Gemini API)** | **4,670 ms** | **4,670 ms** | **1.0x** |
| **End-to-End Latency** | **4,675.90 ms** | **4,742.33 ms** | **~1.014x (+1.4%)** |

### Key Observations:
1. **Retrieval Overhead**: Adding the Cross-Encoder increases retrieval latency by roughly 12x (from ~5.9 ms to ~72.3 ms). Evaluating 10 pairs through a 6-layer transformer takes ~6.6 ms per pair.
2. **End-to-End Context**: In a production RAG application, retrieval latency is dwarfed by LLM generation time (~4.67 seconds). The 66.4 ms re-ranking delay accounts for **less than 1.5% of total user waiting time**, while resolving hallucinations and omissions in 75% of the failure queries.
3. **One-Time Cost**: The one-time offline cost of embedding and indexing all 73 chunks was just **0.45 seconds**.

---

## 5. Scaling Analysis

### Why can we not run the Cross-Encoder over the entire document collection?

Evaluating the Cross-Encoder across an entire corpus is computationally prohibitive because **Cross-Encoder inference cost scales as $\mathcal{O}(N)$ transformer forward passes per query**, where $N$ is the total number of chunks. Because the model computes full cross-attention across the concatenated $(q, d)$ sequence, **no chunk representations can be precomputed or cached**.

### Empirical Benchmarks and Projections

To measure real-world scaling, the Cross-Encoder was executed over all 73 chunks in the Catan rulebook:
- Total runtime for 73 chunks: **1.24 seconds**
- Measured time per query-chunk pair: **~17.0 ms / pair** (unbatched / small-batch overhead)

Extrapolating this empirical throughput to larger corpus sizes:

| Corpus Size ($N$) | Cross-Encoder Time per Query | FAISS Vector Search Time | Speedup Factor |
|---|:---:|:---:|:---:|
| **73 chunks** (Catan) | 1.24 s | 0.03 ms | ~41,000x |
| **1,000 chunks** | 17.0 s (~0.3 min) | 0.15 ms | ~113,000x |
| **10,000 chunks** | 170 s (~2.8 min) | 0.8 ms | ~212,000x |
| **100,000 chunks** | 1,700 s (**~28.3 min**) | 6.2 ms | **~274,000x** |
| **1,000,000 chunks** | 16,998 s (**~4.7 hours**) | 45.0 ms | **~377,000x** |

### Vector Search Benchmark
To provide a direct empirical comparison, a synthetic FAISS `IndexFlatIP` index was constructed over **100,000 vectors** (384 dimensions). A flat exact search for the top 10 nearest neighbors completed in **6.2 milliseconds**. 

Searching 100,000 vectors via FAISS took 6.2 ms, whereas running a Cross-Encoder over the same 100,000 chunks would require roughly **28 minutes** per query on the same hardware.

### The Two-Stage Paradigm
This divergence explains why modern information retrieval architectures rely on a **two-stage cascade**:
1. **Stage 1 (High Recall, Fast)**: The Bi-Encoder projects the query into vector space and leverages approximate nearest neighbor (ANN) or flat vector indexing to filter millions of documents down to candidate pool $K$ (e.g., $K = 10 \text{ to } 50$) in milliseconds.
2. **Stage 2 (High Precision, Deep Cross-Attention)**: The Cross-Encoder is applied *only* to the $K$ candidates, operating in constant time $\mathcal{O}(K)$ relative to the corpus size.

### Systemic Limitation
The primary failure mode of this cascade is that **the second stage cannot recover chunks that the first stage misses entirely**. If the Bi-Encoder fails to place the authoritative chunk in the initial Top-$K$ candidate pool, the Cross-Encoder never sees it. Choosing $K$ represents a direct trade-off between recall ceiling and inference budget.

---

## 6. Summary of Findings

1. **Precision Restoration**: The Cross-Encoder successfully resolved failure modes across Negation, Keyword Distractors, and Exceptions, promoting target chunks from deep baseline ranks (#4–#8) directly into the Top 2.
2. **Impact on Answers**: In the three queries where the retrieved context genuinely changed, the baseline LLM answers suffered from rule misstatements or severe omissions, while re-ranking produced complete and legally precise answers.
3. **Latency Feasibility**: The 66.4 ms re-ranking overhead represents a negligible 1.4% increase in end-to-end response time.
4. **Methodological Lesson**: Lexical regexes are prone to high false-positive rates when evaluating rulebooks; explicit chunk IDs combined with manual inspection are necessary for rigorous retrieval benchmarking.
