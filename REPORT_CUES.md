# Report cues: RAG Baseline vs Re-Ranking (Catan)

> Important: I could not run the notebook (no Gemini key or model downloads in my sandbox). Everything under "expected" below is a **hypothesis from how these models usually behave**, not a measured result. Your report must use the numbers from YOUR run, written in YOUR words (the assignment says AI-generated reports are not accepted).

## 0. How to work through it

1. Run sections 1-3 (setup, chunking, index). Look at the sample chunks. If the text is garbled, change the PDF copy.
2. Run the "check gold regexes" cell. Any query showing `[]` has a regex that doesn't match the PDF wording. Use `find_chunks("some phrase")` and read the chunk, then fix the regex.
3. Run the screener (section 7). It tells you which queries really show a RERANK WIN, without spending Gemini calls.
4. Run `run("<key>")` only for the winners, plus one control and one CE HURT/"not in top-10" case if you find any.
5. Fill the table in section 12 of the notebook with real numbers.

## 1. Queries to try manually (with the correct rule, to check Gemini's answers)

Correct rules below are from my knowledge of the Catan base rules. **Verify each against your PDF** before judging an answer right or wrong.

| Key | Category | Query | Correct rule (verify) | What the baseline may do wrong |
|---|---|---|---|---|
| trade_negation | Negation | Can two players who are not taking their turn trade resources with each other? | No. Only the player whose turn it is may trade; others can only trade with that player. | Gets general "you may trade" chunks, answer says "yes". |
| robber_stay | Negation | Am I allowed to leave the robber on the same hex when I move it? | No. It must be moved to a different hex. | Stealing / 7-roll chunks outrank the "must move" clause. |
| win_other_turn | Negation | I hit 10 victory points while another player was taking their turn, do I win immediately? | No. You win when you have 10+ VP on your own turn. | Generic "10 VP wins" chunk, misses the turn condition. |
| devcard_same_turn | Exception | Can I play a development card right after buying it in the same turn? | No (except a Victory Point card, which can be revealed to win). | Answers "yes" or omits the VP exception. |
| road_broken | Exception | Does my longest road still count if an opponent builds a settlement in the middle of it? | The road is broken at that point; recount the segments. | Describes Longest Road generally, omits the break rule. |
| road_tie | Exception | Two players both have a road of the same length, who keeps the Longest Road card? | The current holder keeps it. | Misses the tie-break sentence. |
| bank_trade_synonym | Keyword trap | Can I swap my resource cards for something I need without involving other players? | Yes, maritime trade: 4:1 with the bank, or 3:1 / 2:1 with harbors. | Pulled toward domestic-trade chunks (lexical overlap with "trade/cards/players"). |
| bigger_settlement | Keyword trap | How do I make my settlement bigger and what extra do I get from it? | Upgrade to a city (replaces the settlement), produces 2 resource cards. | "settlement" pulls in setup/distance-rule chunks. |
| second_settlement | Buried chunk | During setup, which resources do I receive for my second settlement? | One resource card per hex adjacent to the second settlement (placed in reverse order). | Setup text split across chunks, the key one sits at rank 4-10. |
| seven_discard | Control | I rolled a 7 and I only hold six cards, do I discard anything? | No, only players with more than 7 cards discard (half, rounded down). | Probably fine already. |
| knight_army | Control | How many knights do I have to play to take the Largest Army card? | At least 3. | Probably fine already (words match the question). |

Extra ideas if you want more stress cases (write the gold regex yourself after reading the PDF):
- "Can I build a road through a spot where another player has a settlement?"
- "What happens to my resource cards when the bank runs out of a resource?"
- "Can I play more than one development card in a turn?"
- "Can I trade with someone on my turn after I have already built?"

## 2. Expected patterns (to compare with your results)

- **Negation/constraint queries**: bi-encoders embed topic, not polarity. "may trade" and "may NOT trade" are almost the same vector, so the baseline often returns the permissive general rule. The cross-encoder reads query and chunk together, so it can see the restriction sentence actually answers "who can trade".
- **Exceptions**: the exception clause usually lives in a different paragraph/chunk from the general rule. Baseline returns the general rule; the cross-encoder promotes the chunk that directly addresses the specific scenario in the query.
- **Keyword trap**: when the query shares words with the wrong chunk (trade, settlement), cosine similarity rewards the overlap. Cross-encoder scores relevance of the whole pair.
- **Buried chunk**: short rules inside long chunks get diluted in a single 384-d vector.
- **Honest outcomes you may well see** (report these, they show understanding):
  - Controls where both pipelines are identical.
  - A "not in top-10" query: re-ranking cannot recover a chunk the first stage never retrieved. This is a recall limit of two-stage retrieval.
  - A "CE HURT" query: MiniLM trained on MS MARCO (web search) can prefer a chunk that looks like a definition over a short rule sentence.
  - Chunk overlap means neighbouring chunks both contain the answer, so a gold rank of "1 -> 1" can hide that two chunks matched.

## 3. Rank-shift table template (fill from your run)

| Query | Gold chunk | Baseline rank | Re-ranked rank | Gold in LLM context? (base / rerank) |
|---|---|---|---|---|
| | | | | |

Phrase it like "Chunk X (page Y) moved from rank #A to #B", exactly the format the assignment asks for.

## 4. Answer 1 vs Answer 2: what to look for

For each failure query, read both answers against the real rule and classify:
- **Hallucination**: states a rule that is not in the context (common when the needed chunk is missing and the model fills in from general Catan knowledge). Note that the prompt tells Gemini to say "excerpts don't cover this", so also check whether it abstained.
- **Omission**: correct main rule but misses the exception (e.g. dev card answer without the Victory Point exception).
- **Misstatement**: says the opposite (e.g. "yes players can trade with each other").
- **Same answer**: top-3 sets identical, nothing to compare.

Tip: also check whether Answer 1 was *wrong but confident*. That is a stronger point for the report than "slightly less detailed".

## 5. Latency cues (ballpark only, replace with your numbers)

| Stage | Roughly expected (CPU / Colab) |
|---|---|
| Query embedding (MiniLM) | ~5-30 ms |
| FAISS flat search, a few hundred chunks | < 1 ms |
| Cross-encoder, 10 pairs (MiniLM-L6) | ~30-150 ms on CPU, ~10-30 ms on GPU |
| Gemini generation | ~2-20 s (dominates everything, more with a "thinking" model) |
| One-time indexing of the document | a few seconds |

Points to make: re-ranking adds a small, bounded cost (10 pairs, so it does not grow with corpus size), and it is tiny compared to the LLM call. Retrieval overhead went from roughly "embed + search" to "embed + search + 10 transformer passes". Mention warm-up and averaging over repeats.

## 6. Architecture comparison cues (write in your own words)

- Bi-encoder: query and document are encoded **separately** into fixed vectors; relevance = cosine/dot product. Document vectors can be precomputed and indexed, so search is fast and scales. Downside: all query-document interaction is squeezed into one vector, so fine details (negation, exceptions, specific conditions) get blurred.
- Cross-encoder: query and document are concatenated into one input (`[CLS] query [SEP] chunk`) and run through the transformer together, so **every query token can attend to every chunk token** (joint self-attention). Output is a single relevance score. More accurate, but nothing can be precomputed: each pair needs a fresh forward pass.

## 7. Scaling question cues

- Cross-encoder cost is **O(N) transformer forward passes per query** (N = number of chunks), and none of it can be cached because the score depends on the query.
- Use your section-11 measurement: ms per pair x collection size. Typical result is minutes per query at 100k chunks and hours at 1M on CPU, versus milliseconds for a vector search.
- Bi-encoder + ANN index (HNSW/IVF) is sub-linear and the corpus side is precomputed once.
- Hence the two-stage design: cheap recall-oriented first stage (bi-encoder) narrows to ~10-100 candidates, expensive precision-oriented second stage (cross-encoder) re-sorts only those.
- Mention the trade-off: if the first stage misses the right chunk, the second stage can never fix it (your "not in top-10" example if you have one).

## 8. Small things that avoid losing marks

- The assignment sheet writes `cross-encoder/msmarco-MiniLM-L-6-v2`; the real Hugging Face id is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Mention which one you used.
- State your chunk size/overlap (180 words / 40 words) and bi-encoder model in the report.
- Push the notebook with outputs saved, and keep your API key out of the repo (the notebook reads it from Colab secrets / env / prompt).
- Quote the PDF page numbers for the gold chunks, it makes the analysis checkable.
