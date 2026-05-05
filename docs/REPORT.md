# Personalized Career Path & Skill Recommendation System

**Course:** Recommender Systems — Spring 2026
**Institute:** National University of Computer and Emerging Sciences, Karachi
**Instructor:** Syed Zain Ul Hassan
**Team:** Atif Arif (22K-4358) · Unzila Javed (22K-4168) · Mishkaat Yousuf (22K-4624) · Rao Abdul Hassi (22K-4202)

---

## 1. Introduction

Job seekers and career-switchers are routinely told *what role to aim for* but seldom told *which specific skills to acquire next*. We built an explainable, content-based recommender that closes that gap. Given a user's current skills, the system (i) ranks job roles drawn from real LinkedIn postings by a 0–1 *Match Score*, (ii) explains every recommendation by listing the matched, missing, and extra skills, and (iii) suggests a *bridge path* — the shortest sequence of intermediate roles that takes the user from their current skill profile to a chosen target role. The recommender is deliberately not a black box: every output is traceable to TF-IDF cosine similarity, set operations, and Dijkstra shortest-path on a NetworkX graph, in line with the explainability requirement of the proposal and the project brief.

## 2. Dataset

We use the public Kaggle **"LinkedIn Job Postings (2023–2024)"** dataset — 123,849 postings across 31 columns. The fields most relevant to us are `title`, `description`, `formatted_experience_level`, `formatted_work_type`, `location`, `company_name`, `normalized_salary`, and `job_id` (used to synthesize the click-through LinkedIn URL `linkedin.com/jobs/view/{job_id}/`).

A critical observation drove our preprocessing pipeline: the structured `skills_desc` column is populated for only **2.0%** of rows. Therefore skill extraction must work primarily off the free-text `description` field. After text normalization, regex-based skill extraction (parallelized across 11 CPU cores), and dropping rows with empty descriptions or zero extracted skills, we retain **52,074 usable postings**. These are aggregated by normalized title into **1,060 distinct roles** (each backed by ≥5 postings to avoid noisy singletons). This role-level abstraction is the unit the recommender suggests — a user is recommended *career categories*, not 300 copies of the same listing.

## 3. Model

The system is composed of five modules, each individually testable.

**3.1 Skill extractor.** A curated vocabulary of 110 canonical technical skills (`src/skills_vocab.py`), each with surface-form aliases (e.g. *Postgres → PostgreSQL*, *k8s → Kubernetes*). A single compiled regex over all aliases scans free text. Ambiguous short tokens (`R`, `Go`, `C#`, `C++`) require a programming-context cue (e.g. *programming, language, developer, experience*) within ±60 characters to count, eliminating false positives like "go to the store."

**3.2 Role aggregation.** Postings are grouped by normalized title; each role records (a) the union of its skills with per-skill posting frequency, (b) median salary, (c) top hiring companies, (d) experience-level distribution, and (e) five representative LinkedIn postings (those with the richest extracted skill content) for click-through evidence in the UI.

**3.3 TF-IDF + cosine recommender.** Each role is encoded as a "document" of canonical skill tokens, repeated in proportion to the fraction of the role's postings that mention the skill (capped at 10). `TfidfVectorizer` learns IDF weights — universal skills like *Excel* are down-weighted, distinctive skills like *Kubernetes* up-weighted. The user's skill set is encoded as a query document; cosine similarity gives the **Match Score ∈ [0, 1]** specified in the proposal. We use sublinear TF and L2 normalization.

**3.4 Skill-gap analyzer.** Given user skills `U` and a role's skills `R` with frequency map `f`, we report `matched = U ∩ R`, `missing = R \ U`, `extras = U \ R`, plus a **weighted coverage** metric `Σ_{s∈matched} f(s) / Σ_{s∈R} f(s)` that emphasizes how much of the role's *typical* requirement set the user covers — better than raw set overlap because it discounts skills that appear in only a tiny minority of postings for the role.

**3.5 Skill graph + bridge pathing.** We build two NetworkX graphs from the corpus: (a) a **skill co-occurrence graph** with 121 nodes and 3,531 edges, weighted by Normalized Pointwise Mutual Information (NPMI), and (b) a **role-transition graph** with 1,060 nodes and 236,935 directed edges. An edge `A → B` exists iff `B` requires ≤4 skills the user (in role `A`) doesn't already have *and* `|A ∩ B| / |B| ≥ 0.3` (sufficient skill overlap). Edge weight is the size of the skill jump. Dijkstra's shortest-path from a user-chosen source role to a target role yields the **bridge path** — the proposal's novel contribution.

## 4. Experiments

We evaluated five orthogonal aspects of system behavior. All numbers below are reproducible from `scripts/evaluate.py` (random seed = 42).

| Aspect | Definition | Why it matters |
|---|---|---|
| **Self-recommendation precision** | For 522 random postings, encode their skills as a user profile and check whether the posting's *true role* appears in the top-k. P@1, P@5, P@10. | Directly measures recommendation quality. Random baseline at P@10 = 10/1060 ≈ 0.9%. |
| **Skill-extractor agreement** | On 193 postings with `skills_desc` populated, treat extraction over `skills_desc` as soft ground truth and compare against extraction over `description`. P / R / F1 / Jaccard. | Measures how well the regex extractor recovers human-curated skills. |
| **Next-skill prediction** | For 1,000 postings with ≥4 skills, hide one random skill, query with the rest, and check if the held-out skill appears in the union of top-N missing skills across the top-5 recommendations. Hits@3, @5, @10. | Tests whether the recommender suggests *the right next skill to learn* given partial knowledge. Random baseline at hits@5 ≈ 5/121 = 4%. |
| **Coverage** | Over 500 randomly synthesized user profiles (3–7 skills each), what fraction of the 1,060 roles ever appears in any user's top-10? | Detects long-tail bias — does the system always recommend the same popular roles? |
| **Diversity** | Mean intra-list pairwise Jaccard *distance* between role skill-sets in the top-5 recommendations, averaged over the 500 profiles. | Tests whether each query yields varied suggestions or near-duplicates. |
| **Latency** | Wall-clock time per `recommend()` call after warm-up, over 200 queries. Mean / p50 / p95 / p99. | Establishes the system is responsive enough for live UI use. |

## 5. Results and Discussion

| Metric | Value | Random baseline | Multiplier |
|---|---|---|---|
| Self-recommendation P@1 | **26.6%** | 0.09% | 282× |
| Self-recommendation P@5 | **41.2%** | 0.47% | 87× |
| Self-recommendation P@10 | **50.0%** | 0.94% | 53× |
| Skill-extractor F1 vs `skills_desc` | **35.7%** (P=31.6 / R=41.0) | — | — |
| Next-skill hits@3 | **34.3%** | 2.5% | 14× |
| Next-skill hits@5 | **42.9%** | 4.1% | 10× |
| Next-skill hits@10 | **55.6%** | 8.3% | 7× |
| Coverage (500 profiles) | **69.8%** (740 / 1,060 roles) | — | — |
| Mean diversity (Jaccard distance) | **0.78** | 0 (identical) – 1 (disjoint) | — |
| Latency (mean) | **3.2 ms** | — | — |
| Latency (p99) | **4.0 ms** | — | — |

**Discussion.** Self-recommendation precision reaches the true role within the top-10 for half of all sampled postings — 53× a random baseline — and within the top-1 for 27%, indicating that TF-IDF over canonical-skill bags captures meaningful per-role distinctiveness despite the small (121-token) vocabulary. Next-skill prediction recovers a held-out skill in the top-5 missing recommendations 42.9% of the time across 1,000 trials (10× random), validating the explainability story: the system isn't only matching roles, it's identifying *the right gap to close*. Coverage at 69.8% indicates no extreme popularity bias — most of the role index is reachable through reasonable user queries. Mean intra-list diversity of 0.78 confirms that top-5 recommendations are substantively different, not five flavors of the same role. Latency under 4 ms p99 is comfortable for the live Streamlit demo.

The skill-extractor F1 of 0.357 against `skills_desc` is the headline weakness. Two factors contribute: (i) `skills_desc` is itself unstructured free text and contains skills outside our 121-token vocabulary (HR, project management jargon, domain-specific terms), so a "miss" is often a vocabulary-coverage issue rather than an extraction failure; (ii) `description` text and `skills_desc` text don't fully overlap in real postings — recruiters list skills in `skills_desc` that they don't repeat in the prose. Mean recall (41%) exceeds mean precision (32%), which means we more often *miss* skills than *hallucinate* them — the desirable failure mode.

**Qualitative validation.** Three illustrative profiles produce intuitive top recommendations: `{Python, Pandas, scikit-learn, NumPy, Statistics, Machine Learning}` → Quantitative Researcher (66% match) and Senior Data Scientist; `{JavaScript, React, HTML, CSS, TypeScript}` → Javascript Developer (64%), User Interface Architect, Frontend Engineer, React Developer; `{Docker, Kubernetes, AWS, Linux, Bash, Terraform}` → Senior System Administrator (66%), Azure Cloud Engineer, AWS Architect. A bridge path from a Data-Analyst-aligned profile `{Python, SQL, Excel, Statistics}` to a `Machine Learning Engineer` target traverses two intermediate roles, with each hop's required new skills surfaced as concrete learning targets.

**Limitations and future work.** (a) The vocabulary is technology-leaning by design; non-tech roles are recommended poorly. Plugging in the O*NET skills taxonomy (mentioned as supplementary in the proposal) is the natural extension. (b) Aggregating by raw normalized title can fragment near-duplicate categories ("Data Scientist" vs "Sr Data Scientist"); a clustering pass over role embeddings would consolidate them. (c) The role-transition graph uses fixed thresholds (`max_jump=4`, `overlap≥0.3`); making these adaptive per-domain would tighten bridge paths.

---

### Honesty undertaking

Per the project proposal, the team affirms that all code in this repository was written for this project. Algorithms used (TF-IDF, cosine similarity, set operations, Dijkstra shortest-path, NPMI) come from `scikit-learn` and `NetworkX` as the proposal explicitly permits. No copy-pasted implementations from third-party recommender libraries were used, and no part of the dataset, code, or report was fabricated. The deliverable system is fully reproducible from the public LinkedIn dataset and the source files in this repository.

### Reproducibility

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python scripts/build_model.py    # 1× preprocessing, ~7 min
python scripts/evaluate.py       # produces docs/evaluation_metrics.json
streamlit run app/streamlit_app.py
```
