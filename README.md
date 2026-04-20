# LM1: Do Large Language Models Develop Dependency Grammar?

**Evidence from Attention-Derived Prefix Trees**

> Final Project — Computational Linguistics (CSG), April 2026

## Authors

- Jani Ravi Kailash
- Aditya Panwar
- Ishan Trikha
- Sandeep Kumar Gupta

## Overview

This project investigates whether transformer-based LLMs implicitly learn **dependency grammar** by analyzing the structural stability of attention-derived dependency trees across growing sentence prefixes.

We extract layer-wise attention matrices from **mBERT** (bert-base-multilingual-cased), construct dependency trees using the **Chu-Liu/Edmonds** maximum spanning arborescence algorithm, and measure **Incremental Edge Change (IEC)** — the fraction of head assignments that change when a single token is appended to a prefix.

### Key Findings

- ✅ **H₁ (Structural Stability):** LLM attention trees stabilise as prefix length grows (IEC decreases monotonically).
- ✅ **H₂ (Non-Randomness):** LLM IEC is significantly lower than random baselines (p < 0.001, Cohen's d ≈ 1.2–1.8).
- Middle BERT layers (5–8) show maximum stability, consistent with prior work on syntactic specialisation.
- The pattern holds across **English, Hindi, German, and French**.

## Repository Structure

```
├── pipeline.py              # Complete analysis pipeline (single-file)
├── requirements.txt         # Python dependencies
├── LM1_Final_Report.md      # Full academic report
├── results/                 # Generated figures
│   ├── fig1_stability_curves.png
│   ├── fig2_language_comparison.png
│   ├── fig3_layer_analysis.png
│   └── fig4_depth_change.png
├── Project-dependency-grammar-built-by-LLMs.pdf    # Reference paper
├── PROPOSAL_SANDEEP_KUMAR_GUPTA_240928.pdf         # Project proposal
└── Voita-et-al.pdf                                 # Reference paper
```

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Sandeepgupta-24/LM1-dependency-grammar.git
cd LM1-dependency-grammar

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python pipeline.py                     # Full run (100 sentences/language)
python pipeline.py --dry_run           # Quick test (5 sentences/language)
python pipeline.py --max_sentences 200 # Larger sample
```

UD treebank data is downloaded automatically on first run and cached locally.

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| **1. Data Loading** | Downloads UD treebanks (English EWT, Hindi HDTB, German GSD, French GSD) from GitHub |
| **2. Attention Extraction** | Extracts word-level attention matrices from mBERT for every prefix S₂, S₃, …, Sₙ |
| **3. Tree Construction** | Builds dependency trees via Chu-Liu/Edmonds maximum spanning arborescence |
| **4. Stability Evaluation** | Computes IEC, tree depth change, UAS, gold-tree baseline, random baseline |
| **5. Statistical Testing** | Paired t-test with Cohen's d effect size |
| **6. Visualisation** | Generates four publication-quality figures |

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- See `requirements.txt` for full list

## References

- Clark et al. (2019). *What Does BERT Look At? An Analysis of BERT's Attention.* BlackboxNLP.
- Hewitt & Manning (2019). *A Structural Probe for Finding Syntax in Word Representations.* NAACL.
- Voita et al. (2019). *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting.* ACL.
- Chu & Liu (1965). *On the Shortest Arborescence of a Directed Graph.*

## License

This project is for academic purposes (CSG course project).
