# Sociotechnical Autonomy Analysis

Research code accompanying the manuscript  
**“The Emergence of Sociotechnical Autonomy”**  
(submitted to *Nature Communications*, 2025).

This repository contains the full analysis pipeline used to compute
all quantitative results reported in the manuscript.

---

## ⚠️ Important Note

This is **research code**, written to support empirical analysis rather
than as a production software package.  
The code prioritizes transparency and traceability over architectural
cleanliness.

---

## Requirements

- Python 3.10+
- ~200 GB disk space (for cloning large open-source repositories)
- Dependencies listed in `requirements.txt`

---

## Reproducibility Overview

Quantitative results are computed by the analysis pipeline and emitted in execution logs during runtime; values reported in the manuscript correspond to aggregated results exported as tables and figures, which serve as the canonical source for reporting.

---

## Reproduction Steps

1. Clone repositories listed in:
data/repos_config.json



2. Set local paths in `analysis.py`:
BASE_PATH = "/path/to/local/storage"

3. Run the full analysis:
python analysis.py