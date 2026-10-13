# Getting started

RISK//RING is a reproducible financial-crime network experiment. It generates a transaction graph with injected typologies, trains a classifier against known synthetic ground truth, explains flagged transactions with SHAP, and searches the resulting graph for collusion communities.

## Explore the result

Open the [live report](https://jeevan-0508.github.io/risk-ring/). The published page is a static presentation of the checked-in analysis outputs. Metrics describe the configured synthetic run; they are not claims about a live bank, customer population or production detection rate.

## Reproduce locally

Create a Python environment and install the pinned analysis dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the repository's analysis entry points as described in the README, then run the test suite:

```bash
pytest
```

The pipeline keeps feature construction causal and evaluates against injected ground truth. Review the seed and configuration before comparing two runs.

## Interpret carefully

Precision, recall, PR-AUC, SHAP values and community scores are measurements of this synthetic experiment. They support transparent reasoning about the method, not a deployment recommendation or a guarantee of performance on real financial data.
