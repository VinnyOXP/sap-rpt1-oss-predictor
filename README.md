# SAP-RPT-1-OSS Predictor Skill

A Claude skill for using SAP's open source **SAP-RPT-1-OSS** tabular foundation model for predictive analytics on SAP business data.

## Overview

SAP-RPT-1-OSS is SAP's open source (Apache 2.0) tabular foundation model announced at TechEd 2025. Unlike LLMs that predict text, RPT-1 predicts field values in table rows using in-context learning—no model training required.

- **Repository**: https://github.com/SAP-samples/sap-rpt-1-oss
- **Model**: https://huggingface.co/SAP/sap-rpt-1-oss

This skill enables Claude to:
- Set up and authenticate with Hugging Face for model access
- Prepare SAP data extracts for prediction
- Run classification and regression using the local OSS model
- Handle batch processing for large datasets
- Optionally use RPT Playground API as alternative

## Use Cases

| SAP Module | Prediction Type | Example |
|------------|-----------------|---------|
| FI-AR | Payment Default Risk | Predict which invoices will go unpaid |
| SD | Customer Churn | Identify at-risk customers |
| SD/LE | Delivery Delays | Forecast shipping delays |
| FI-GL | Journal Anomalies | Detect unusual postings |
| MM | Vendor Performance | Score supplier reliability |
| PP/MM | Demand Forecast | Predict future quantities |

## Structure

```
sap-rpt1-oss-predictor/
├── SKILL.md                     # Main skill instructions
├── scripts/
│   ├── rpt1_oss_predict.py      # Local OSS model wrapper
│   ├── prepare_sap_data.py      # SAP data extraction utilities
│   ├── batch_predict.py         # Batch processing for large datasets
│   └── rpt1_api.py              # Optional: RPT Playground API client
├── references/
│   ├── sap-use-cases.md         # Detailed SAP prediction scenarios
│   └── api-reference.md         # Complete API documentation
└── examples/
    ├── customer_churn_sample.csv
    └── payment_default_sample.csv
```

## Requirements

- **Hugging Face account** (free) - for model access
- **GPU recommended**: 24-80GB VRAM for optimal performance
- Python 3.11+ with pandas, torch

## Quick Start

```bash
# Install model
pip install git+https://github.com/SAP-samples/sap-rpt-1-oss

# Authenticate with Hugging Face
huggingface-cli login
```

```python
from sap_rpt_oss import SAP_RPT_OSS_Classifier

clf = SAP_RPT_OSS_Classifier(max_context_size=4096, bagging=4)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

## How to Use This Skill

### Installation Options

**Option 1 — Claude Code (CLI):**
```bash
git clone https://github.com/amitlals/sap-rpt1-oss-predictor
cd sap-rpt1-oss-predictor
claude  # Skill auto-detected via .claude-plugin/marketplace.json
```

**Option 2 — Claude.ai (Pro/Team only):**
1. Go to [claude.ai](https://claude.ai) → Projects (left sidebar)
2. Create new project → Add to Project Knowledge → Upload `SKILL.md`
3. Start chatting in that project

**Option 3 — Claude.ai (Free tier):**
1. Copy contents of `SKILL.md` 
2. Paste at the start of your conversation as context
3. Then ask your prediction questions

**Option 4 — GitHub Copilot:**
- Clone repo, skill available in `.github/skills/` directory

### Example Prompts

Once installed, prompt Claude with:

**Setup:**
```
Set up SAP-RPT-1-OSS for predictions on my SAP data
```

**Classification:**
```
Predict which customers will churn using SAP-RPT-1-OSS
```

```
Classify payment default risk for these SAP invoices
```

**Forecasting:**
```
Forecast demand for next quarter using my SAP sales data
```

**Data Preparation:**
```
Help me extract SAP FI-AR data for payment prediction
```

**Batch Processing:**
```
Run batch predictions on 50,000 SAP records using RPT-1
```

## Related Resources

- [SAP-RPT-1-OSS GitHub](https://github.com/SAP-samples/sap-rpt-1-oss)
- [Model on Hugging Face](https://huggingface.co/SAP/sap-rpt-1-oss)
- [ConTextTab Research Paper](https://arxiv.org/abs/2506.10707)

## License

Apache 2.0
