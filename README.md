# Vendor Evaluation Agent

A LangGraph-based agentic pipeline that reads vendor quotations (PDF), extracts structured procurement data, flags incomplete submissions, scores vendors against configurable weighted criteria, and generates a ranked comparison report — with a live Streamlit dashboard for adjusting priorities on the fly.

Built to solve a real problem from 8+ years running a garments/textile sourcing business: comparing vendor quotations manually across emails and spreadsheets, with no consistent scoring criteria.

## Live demo

[Add your Streamlit Cloud link here once deployed]

## What it does

1. **Upload** — drop in multiple vendor quotation PDFs (one per vendor)
2. **Extract** — an LLM pulls structured fields from each document: price per unit, currency, MOQ, lead time, payment terms, quality certifications
3. **Check completeness** — vendors missing critical fields are flagged
4. **Draft clarification emails** — for any vendor with missing data, the agent drafts a ready-to-send follow-up email requesting the specific missing details
5. **Score & rank** — complete vendors are scored using a weighted formula (price, lead time, MOQ, quality), normalized relative to the other vendors in the batch
6. **Compare live** — adjust scoring weights via sliders in the Streamlit UI and watch rankings update instantly, with no re-extraction or LLM calls needed

Incomplete and complete vendors are handled in parallel — one vendor missing a field doesn't block the rest of the batch from being scored.

## Why weighted, adjustable scoring

Different procurement situations need different priorities: an urgent order should weight lead time heavily, a bulk cost-sensitive order should weight price heavily, an export order might weight quality certifications heavily. Rather than hardcoding one "correct" formula, the coach/procurement user sets weights per evaluation — the score reflects an actual business decision, not an assumption baked into the code.

Scoring itself is deterministic (pure math, normalized per-batch), not LLM-generated — this keeps it fast, reproducible, and explainable (`score_breakdown` shows exactly how each vendor's score was composed), reserving LLM calls for the parts that genuinely need language understanding: extraction and email drafting.

## Architecture

Built as a LangGraph `StateGraph` with one conditional fan-out:

```
START → load_pdf → extract_vendor_info → completeness_check
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        │ (always)                                       │ (if any vendor incomplete)
                        ▼                                                 ▼
                  score_vendors                                request_clarification
                        │                                                 │
                        ▼                                                 ▼
              rank_compare_vendors                                      END
                        │
                        ▼
               generate_report → END
```

| Node | Responsibility |
|---|---|
| `load_pdf` | Extracts raw text from each vendor's PDF via `pypdf` |
| `extract_vendor_info` | LLM structured extraction (Pydantic schema) into price, MOQ, lead time, payment terms, certifications |
| `completeness_check` | Flags vendors missing required fields |
| `request_clarification` | Drafts a follow-up email per incomplete vendor (Pydantic-validated subject/body) |
| `score_vendors` | Deterministic weighted scoring, normalized per-batch |
| `rank_compare_vendors` | Sorts vendors by score, computes score gap between top two |
| `generate_report` | Produces the final text report |

`score_vendors` and `rank_compare_vendors` are also called directly (outside the graph) from the Streamlit UI whenever weight sliders move — this avoids re-running the LLM extraction step for every weight adjustment.

## Tech stack

- **LangGraph** — StateGraph orchestration, conditional fan-out routing
- **LangChain + Groq** — LLM calls (`qwen/qwen3.6-27b`) with structured output
- **Pydantic** — schema validation for extraction and email drafting
- **pypdf** — PDF text extraction
- **Streamlit** — upload interface, live weight sliders, ranked comparison table
- **pandas** — table rendering

## Project structure

```
vendor-evaluation-agent/
├── graph.py          # LangGraph nodes, state schema, graph construction
├── main.py           # CLI entry point for running the pipeline directly
├── app.py            # Streamlit UI
├── requirements.txt  # or pyproject.toml if using uv
├── .env              # local secrets (not committed)
└── README.md
```

## Setup

**1. Clone and install dependencies (using [uv](https://github.com/astral-sh/uv)):**

```bash
git clone <your-repo-url>
cd vendor-evaluation-agent
uv init --no-workspace
uv add langgraph langchain-groq langchain-core pypdf python-dotenv pydantic streamlit pandas
```

**2. Set environment variables** — create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
COMPANY_NAME=Your Company Name
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

**3. Run locally:**

```bash
# CLI version
uv run main.py

# Streamlit dashboard
uv run streamlit run app.py
```

## Sample data

Sample vendor quotation text is included for testing completeness-check behavior — one complete quotation, one missing a required field, and one in a different format/currency to test normalization across varied inputs.

## Known limitations

- PDF extraction relies on selectable text; scanned/image-only PDFs return no text and are flagged as errors rather than OCR'd
- Extraction accuracy depends on the underlying LLM; ambiguous or inconsistently formatted quotations may require prompt tuning
- Clarification emails are drafted, not sent — sending requires wiring up an email API (e.g. Gmail) as a follow-up step
- Vendors that reply to a clarification email currently require manual re-upload and re-run rather than automatic pipeline resumption

## Author

**S. Nataraj** — GenAI Engineer · Deep Learning & AI
Tirupur, Tamil Nadu, India
📧 connectnataraj@outlook.com
🔗 [GitHub](https://github.com/connectnataraj-boop) · [LinkedIn](https://linkedin.com/in/nataraj-sb-b5a84a3b7)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).