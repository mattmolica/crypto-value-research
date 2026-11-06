# Crypto Value Research

Research tools for applying value-investing ideas to crypto assets.

## Why

Most crypto analysis is about price momentum and narratives. This project asks a
different question: **what is a token actually worth, based on what the network
does?** It borrows the habits of value investing (look at fundamentals, compare
price to what you get, and insist on a margin of safety) and applies them to
on-chain and market data.

Examples of the metrics it is meant to build and track:

- **Usage and revenue:** active addresses, transaction counts, fees paid to the
  protocol, and revenue that reaches token holders.
- **Valuation ratios:** market cap or fully diluted value against fees and
  revenue (crypto's version of price-to-sales and price-to-earnings), and
  NVT (network value to transactions).
- **Supply quality:** inflation, unlock schedules, and holder concentration,
  which dilute value the way share issuance does for stocks.
- **Screens:** ranking assets on these measures to find ones that look cheap
  relative to their fundamentals, and flagging ones that only look cheap.

## Layout

```
src/crypto_value_research/   the Python package (loaders, metrics, screens)
notebooks/                   exploratory research notebooks
data/                        local data downloads (git-ignored)
tests/                       pytest tests
```

## Setup

Needs Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[notebooks,bigquery,dev]"
pytest
```

`requirements.txt` lists the same direct dependencies if you prefer
`pip install -r requirements.txt`.

## Data sources

- Public market and on-chain APIs, fetched with `requests` / `httpx`.
- Google BigQuery public crypto datasets (`bigquery-public-data.crypto_*`).
  These tables are large, so every query should be dry-run first and run with a
  `maximum_bytes_billed` cap. Never commit credentials; authenticate with
  `gcloud auth application-default login` or an environment-provided token.

## Status

Early scaffold. Analysis code and notebooks come next.
