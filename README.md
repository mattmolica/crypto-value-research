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

## Token valuation ranking

```bash
python -m crypto_value_research.rank_tokens               # rank by P/S
python -m crypto_value_research.rank_tokens --sort p_f --out data/ranking.csv
```

Pulls daily fees, revenue and holders revenue from DefiLlama and market cap,
FDV and supply from CoinGecko, totals the flows over the trailing 90 days
(change with `--window 30|90|365`), scales them to a year and ranks tokens
cheapest first. The trailing year's P/S (`p_s_1y`) is shown beside it as a
check. Multiples:

- **P/F**: market cap / annualized fees (everything users pay)
- **P/S**: market cap / annualized protocol revenue
- **FDV/Rev**: fully diluted value / annualized revenue (prices in future unlocks)
- **P/E (holders)**: market cap / annualized revenue that reaches token holders
- **MCap/FDV**: share of the diluted supply already circulating

Set `COINGECKO_API_KEY` to use a CoinGecko Demo key for higher rate limits.
The token list lives in `DEFAULT_TOKENS` in `rank_tokens.py`.

Two things to know when reading the table:

- **Locked tokens.** For vote-escrow tokens (listed in `LOCKED_SUPPLY`, e.g.
  AERO, CRV), holders lock tokens to earn the fees and CoinGecko leaves the
  locked tokens out of market cap. These are valued on every issued token
  (price x total supply) instead, so their multiples aren't flattered.
- **Notes column.** Flags tokens valued that way, and tokens that paid holders
  in the past year but paid nothing in the last 30 days (a buyback that was
  paused, for example).

## Valuation history

```bash
python -m crypto_value_research.history                   # writes data/history/*.csv
python -m crypto_value_research.history --tokens aave gmx aerodrome=aerodrome-finance
python -m crypto_value_research.history --tokens-file tokens.txt --no-market-cap
```

It runs on any list of tokens. Name each by its DefiLlama protocol slug, adding
`=coingecko-id` only when DefiLlama doesn't know the CoinGecko id. A token file
has one per line (`#` starts a comment). With no list it uses the ranking's
`DEFAULT_TOKENS`. Tokens that can't be built (no price history, no supply, a
failed download) are skipped with a warning and the rest carry on. For long
lists, `--no-market-cap` skips the slow one-year CoinGecko download per token.
From Python, `history.build_all({"aave": None, ...})` returns the same two tables.

Shows how expensive each token has been over its whole life, so "cheap" can
be judged against the token's own past. It writes a weekly history per token
(`full_history.csv`) and a summary (`full_summary.csv`) with today's P/S, the
lifetime low, median and high, and `cheaper_than_pct`: the share of past days
with a higher P/S than today (0.96 means it has rarely been this cheap).

Free data has no record of past circulating supply, and CoinGecko's free API
stops at 365 days of market cap. So the long history uses **fully diluted
value**: daily price from DefiLlama's coins API (back to launch) times total
supply. Supply counts as exact only when a token's total supply has reached its
fixed maximum; it is modeled for CRV from its emission schedule, estimated with
a stated lean for hand-checked tokens whose supply changed (`SUPPLY_RULES` in
`history.py`), and marked "estimated" for every other token, so a long list
never claims more accuracy than it has (see `supply_method` in the summary).
The last year of circulating market cap is kept beside it as `ps_mcap`.

Needs `coins.llama.fi` reachable in addition to the hosts the ranking uses.

## Status

Early stage: a first valuation ranking is in; more metrics and screens next.
