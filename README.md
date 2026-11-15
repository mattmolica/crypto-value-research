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
python -m crypto_value_research.rank_tokens               # every token with revenue, by P/S
python -m crypto_value_research.rank_tokens --sort p_f --out data/ranking.csv
python -m crypto_value_research.rank_tokens --min-revenue 5e6 --universe-out data/universe.csv
python -m crypto_value_research.rank_tokens --tokens default   # the 14 hand-picked tokens
```

**Which tokens.** By default it ranks every project DefiLlama tracks that
earned at least $1M of revenue over the past year (`--min-revenue`) and has a
token on CoinGecko. Products are folded into their parent project first
(Uniswap V2, V3 and V4 count as Uniswap, which owns UNI). Chains are left out,
because their "revenue" is burned gas. `--universe-out` writes every project
with the reason it was or wasn't ranked: `ranked`, `no token` (Tether, Circle,
Polymarket and other big earners with nothing to buy), `revenue under floor`,
`token shared with a bigger project`, or `excluded` (a gold-backed token, whose
price follows gold). `python -m crypto_value_research.universe` prints just
this list, and `universe.revenue_tokens(...)` hands it to `history.build_all`.

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
The hand-picked list lives in `DEFAULT_TOKENS` in `rank_tokens.py`.

Things to know when reading the table:

- **Locked tokens.** For vote-escrow tokens (listed in `LOCKED_SUPPLY`, e.g.
  AERO, CRV), holders lock tokens to earn the fees and CoinGecko leaves the
  locked tokens out of market cap. These are valued on every issued token
  (price x total supply) instead, so their multiples aren't flattered.
- **No circulating market cap.** When CoinGecko can't verify a token's
  circulating supply it reports no market cap; those tokens are valued on fully
  diluted value instead (this can only make them look dearer) and noted.
- **Caution column.** `True` when the data makes the multiple hard to trust:
  the token isn't on CoinGecko, no revenue in the window, market value under
  $10M, less revenue history than the window, or revenue larger than the whole
  market value (usually a wrong token match). Filter these out for a clean list.
- **Notes column.** Explains each caution, and also flags tokens valued on
  issued supply or FDV, tokens that paid holders in the past year but nothing
  in the last 30 days (a paused buyback, for example), and tokens whose revenue
  run rate has fallen by more than half against the past year, since cheap can
  mean shrinking.

## Value score (cheap for a good reason, or a value trap?)

```bash
python -m crypto_value_research.rank_tokens --out data/ranking.csv
python -m crypto_value_research.score --ranking data/ranking.csv --out data/scores.csv
python -m crypto_value_research.score --ranking data/ranking.csv --no-supply-growth  # faster
```

A low P/S alone can't tell a bargain from a project that's shrinking. The score
(0-100, higher is better) puts cheapness beside business health and dilution:

| Part | Metric | Weight |
|---|---|---|
| Cheap (40) | P/S on the last 90 days (lower is better) | 25 |
| | Holder yield: payouts to holders / market value | 15 |
| Healthy (40) | Revenue trend: last 90 days vs last year | 20 |
| | Share of revenue that reaches holders | 10 |
| | Size of revenue | 10 |
| Low dilution (20) | Share of total supply already circulating | 10 |
| | Circulating supply growth over the past year (lower is better) | 10 |

Each metric is a percentile among the rated tokens (100 = best of the group,
50 = middle, unknown counts as 50), so there are no hand-tuned cutoffs to
overfit. Rows the ranking marks `caution` are **Unrated** and don't affect the
others. If holder payouts stopped in the last 30 days, the holder metrics count
as zero. Supply growth is circulating market cap / price a year ago vs today,
from CoinGecko's free one-year chart (one request per token, ~2.5s each).

The `verdict` column reads the Cheap and Healthy parts together: **Value
candidate** (both 60+), **Value trap risk** (cheap 60+, healthy under 40),
**Good, priced in** (healthy 60+, not cheap), **Avoid** (healthy under 40, not
cheap), **Mixed** (the rest). Weights live in `WEIGHTS` in `score.py`. The
score is not back-tested against returns; treat it as a screen for what to
research, not a buy signal.

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

## Track record: do the scores work?

Two checks, because a score is only worth something if it picks tokens that
then do better than the alternatives.

**Forward tracking** (the real test, it needs time):

```bash
python -m crypto_value_research.score --ranking data/ranking.csv --snapshot   # save a dated snapshot
python -m crypto_value_research.track save --scores data/scores.csv --scored-at 2026-11-13T18:55:00Z
python -m crypto_value_research.track            # measure every snapshot in snapshots/
```

Each score run saved in `snapshots/` is frozen. `track` prices every token at
the snapshot time and now (DefiLlama coins API) and reports the return of each
label and each fifth of the score, next to Bitcoin, Ether and an equal-weight
basket of every scored token. It also gives the rank correlation between score
and return (+1: higher score always did better; 0: the score told us nothing).
Writes per-token and per-group CSVs, day-by-day paths for charting, and
`data/tracking/tracking.json`. Price only: holder payouts are not added back.
A few days of returns are noise; judge it after months and several snapshots.

**Backtest** (what history can say, with big caveats):

```bash
python -m crypto_value_research.backtest --history data/history/full_history.csv --out data/backtest.json
```

The full value score can't be rebuilt for past dates (holder payouts, supply
growth and circulating supply have no free history), so this tests P/S alone
on the tokens `history` covers: did the cheapest third beat the most expensive
third over the next 13, 26 and 52 weeks; a growth-of-$1 line rebalanced every
13 weeks, rerun with each token left out to see if one token carries the
result; and whether tokens cheap or expensive against their own past then beat
the average token. Read it with the limits in mind: 6 to 14 tokens, a few
years, survivorship (today's winners), FDV estimated for tokens whose supply
moved, and overlapping windows (the output counts the truly separate periods).

## Automatic updates

`.github/workflows/update.yml` runs on GitHub's servers with no one at the keyboard:

- **Every day** it re-prices every saved snapshot (`track`) and rewrites
  `results/README.md` (tables GitHub shows in the repo) and `site/data/`.
- **Every Monday** it first re-ranks and re-scores every token and saves a new
  dated snapshot in `snapshots/`, so the track record grows week by week.
- It commits those files back to `main`, then publishes `site/` (the charts
  page) to GitHub Pages.

To run it now: Actions tab > "update track record" > Run workflow (tick
"rescore" to also re-score). For the page to go live, the repository must be
public and Settings > Pages > Source must be set to "GitHub Actions".
The backtest (`site/data/backtest.json`) is not re-run by the schedule, since
it needs the slow full-history download; rebuild it by hand when wanted.

## Status

Early stage: a valuation ranking over every token with revenue and a value score that separates cheap-and-healthy from value traps are in; a track record (forward tracking of saved score snapshots, plus a P/S backtest) is in; a regularly updated public page is next.
