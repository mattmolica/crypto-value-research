# What do the tokens that beat Bitcoin have in common?

Run 2026-11-17 with `python -m crypto_value_research.winners` on the 153 revenue
tokens (`history --no-market-cap` output), tokens worth $10M+ (FDV) at the start
of each window. Raw numbers: [winners.json](winners.json).

## Short answer

At the start of a window, the tokens that went on to beat BTC looked almost
exactly like the ones that didn't. Of 12 things we could measure, only one
held up in every check: **tokens trading pricey compared with their own past
P/S tended to fall behind**. It is a modest edge, not a way to pick winners.

## Setup

- Cut 2021 to 2026 into windows that do not overlap: 21 windows of 13 weeks,
  10 windows of 26 weeks. In each, every token is a *winner* if its price beat
  BTC's, else a *loser*.
- Base rate: 34% of token-windows beat BTC over 13 weeks, 24% over 26 weeks.
- Features use only data from before the window started (see the module docstring).

## 1. One feature at a time (13-week windows)

| Feature | Rank corr. with beating BTC | t | Top third beat BTC | Bottom third beat BTC |
|---|---|---|---|---|
| P/S vs own past (pricier) | **-0.15** | **-3.2** | 29% | 36% |
| Token value (FDV, bigger) | -0.09 | -2.2 | 32% | 35% |
| Revenue growth, last 52 weeks | +0.06 | 1.1 | 33% | 28% |
| P/S vs other tokens | -0.03 | -0.5 | 30% | 33% |
| Everything else (revenue size, revenue growth 13w, fee share kept, momentum 13w/52w, volatility, distance from high, age) | between -0.05 and +0.03 | under 1 | | |

P/S vs own past also held over 26 weeks (-0.20, t -2.4), kept the same sign for
every possible window start date, and stayed between -0.14 and -0.17 when any
single token was dropped. Nothing else passed all of those checks. With 12
features and 2 horizons tested, a t of around 2 can show up by luck, so
"smaller tokens" and "growing revenue over a year" are hints at most.

## 2. A model, tested only on the future

Logistic regression, trained only on windows that had already ended and tested
on the next one (17 test windows at 13 weeks):

| Model | AUC (0.5 = coin flip) | Top third beat BTC | All tokens beat BTC |
|---|---|---|---|
| All 12 features | 0.52 | 35% | 33% |
| P/S vs own past only | **0.56** (above 0.5 in 82% of windows) | 38% | 33% |
| Own past + size + 52w revenue growth | 0.55 | 35% | 33% |

More features made it worse: the full model learns noise. The one-feature
model's edge is real but small.

## 3. What the biggest winners looked like

The top 10% of each window (e.g. AERO x17 vs BTC in early 2024, THE, CARDS,
RAY, AXS in 2021) had feature ranks near the middle on everything. Many were
too new to have any history to judge. New tokens as a group did *not* beat BTC
more often (their median was worse), so the big wins look like lottery
tickets rather than something visible beforehand.

## 4. Timing

Whether alts beat BTC at all is mostly the overall market tide. Neither BTC's
own last 13 weeks nor alts' last 13 weeks vs BTC predicted it well
(correlations between -0.1 and -0.5 over only 10 to 21 windows, too few to trust).

## Limits

- Today's revenue earners only: tokens that died are missing, which flatters
  every result (especially for young tokens).
- Past FDV uses estimated supply for about half the tokens.
- Price only, no holder payouts.
- P/S vs own past was already chosen for the value score (PR #12) from an
  earlier backtest on the same data, so this is a re-check from a new angle,
  not a fully fresh test.
