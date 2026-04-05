# Liquidity Extension Pipeline (combine_final_clean → FILLED_liquidity_extended)

This document explains how the file

- `*_combine_final_clean.csv`  (**original / cleaned panel**)

is transformed into

- `*_combine_final_clean_FILLED_liquidity_extended.csv` (**same rows + added liquidity/microstructure columns**)

The goal is to **append** market microstructure proxies (HL spread, VWAP proxy, dollar volume, Kyle/Amihud impact proxies, turnover, Roll’s measure) while leaving all original columns intact.

---

## 1) Inputs and outputs

### Input (per ticker)
A daily panel CSV with columns like:

- Keys: `symbol`, `date`, `channel_classification`
- Market data (same across channels on a given date): `open`, `high`, `low`, `close`, `volume`, `adjusted`
- Sentiment/other existing variables: `v2tone1`, `amihud_ratio`, `illiq`, `r`, `volatility`, `var1pct`, `lnvol_chg`, etc.

### Output (per ticker)
The same panel, with:

- `date` standardized to ISO format: `YYYY-MM-DD`
- Any fully blank trailing rows removed (rows where `date` / `symbol` / `channel_classification` are missing)
- **20 new columns appended** (see Section 3)

---

## 2) Core design choices (important)

### A) Compute market-derived columns at the **symbol-date** level
Because the panel contains multiple `channel_classification` rows per trading day (Casual / Mixed / Professional), the extension first builds a **market-only** table:

- Unique rows by `(symbol, date)`
- Compute market-derived columns once
- Merge those columns back to every channel row for that date

This guarantees that liquidity variables are **identical across channels on the same day**.

### B) Market return used for liquidity proxies
For liquidity proxies, a **market-only log return** is computed from `adjusted`:

- `r_mkt[t] = ln(adjusted[t]) − ln(adjusted[t−1])`  (computed on the unique symbol-date series)

This avoids “multi-day jumps” caused by channel-specific missing dates.

> Note: the existing column `r` in your original data is **NOT overwritten**.  
> The pipeline computes `r_mkt` internally only for the new liquidity proxy columns.

### C) Shares outstanding is an external constant
`shares_outstanding` is treated as a constant per ticker (one value applied to all dates), with metadata columns documenting the source and as-of date.  
(These values were taken from Yahoo Finance Key Statistics in your current extended files.)

---

## 3) New columns and exact formulas

All computations are performed on the **unique symbol-date market table**, then merged back.

### 3.1 Price / spread proxies

**(1) typical_price**
- `typical_price = (high + low + close) / 3`

**(2) hl_spread**  (high–low spread proxy, relative)
- `hl_spread = (high − low) / ((high + low)/2)`
- Equivalent: `hl_spread = 2*(high − low)/(high + low)`

### 3.2 VWAP proxies

**(3) vwap_proxy**
- In daily OHLCV data (no intraday prints), we use:
- `vwap_proxy = typical_price`

**(4) vwap_cum** (cumulative VWAP proxy)
- `vwap_cum[t] = sum_{τ<=t}(vwap_proxy[τ] * volume[τ]) / sum_{τ<=t}(volume[τ])`

### 3.3 Volume / dollar volume

**(5) dollar_volume**
- `dollar_volume = close * volume`

**(6) volume_zero_flag**
- `volume_zero_flag = 1 if volume == 0 else 0`

### 3.4 Return and flags

**(7) zero_return_flag**
- Using market return `r_mkt` (log return from adjusted):
- `zero_return_flag = 1 if (r_mkt == 0) else 0`
- If `r_mkt` is missing (first date), flag is `0`.

### 3.5 Price impact proxies

Let `r_mkt` be the market log return from adjusted.

**(8) amihud_impact_proxy**
- `amihud_impact_proxy = |r_mkt| / dollar_volume`

**(9) kyle_lambda_proxy**
- `kyle_lambda_proxy = |r_mkt| / volume`
- (A lightweight Kyle-style proxy: return per share traded.)

### 3.6 Turnover

**(10) shares_outstanding**
- Constant per ticker (external), applied to all dates.

**(11) turnover_ratio**
- `turnover_ratio = volume / shares_outstanding`

Metadata columns added with the constant:
- `shares_outstanding_source` = `"Yahoo Finance Key Statistics"`
- `shares_outstanding_asof` = `"2026-01-07"`
- `shares_outstanding_confidence` = `"low"`
- `external_identity_status` = `"ticker_match_ok (SEC validation recommended)"`

### 3.7 Roll’s measure (effective spread estimator)

**(12) rolls_measure** (constant per ticker in your current files)
Compute on the **unique symbol-date adjusted price series**:

1. `Δp[t] = adjusted[t] − adjusted[t−1]`
2. `cov = Cov(Δp[t], Δp[t−1])`
3. If `cov < 0`:  
   `rolls_measure = 2 * sqrt(-cov)`  
   else: `rolls_measure = NaN`

This becomes a single scalar value per ticker (then copied to all rows/dates).

Additional placeholder/meta columns:
- `rolls_measure_rolling` = NaN
- `rolls_measure_strict` = NaN
- `rolls_measure_window` = 252
- `rolls_measure_min_periods` = 126

---

## 4) Sorting / formatting

- `date` is stored as a string: `YYYY-MM-DD`
- Final output is sorted by:
  1) `date` ascending  
  2) `channel_classification` in the order: `Casual Investors`, `Mixed Investors`, `Professional Investors`

---

## 5) Reproducibility checklist

To reproduce exactly:
1. Drop fully blank rows (missing `date`, `symbol`, or `channel_classification`)
2. Standardize `date` to ISO
3. Build unique `(symbol,date)` market table
4. Compute new columns using formulas above
5. Merge back to full panel
6. Sort and save

