Copyright (c) 2026 Kaden Najarali

All rights reserved.

This repository contains experimental quantitative trading research and backtesting tools developed by the author.

This project is currently a **demonstration version** of a larger research framework. It is under active development and will undergo **significant upgrades, improvements, and structural changes** over time.

The code and strategies included here are intended for **educational and research purposes only**. They do not constitute financial advice and should not be used for live trading without further validation and testing.

Unauthorized reproduction, redistribution, or commercial use of this code without explicit permission from the author is prohibited.--

# Overview

This repository contains multiple tools used for quantitative strategy research:

- Historical market data downloading
- Strategy backtesting
- Parameter optimization
- Walk-forward testing
- Strategy performance comparison
- Trade logging and analysis

The project primarily focuses on futures markets and includes testing environments for both **NQ** and **MES** strategies.

---

# Strategies Implemented

## Opening Range Breakout (ORB)

The ORB strategy tests breakout conditions during the early session when volatility is typically highest.

The strategy evaluates how price behaves after breaking above or below the opening range and tests different filters and parameters to improve performance.

Files related to this strategy include:

- backtest_nq_orb.py
- nq_orb_backtest_results.csv
- nq_orb_confirmed_results.csv

---

## Accumulation Manipulation Distribution (AMD)

The AMD strategy is based on market structure and liquidity concepts.

The model assumes markets move through three phases:

1. Accumulation
2. Manipulation
3. Distribution

The strategy attempts to identify manipulation phases and trade the resulting directional expansion.

Files related to this strategy include:

- backtest_mes_amd_short.py
- optimize_mes_amd_short.py
- mes_amd_short_trade_log.csv

---

## Combined Strategy

The combined strategy integrates multiple trading models into a single framework.  
It attempts to combine signals from different strategies to produce a more robust system.

Files related to this model include:

- backtest_mes_combined.py
- optimize_combined_fast.py
- mes_combined_trade_log.csv

---

## Walk Forward Testing

Walk-forward testing is used to evaluate whether a strategy remains effective when applied to unseen data.

Instead of optimizing parameters on the entire dataset, the model trains on one period and tests on the next.

This helps avoid overfitting.

Files related to this process include:

- walk_forward.py
- walk_forward_results.csv
- oos_test_results.csv

---

# Data Collection

Historical data used for backtesting is downloaded using custom scripts.

These scripts retrieve market data and prepare it for strategy testing.

Examples include:

- download_data.py
- download_spy_massive.py
- download_spy_massive_full.py
- download_mes_massive.py

Large raw datasets are excluded from the repository using `.gitignore`.

---

# Project Structure

The repository contains several types of files.

Backtesting scripts:

- backtest_nq.py
- backtest_nq_orb.py
- backtest_mes_combined.py
- backtest_mes_amd_short.py

Optimization scripts:

- optimize_nq.py
- optimize_strategy.py
- optimize_combined_fast.py
- optimize_mes_full.py
- optimize_mes_shorts_only.py

Data download scripts:

- download_data.py
- download_spy_massive.py
- download_mes_massive.py

Testing and research tools:

- walk_forward.py
- compare_strategies.py
- test_polygon.py
- test_massive_futures.py

Output files such as trade logs, optimization results, and performance metrics are generated automatically during strategy runs.

---

# Installation
DM me at @Kadennaj
