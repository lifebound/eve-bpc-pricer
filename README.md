# EVE Online Blueprint Copy (BPC) Valuation Tool

A Python tool for determining fair market value of EVE Online Blueprint Copies using Adam4EVE's historical pricing data.

## Overview

This tool helps EVE Online players determine the fair market value of Blueprint Copies (BPCs) by analyzing historical pricing data from the Adam4EVE API. Instead of calculating manufacturing profitability, it focuses on what BPCs are actually worth as tradeable items.

## Why This Tool Exists

Blueprint copies are one of the most opaque markets in EVE Online. Unlike normal
items, BPCs never appear on the regional market—only on **contracts**. This
means:

- There is **no in-game price history** for BPCs.
- You can only see **current contracts**, not what players actually paid.
- Historical contract prices exist only through third-party services like
  Adam4EVE, and even then the data is:
  - sparse,
  - irregular,
  - split across many ME/TE/run combinations,
  - and based solely on *completed* contracts.

As a result, most players price BPCs by copying the lowest active contract or
just guessing, which often leads to wildly inaccurate values.

This tool attempts to fix that.

It uses the historical record of **actual completed contracts** for each
specific blueprint—each sale, at each price—to approximate what the EVE
playerbase has collectively demonstrated they are willing to pay. With this
approach, you can:

- **Value BPCs from exploration loot** without guesswork
- **Evaluate bulk contracts** by checking each included blueprint individually
- **Price your own BPC listings** based on demonstrated market behavior
- **Estimate the value of your BPC portfolio** in a consistent, explainable way


## Features

- Fetch historical pricing data for BPCs from Adam4EVE
- Calculate fair market value based on historical trends
- Support for single BPC analysis or batch processing from CSV
- Multiple valuation methods (median, average, weighted)
- Regional market analysis for BPC values

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python main.py`

## Usage

### Single BPC Valuation
```bash
python main.py --bpc "Raven Blueprint Copy" --me 10 --te 20 --runs 10
```

### Batch Processing from CSV
```bash
# Basic batch analysis (shows top 10 in terminal)
python main.py --csv blueprints.csv

# Save condensed results to CSV
python main.py --csv blueprints.csv --output results.csv

# Generate comprehensive CSV with ALL items (including failed ones)
python main.py --csv blueprints.csv --output-csv comprehensive_results.csv
```

### CSV Format
```csv
bpc_name,me_level,te_level,runs,notes
Raven Blueprint,10,20,10,"Perfect BPC"
Dominix Blueprint,9,18,1,"Nearly perfect"
```
you can also pass a 'simple' list, as such:

```csv
Raven Blueprint
Dominix Blueprint
```

## Understanding Output

### Single BPC Analysis

When analyzing a single BPC with `--test-api`, you'll see output like:

```
1. Searching for item ID...
   Found item ID: 23914

2. Fetching historical price data...
   Using efficiency matching: 9/16/1 -> 9/16/1 (column 5)
   Found 29 price data points

3. Price Deviation Analysis:
   Comparing price: 250,000,000 ISK
   vs 10-day EWA:
   Deviation: -7,663,270 ISK (-3.0%)
   ⚪ Within normal range
```

**Key Elements:**
- **Item ID**: EVE Online database ID for the BPC
- **Efficiency Matching**: Shows which efficiency column was used (if specified)
- **Data Points**: Number of historical price records found
- **Price Deviation**: How your price compares to market trends
  - 🔴 **Overpriced**: >20% above market trend
  - 🟢 **Underpriced**: >20% below market trend  
  - ⚪ **Normal Range**: Within ±20% of trend

### Batch Analysis Results

For CSV analysis, the output shows a ranked summary:

```
=== Batch BPC Valuation Results ===
Analyzed 27 BPCs

Total Portfolio Value: 2,450,123,456 ISK
Average BPC Value: 90,745,313 ISK

Top 10 Most Valuable BPCs:
Rank BPC Name                             Value (ISK)     Confidence   Efficiency
--------------------------------------------------------------------------------
1    Nyx Blueprint                          318,687,180 High         ME0/TE0
2    Revelation Blueprint                    31,981,919 Medium       ME0/TE0
3    'Chivalry' Large Remote Capacitor       19,596,476 Medium       ME0/TE0
```

**Column Explanations:**

- **Rank**: Sorted by fair market value (highest to lowest)
- **BPC Name**: Blueprint Copy name as found in EVE database
- **Value (ISK)**: Fair market value in ISK based on historical data
- **Confidence**: Reliability of the valuation (see below)
- **Efficiency**: ME/TE/Runs format showing the BPC specifications

### Confidence Levels

The confidence system evaluates how reliable each valuation is:

#### 🟢 **High Confidence (Score: 80-100)**
- **Requirements**: 15+ data points AND low price volatility (<20%)
- **Meaning**: Very reliable valuation with consistent pricing history
- **Action**: Safe to use this value for trading decisions

#### 🟡 **Medium Confidence (Score: 50-79)**  
- **Requirements**: 10+ data points OR moderate volatility (20-50%)
- **Meaning**: Reasonably reliable but some uncertainty exists
- **Action**: Good baseline, consider recent market conditions

#### 🔴 **Low Confidence (Score: 0-49)**
- **Requirements**: <10 data points AND/OR high volatility (>50%)
- **Meaning**: Limited data or very volatile pricing
- **Action**: Use with caution, verify with current market research

### Efficiency Matching

When you specify BPC efficiency (ME/TE/Runs), the tool finds the closest matching column:

```
Target efficiency: 8/15/1
Matched efficiency: 9/14/1 (column 6)
Target equivalent runs: 1.28
Matched equivalent runs: 1.28
```

- **Target**: Your BPC's efficiency specifications
- **Matched**: Closest available efficiency column in historical data
- **Equivalent Runs**: Mathematical efficiency comparison using formula `N = C/((1-A/100)(1-B/100))`

### Price Valuation Methods

The tool uses **Exponential Weighted Average** by default, which:
- Weights recent prices more heavily than older ones
- Uses configurable half-life periods (5-day and 10-day)
- Provides better trend analysis than simple averages
- Adapts to changing market conditions

Other methods available via `--method` flag:
- `median`: Middle value of all prices
- `mean`: Simple average of all prices  
- `conservative`: Lower percentile for conservative estimates

### CSV Output Formats

Two types of CSV output are available:

#### Standard Output (`--output filename.csv`)
- Contains only successfully valued BPCs
- Compact format suitable for further analysis
- Includes all pricing metrics and statistics

#### Comprehensive Output (`--output-csv filename.csv`)
- Contains **ALL** BPCs from input file
- Includes failed BPCs with error messages
- Adds ranking column based on fair market value
- Adds status column (SUCCESS/FAILED)
- Perfect for complete audit and debugging

## API Documentation

This project uses the Adam4EVE API for market data:
- Base URL: https://www.adam4eve.eu/
- No API key required for basic usage
- Rate limiting applies

## Project Structure

```
eve_bpc_pricer/
├── src/
│   ├── api/
│   │   └── adam4eve_client.py
│   ├── models/
│   │   ├── bpc.py
│   │   └── valuation.py
│   ├── valuators/
│   │   └── bpc_valuator.py
│   ├── parsers/
│   │   └── csv_parser.py
│   └── utils/
│       └── helpers.py
├── data/
│   └── sample_blueprints.csv
├── tests/
├── requirements.txt
└── main.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License