# Farmer Mitra - Smart Crop Recommendation System 🌾

Farmer Mitra is an advanced agricultural decision-support platform that has evolved from a purely agronomic Machine Learning model into a highly sophisticated, market-driven deterministic engine using the proprietary **DYMRA-CCI v3 Algorithm**.

## What is this Project?

At its core, Farmer Mitra solves a critical flaw in traditional agriculture tech: *recommending crops that grow perfectly but sell for a devastating financial loss.* 

Instead of just checking soil capability, Farmer Mitra mathematically evaluates historical market arrivals, price volatility, and demand momentum to recommend crops that are not only agronomically viable but also **financially safe and profitable**.

---

## 🧠 The DYMRA-CCI v3 Algorithm

The backend recommendation engine (`/recommend` in `app.py`) uses the **Dynamic Yield Market Recommendation with Dual Soil Intelligence (DYMRA-CCI v3)** algorithm. 

It calculates a final score out of 100 based on a strict 35/35/15/10/5 distribution, operating dynamically on the user's localized district data:

### 1. Market Score (35%)
- **Profitability (40%)**: Analyzes historical average prices multiplied by market arrival volume *for the specific district selected* to ensure high local liquidity.
- **Price Stability (30%)**: Punishes crops with high price standard deviations (boom-and-bust commodities).
- **Demand Momentum (30%)**: Compares the latest 30% of market data against the historical 70% to detect surging demand trends.

### 2. Yield Score (35%)
- **Yield Productivity (50%)**: Ranks crops by total local output (Tonnes/Hectare).
- **Yield Stability (30%)**: Punishes crops that have erratic, unpredictable local harvests.
- **Yield Growth (20%)**: Compares chronological historical data to reward crops whose local production is growing year over year.

### 3. Dual-Layer Soil Intelligence (25%)
- **District Soil Suitability (15%)**: A deterministic mathematical check that compares the user's explicit inputs (`pH`, `N`, `P`, `K`) against the strict ideal requirements mapped inside `soil_crop.csv`.
- **Crop-Soil Compatibility (10%)**: Scans the localized market data to see if the crop is historically heavily traded in the selected district. If a crop has massive local volume, it serves as proof of "Historical Suitability."

### 4. Confidence Score (5%)
A systemic check that ensures the recommendation is based on solid data. It rewards crops that have high data completeness (existing across `market.csv`, `soil.csv`, and `crop_yield.csv` in the selected district).

---

## 📁 System Architecture

- **`app.py`**: The central Flask backend server. It handles RAM caching of massive datasets upon boot to ensure millisecond API latency, executes the complex DYMRA-CCI v3 loop, and hosts the data analytics pipeline.
- **`templates/index.html`**: The frontend is a React-powered Single Page Application (SPA) styled heavily with modern TailwindCSS (glass-morphism, dynamic coloring based on Confidence Score). It uses Recharts for the Analytics Modal.
- **`analytics_engine.py`**: An aggregation utility that parses horrific, inconsistent date formats from government CSVs into standardized Pandas DataFrames to generate time-series JSON payloads for the frontend charts.
- **`analyze_data.py`**: A developer utility script used to quickly parse and validate the structural integrity of the CSV datasets.

## 💾 Datasets

The project runs on an offline data architecture utilizing four core `.csv` files stored in the `/data` folder:
1. `market.csv`: 4180 rows of monthly price and volume arrivals.
2. `crop_yield.csv`: Historical production metrics (T/Ha) per district.
3. `soil_crop.csv`: The absolute ideal nutrient profiles (pH, N, P, K) for 42 different crops.
4. `soil.csv`: Regional district soil profiles.

*(Note: Because these datasets originate from different government sectors, they lack shared primary keys. `app.py` heavily utilizes fuzzy Regular Expression (Regex) logic to join data on the fly, e.g., mapping "Green Gram" to "Moong (Green Gram)").*
