# Financial Geometric Algebra Portfolio Analysis

## Architecture
This project leverages Geometric Algebra (GA) to analyze multi-asset portfolios. It uses GPU-accelerated tensor operations to model market dynamics as geometric transformations (rotations, dilations).

### Core Components
1.  **Data Pipeline (`src/pipeline`)**: 
    - **`fetch_data.py`**: Fetches raw market data using **OpenBB**.
    - **`preprocess.py`**: implements feature engineering from `multi_asset_regime_analysis.ipynb`.
        - Calculates **Log Returns**, **Volatility (20d)**, **Momentum (10d)**, and **Volume Ratio**.
        - Normalizes indices and aligns data for tensor ingestion.
2.  **Geometry Engine (`src/geometry`)**: 
    - **`tensor_ga.py`**: PyTorch-based GA engine for **Cl(4,0) Euclidean Space**.
        - **Full Geometric Product**: Implements Cayley table-based tensor multiplication for the 16-dimensional algebra.
        - **Rotor Estimation**: Calculates exact rotors $R = \frac{1+vu}{|1+vu|}$ to model regime shifts between time steps.
        - **Embedding**: Maps features to vector basis $e_1, e_2, e_3, e_4$.
3.  **Models (`src/models`)**:
    - GA-based neural networks or analytical solvers for portfolio optimization.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run data fetcher: `python src/pipeline/fetch_data.py`
3. Process features: `python src/pipeline/preprocess.py`
4. Run tests: `python tests/test_tensor_ga.py`
