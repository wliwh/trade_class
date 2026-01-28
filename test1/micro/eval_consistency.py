from jqdata import *
import pandas as pd
import random
import sys
from pathlib import Path

# Setup path to import from sibling directories
# Project root is assumed to be 3 levels up from this file (micro -> test1 -> trade_class)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from test1.Pools.pca_analysis import get_pca_source_data, calc_rolling_ar, plot_ar_analysis
except ImportError:
    # Fallback if package structure is different, try inserting parent of test1
    sys.path.append(str(current_file.parent.parent.parent))
    from test1.Pools.pca_analysis import get_pca_source_data, calc_rolling_ar, plot_ar_analysis

def eval_small_cap_consistency():
    CONFIG = {
        'INDEX_CODE': '399303.XSHE', # GuoZheng 2000 (Small/Micro Cap Proxy)
        'SAMPLE_SIZE': 200,          # Sampling to speed up calculation
        'LOOKBACK_DAYS': 400,        # Data length
        'ROLLING_WINDOW': 60,        # Rolling PCA window
        'PLOT_FILENAME': 'small_cap_consistency.png'
    }

    print(f"Starting Consistency Evaluation for {CONFIG['INDEX_CODE']}...")

    # 1. Fetch Index Constituents
    print("Fetching index constituents...")
    try:
        stocks = get_index_stocks(CONFIG['INDEX_CODE'])
        print(f"Index has {len(stocks)} constituents.")
    except Exception as e:
        print(f"Error fetching index stocks: {e}")
        return

    # 2. Random Sampling
    if len(stocks) > CONFIG['SAMPLE_SIZE']:
        sample_pool = random.sample(stocks, CONFIG['SAMPLE_SIZE'])
        print(f"Randomly sampled {CONFIG['SAMPLE_SIZE']} stocks.")
    else:
        sample_pool = stocks
        print(f"Using all {len(stocks)} stocks.")

    # 3. Get Data
    print(f"Fetching price data (Lookback: {CONFIG['LOOKBACK_DAYS']} days)...")
    try:
        returns = get_pca_source_data(sample_pool, count=CONFIG['LOOKBACK_DAYS'])
        print(f"Data retrieved. Shape: {returns.shape}")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if returns.empty:
        print("No returns data available.")
        return

    # 4. Calculate Rolling AR(1)
    # AR(1) measures the dominance of the first principal component (Market Factor)
    print(f"Calculating Rolling Absorption Ratio (AR(1)) with window={CONFIG['ROLLING_WINDOW']}...")
    ar_df = calc_rolling_ar(returns, window=CONFIG['ROLLING_WINDOW'], k=1)
    
    if ar_df.empty:
        print("Calculation yielded no results.")
        return

    # 5. Analysis & Plotting
    last_ar = ar_df['AR(1)'].iloc[-1]
    last_date = ar_df.index[-1]
    
    print("-" * 40)
    print(f"Analysis complete. Latest Date: {last_date}")
    print(f"Latest AR(1): {last_ar:.2%}")
    print("-" * 40)
    
    if last_ar > 0.75:
        print("[WARNING] High Consistency! Market is driving most variance (Risk On/Off or Beta driven).")
    elif last_ar < 0.40:
        print("[INFO] Low Consistency. Market is fragmented (Alpha driven or Sector rotation).")
    else:
        print("[INFO] Moderate Consistency.")

    save_path = str(current_file.parent / CONFIG['PLOT_FILENAME'])
    plot_ar_analysis(ar_df, k=1, rolling_window=CONFIG['ROLLING_WINDOW'], save_path=save_path, thresholds=[0.75, 0.40])
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    eval_small_cap_consistency()
