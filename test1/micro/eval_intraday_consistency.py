from jqdata import *
import sys
from pathlib import Path
import datetime

# Setup path to import from sibling directories
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from test1.Pools.pca_analysis import calc_intraday_daily_ar, plot_ar_analysis
except ImportError:
    sys.path.append(str(current_file.parent.parent.parent))
    from test1.Pools.pca_analysis import calc_intraday_daily_ar, plot_ar_analysis

def eval_intraday_consistency():
    CONFIG = {
        'INDICES': {
            '000016.XSHG': 'SZ50',      # Super Large / Finance
            '000300.XSHG': 'HS300',     # Core Assets
            '000905.XSHG': 'ZZ500',     # Mid Cap
            '000852.XSHG': 'ZZ1000',    # Small Cap
            '399006.XSHE': 'CYB',       # Growth / Tech
            '000688.XSHG': 'KC50'       # Hard Tech
        },
        'FREQUENCY': '5m',
        'DAYS_TO_ANALYZE': 100,
        'PLOT_FILENAME': 'intraday_consistency_trend.png'
    }

    print(f"Starting Intraday Consistency Analysis (Last {CONFIG['DAYS_TO_ANALYZE']} days)...")
    
    # 1. Calculate Intraday AR
    res_df = calc_intraday_daily_ar(CONFIG['INDICES'], count=CONFIG['DAYS_TO_ANALYZE'], frequency=CONFIG['FREQUENCY'])
    
    if res_df.empty:
        print("No results generated.")
        return

    # 2. Plotting
    save_path = str(current_file.parent / CONFIG['PLOT_FILENAME'])
    plot_ar_analysis(res_df, k=1, rolling_window=None, 
                     save_path=save_path, 
                     ma_window=5, 
                     thresholds=[0.75, 0.40])
    
    print(f"Latest Smoothed AR(1): {res_df['AR(1)'].rolling(5).mean().iloc[-1]:.2%}")

if __name__ == "__main__":
    eval_intraday_consistency()
