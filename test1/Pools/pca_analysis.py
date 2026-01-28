from jqdata import *
import pandas as pd
import numpy as np
import datetime
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_pca_source_data(etf_list, start_date=None, end_date=None, count=100):
    """
    获取价格数据并计算收益率 (使用对数收益率)
    支持 start_date/end_date 或 count
    """
    import datetime
    
    # 1. 确定 End Date
    if end_date is None:
        end_date = datetime.date.today()
        
    # 2. 获取成交额中位数，过滤掉流动性极差的 (取截止日前的20天)
    money_df = get_price(etf_list, end_date=end_date, count=20, frequency="1d", fields=["money"], panel=False)
    if money_df.empty:
        return pd.DataFrame()
        
    money_df = money_df.pivot(index='time', columns='code', values='money')
    money_median = money_df.median()
    valid_etfs = [etf for etf in etf_list if money_median.get(etf, 0) > 0]
    
    if not valid_etfs:
        return pd.DataFrame()

    # 3. 获取收盘价
    if start_date:
        price_data = get_price(valid_etfs, start_date=start_date, end_date=end_date, frequency="1d", fields=["close"], panel=False)
    else:
        price_data = get_price(valid_etfs, end_date=end_date, count=count, frequency="1d", fields=["close"], panel=False)
        
    if price_data.empty:
        return pd.DataFrame()
        
    price_data = price_data.pivot(index='time', columns='code', values='close')
    
    # 4. 数据清洗
    # 移除缺失数据过多的列 (>10% 缺失)
    missing_ratio = price_data.isnull().mean()
    price_data = price_data.loc[:, missing_ratio < 0.1]
    
    if price_data.empty:
        return pd.DataFrame()
        
    # 填充剩余 NaNs
    price_data.fillna(method="ffill", inplace=True)
    price_data.dropna(axis=0, how='any', inplace=True) # 移除头部无法填充的行
    
    # 5. 计算对数收益率: ln(P_t / P_{t-1})
    if len(price_data) < 2:
        return pd.DataFrame()
        
    returns = np.log(price_data / price_data.shift(1)).dropna()
    return returns

def calc_absorption_ratio(returns, k=None):
    """
    执行 PCA 并计算吸收比率 (Absorption Ratio)
    """
    if returns.empty or len(returns.columns) < 2:
        return None

    pca = PCA()
    pca.fit(returns)
    
    evr = pca.explained_variance_ratio_
    cum_var = np.cumsum(evr)
    
    # 确定 K 值
    n = len(returns.columns)
    if k is None:
        k = min(max(1, int(n / 5)), len(evr))
        k = min(k, 10)
    else:
        k = min(k, len(evr))
    
    ar_val = cum_var[k-1]
    
    return {
        "ar_k": ar_val,
        "ar_1": cum_var[0],
        "k": k,
        "evr": evr,
        "cum_var": cum_var,
        "n_assets": n
    }

def calc_rolling_ar(returns, window, k=None):
    """
    滚动计算吸收比率
    """
    if len(returns) < window:
        return pd.DataFrame()

    ar_series = []
    
    for i in range(window, len(returns) + 1):
        window_data = returns.iloc[i-window : i]
        current_date = window_data.index[-1]
        
        try:
            res = calc_absorption_ratio(window_data, k=k)
            if res:
                ar_series.append({
                    "date": current_date,
                    f"AR({res['k']})": res["ar_k"],
                    "AR(1)": res["ar_1"]
                })
        except:
            continue
            
    if not ar_series:
        return pd.DataFrame()
        
    return pd.DataFrame(ar_series).set_index("date")


def get_intraday_data_for_date(targets, date_str, frequency='5m', strict=True):
    """
    获取指定日期、指定标的列表的日内分钟数据，并合并为一个 DataFrame
    targets: list of codes OR dict {code: name}
    strict: 是否要求必须包含所有标的的数据 (True用于多指数对比，False用于成分股分析)
    """
    if isinstance(targets, dict):
        codes = list(targets.keys())
    else:
        codes = list(targets)
        
    # 批量获取数据
    df = get_price(codes, start_date=date_str, end_date=date_str+' 16:00:00', frequency=frequency, fields=['close'], panel=False)
    
    if df.empty:
        return pd.DataFrame()
        
    # 重组为 wide format: index=time, columns=code
    daily_prices = df.pivot(index='time', columns='code', values='close')
    
    # 检查数据完整性
    if strict:
        # 严格模式：任一标的缺失则该日无效
        if daily_prices.shape[1] < len(codes):
            return pd.DataFrame()
        # 严格模式：任一行有空值也丢弃 (通常dropna会做，但这里强调全齐)
        daily_prices.dropna(inplace=True)
    else:
        # 宽松模式：允许部分标的缺失 (如停牌)，但为了PCA计算，必须删除含NaN的行(某个时间点某股票无数据)
        # 或者删除含NaN的列(某股票全天无数据)?
        # PCA要求矩阵完整。
        # 策略：
        # 1. Drop columns with mostly Intraday NaNs (suspended all day)
        daily_prices.dropna(axis=1, how='all', inplace=True)
        # 2. Drop rows with any NaNs (partial suspension or missing bars)
        daily_prices.dropna(axis=0, how='any', inplace=True)
        
        # 如果剩余的资产太少(比如小于2个)，无法做PCA
        if daily_prices.shape[1] < 2:
            return pd.DataFrame()

    return daily_prices


def calc_intraday_daily_ar(targets, start_date=None, end_date=None, count=100, frequency='5m', sample_n=None):
    """
    计算每日的日内 AR(1) 值
    targets: 
      - dict: {code: name} -> 多指数宏观一致性 (Strict)
      - str: Index Code (e.g. '000300.XSHG') -> 成分股一致性 (Loose)
      - list: Fixed List -> 成分股一致性 (Loose)
    sample_n: 如果 targets 是成分股，是否进行随机抽样以加速
    """
    import datetime
    import random
    from jqdata import get_trade_days, get_price, get_index_stocks
    
    now = datetime.datetime.now()
    
    # 1. 确定 End Date
    if end_date is None:
        if now.time() < datetime.time(15, 30):
            end_date = (now - datetime.timedelta(days=1)).date()
        else:
            end_date = now.date()
            
    # 2. 获取 Trade Days
    if start_date:
        trade_days = get_trade_days(start_date=start_date, end_date=end_date)
    else:
        trade_days = get_trade_days(end_date=end_date, count=count)
    
    results = []
    
    min_bars = 40 if frequency == '5m' else (200 if frequency == '1m' else 10)
    
    for date in trade_days:
        date_str = date.strftime("%Y-%m-%d")
        print(f"Processing {date_str}...", end='\r')
        
        try:
            # Determine Codes and Strict Mode
            target_codes = []
            is_strict = False
            
            if isinstance(targets, dict):
                target_codes = list(targets.keys())
                is_strict = True # 多指数对比要求数据对齐
            elif isinstance(targets, str):
                # 动态获取指数成分股
                target_codes = get_index_stocks(targets, date=date)
                is_strict = False
            elif isinstance(targets, list):
                target_codes = targets
                is_strict = False
                
            if not target_codes:
                continue
                
            # Sampling
            if sample_n and len(target_codes) > sample_n:
                target_codes = random.sample(target_codes, sample_n)
            
            daily_prices = get_intraday_data_for_date(target_codes, date_str, frequency, strict=is_strict)
            
            if len(daily_prices) < min_bars:
                continue
                
            # 计算对数收益率
            daily_returns = np.log(daily_prices / daily_prices.shift(1)).dropna()
            
            # Intraday AR(1)
            res = calc_absorption_ratio(daily_returns, k=1)
            
            if res:
                results.append({
                    'date': date,
                    'AR(1)': res['ar_1'],
                    'bars': len(daily_prices),
                    'n_assets': len(daily_prices.columns)
                })
                
        except Exception as e:
            print(f"\nError processing {date_str}: {e}")
            continue
            
    print(f"\nProcessing complete. Days with valid data: {len(results)}")
    
    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results).set_index('date')
    return res_df


def plot_ar_analysis(ar_df, k, rolling_window=None, overlay_code=None, save_path="ar_rolling_plot.png", ma_window=None, thresholds=None):
    """
    绘制 AR 分析图表
    ma_window: 如果提供，绘制移动平均线
    thresholds: 如果提供列表 [0.75, 0.40]，绘制阈值线
    """
    import matplotlib.dates as mdates
    
    if ar_df.empty:
        print("  -> No data to plot.")
        return

    # 确保索引是 datetime 类型以便正确绘图
    ar_df.index = pd.to_datetime(ar_df.index)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制 AR 曲线 (左轴)
    # 如果是日内 AR 分析 (没有 rolling_window 或者是 None)，样式可能需要不同
    if rolling_window is None:
        ax1.plot(ar_df.index, ar_df[f"AR({k})"], label=f"Intraday AR({k})", 
                 color='lightgray', marker='.', linestyle='-', linewidth=0.5, markersize=4)
        title_suffix = "Intraday Consistency"
    else:
        ax1.plot(ar_df.index, ar_df[f"AR({k})"], label=f"Absorption Ratio (Top {k})", linewidth=1.5, color='blue')
        title_suffix = f"Rolling Absorption Ratio (Window={rolling_window})"
    
    # 绘制 MA
    if ma_window:
        ma_col = f'MA{ma_window}'
        ar_df[ma_col] = ar_df[f"AR({k})"].rolling(ma_window).mean()
        ax1.plot(ar_df.index, ar_df[ma_col], label=f'{ma_window}-Day Trend', color='blue' if rolling_window is None else 'orange', linewidth=2)

    # 绘制阈值
    if thresholds:
        colors = ['red', 'green', 'orange']
        for i, th in enumerate(thresholds):
            ax1.axhline(y=th, color=colors[i%len(colors)], linestyle='--', alpha=0.3, label=f'Threshold ({th})')

    # AR(1) overlay for static/rolling case
    if rolling_window and k != 1 and "AR(1)" in ar_df.columns:
        ax1.plot(ar_df.index, ar_df["AR(1)"], label="Market Factor (Top 1)", linestyle="--", alpha=0.7, linewidth=1, color='cyan')
        
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Absorption Ratio (Variance Explained)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # 优化 X 轴日期显示
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate() # 自动旋转日期标签防止重叠
    
    # 如果指定了叠加 ETF (仅在非日内分析时推荐，或者双轴)
    if overlay_code:
        try:
            overlay_price = get_price(overlay_code, start_date=ar_df.index[0], end_date=ar_df.index[-1], fields=['close'])
            
            if not overlay_price.empty:
                ax2 = ax1.twinx()
                ax2.plot(overlay_price.index, overlay_price['close'], label=f"{overlay_code} Price", color='red', linewidth=1.5, alpha=0.8)
                ax2.set_ylabel(f"{overlay_code} Price", color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # 合并图例并放置在图外，防止遮挡
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
            else:
                ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        except Exception as e:
            print(f"  -> Failed to fetch overlay data: {e}")
            ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    else:
        ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
    plt.title(title_suffix)
    plt.tight_layout() # 调整布局，确保图例可见
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Plot saved to: {save_path}")

def analyze_pool_pca(etf_list, config, overlay_code=None):
    """
    对筛选出的 ETF 池进行 PCA 分析，计算吸收比率 (Absorption Ratio)
    """
    print(f"Step 2.5: Pool PCA Analysis (Absorption Ratio)...")
    
    if len(etf_list) < 2:
        print("  -> Pool too small for PCA analysis.")
        return

    # 1. 获取收益率数据
    returns = get_pca_source_data(etf_list, config["min_listing_days"])
    if returns.empty:
        print("  -> Returns data empty for PCA.")
        return

    # 2. 静态分析 (最后一段窗口)
    static_window = config["pca_rolling_window"]
    static_data = returns[-static_window:]
    res = calc_absorption_ratio(static_data)
    
    if not res:
        print("  -> PCA calculation failed.")
        return

    print(f"  -> Explained Variance by Top Components:")
    for i in range(min(3, len(res["evr"]))):
        print(f"     PC{i+1}: {res['evr'][i]:.2%}")
        
    print(f"  -> Absorption Ratio (AR) parameters:")
    print(f"     N (Assets): {res['n_assets']}")
    print(f"     K (Eigenvectors): {res['k']}")
    print(f"     AR(K) (Systemic Risk Indicator): {res['ar_k']:.2%}")
    if res['k'] != 1:
        print(f"     AR(1) (Market Factor Dominance): {res['ar_1']:.2%}")
        
    if res['ar_1'] > 0.75:
        print("  -> [WARNING] High AR(1)! Market is highly correlated (Fragile).")
    elif res['ar_1'] < 0.4:
        print("  -> [INFO] Low AR(1). Market is diverse.")

    # 3. 滚动分析
    print(f"  -> Generating Rolling AR Plot (Window={static_window})...")
    ar_df = calc_rolling_ar(returns, static_window, k=res['k'])
    
    # 4. 绘图
    if not ar_df.empty:
        plot_ar_analysis(ar_df, res['k'], static_window, overlay_code=overlay_code)
    else:
        print("  -> Failed to generate rolling data.")

