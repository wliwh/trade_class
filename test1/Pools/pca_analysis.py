from jqdata import *
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def analyze_pool_pca(etf_list, config, overlay_code=None):
    """
    对筛选出的 ETF 池进行 PCA 分析，计算吸收比率 (Absorption Ratio)
    用于评估组合的系统性风险和分散程度
    """
    print(f"Step 2.5: Pool PCA Analysis (Absorption Ratio)...")
    
    if len(etf_list) < 2:
        print("  -> Pool too small for PCA analysis.")
        return

    # 1. 获取价格数据
    money_median_20d = history(20, "1d", "money", etf_list, df=True).median()
    # 再次过滤一下确保有数据 (虽然传入的应该是 valid 的)
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > 0]
    
    # 使用从 config 传入的 min_listing_days
    price_data = history(config["min_listing_days"], "1d", "close", valid_etfs, df=True)
    price_data = price_data.fillna(method="ffill").dropna(axis=1, how="any")
    
    if price_data.empty:
        print("  -> Price data empty for PCA.")
        return
        
    returns = price_data.pct_change().dropna()
    
    if returns.empty:
        print("  -> Returns data empty for PCA.")
        return

    # 2. 执行 PCA
    pca = PCA()
    # 使用滚动窗口大小的最后一段数据进行静态分析
    pca.fit(returns[-config["pca_rolling_window"]:])
    
    # 3. 计算解释方差比
    evr = pca.explained_variance_ratio_
    cum_var = np.cumsum(evr)
    
    print(f"  -> Explained Variance by Top Components:")
    for i in range(min(3, len(evr))):
        print(f"     PC{i+1}: {evr[i]:.2%}")
        
    # 4. 计算吸收比率 (AR)
    # K 取 N/5, 或者是 1
    n = len(valid_etfs)
    # 确保 k 不超过实际的主成分数量 (受限于样本量 rolling_window)
    k = min(max(1, int(n / 5)), len(evr))
    k = min(k, 10)
    
    ar_val = cum_var[k-1]
    
    print(f"  -> Absorption Ratio (AR) parameters:")
    print(f"     N (Assets): {n}")
    print(f"     K (Eigenvectors): {k}")
    print(f"     AR(K) (Systemic Risk Indicator): {ar_val:.2%}")
    if k != 1:
        print(f"     AR(1) (Market Factor Dominance): {cum_var[0]:.2%}")
        
    if cum_var[0] > 0.75:
        print("  -> [WARNING] High AR(1)! Market is highly correlated (Fragile).")
    elif cum_var[0] < 0.4:
        print("  -> [INFO] Low AR(1). Market is diverse.")

    # ----------------------------------------------------
    # 新增：滚动 AR 分析与绘图
    # ----------------------------------------------------
    print(f"  -> Generating Rolling AR Plot (Window={config['pca_rolling_window']})...")
    
    rolling_window = config["pca_rolling_window"]
    if len(returns) < rolling_window + 10:
        print("  -> Not enough data for rolling PCA.")
        return

    ar_series = []
    
    # 滚动窗口计算
    for i in range(rolling_window, len(returns)):
        window_data = returns.iloc[i-rolling_window : i]
        current_date = returns.index[i]
        
        try:
            pca_roll = PCA()
            pca_roll.fit(window_data)
            
            # 这里我们主要关注 AR(1) 或者是 AR(k)
            evr_roll = pca_roll.explained_variance_ratio_
            cum_var_roll = np.cumsum(evr_roll)
            
            # 使用前面静态计算确定的 k, 但同样需要受限于 rolling 窗口产生的主成分数量
            # 通常 rolling_window=60, 如果 k > 60 会报错
            k_rolling = min(k, len(evr_roll))
            
            ar_k_val = cum_var_roll[k_rolling-1]
            
            ar_series.append({
                "date": current_date,
                f"AR({k})": ar_k_val,
                "AR(1)": cum_var_roll[0]
            })
        except:
            continue
            
    if not ar_series:
        print("  -> Failed to generate rolling data.")
        return
        
    ar_df = pd.DataFrame(ar_series).set_index("date")
    
    
    # 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 绘制 AR 曲线 (左轴)
    ax1.plot(ar_df.index, ar_df[f"AR({k})"], label=f"Absorption Ratio (Top {k})", linewidth=1.5, color='blue')
    
    if k != 1:
        ax1.plot(ar_df.index, ar_df["AR(1)"], label="Market Factor (Top 1)", linestyle="--", alpha=0.7, linewidth=1, color='cyan')
        
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Absorption Ratio (Variance Explained)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # 如果指定了叠加 ETF，则绘制价格曲线 (右轴)
    if overlay_code:
        try:
            print(f"  -> Fetching overlay price data for {overlay_code}...")
            # 获取对应日期的价格数据
            overlay_price = get_price(overlay_code, start_date=ar_df.index[0], end_date=ar_df.index[-1], fields=['close'])
            
            if not overlay_price.empty:
                ax2 = ax1.twinx()
                ax2.plot(overlay_price.index, overlay_price['close'], label=f"{overlay_code} Price", color='red', linewidth=1.5, alpha=0.8)
                ax2.set_ylabel(f"{overlay_code} Price", color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                # 合并图例
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            else:
                print(f"  -> Overlay price data empty for {overlay_code}.")
                ax1.legend(loc='upper left')
        except Exception as e:
            print(f"  -> Failed to fetch overlay data: {e}")
            ax1.legend(loc='upper left')
    else:
        ax1.legend(loc='upper left')
        
    plt.title(f"Rolling Absorption Ratio (Window={rolling_window} days)")
    
    # 保存图像
    plot_filename = "ar_rolling_plot.png"
    plt.savefig(plot_filename)
    print(f"  -> Plot saved to: {plot_filename}")
    plt.close() # 关闭图形释放内存
