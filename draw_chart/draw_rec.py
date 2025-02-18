"""
绘制递归图
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from index_get.get_index_value import other_index_getter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

Search_Index = {
    '道琼斯':'道琼斯',
    '标普500':'标普500',
    '纳斯达克':'NDX',
    '纳指100':'NDX100'
}

def recurrence_plot(data, threshold=0.1):
    """
    Generate a recurrence plot from a time series.

    :param data: Time series data
    :param threshold: Threshold to determine recurrence
    :return: Recurrence plot
    """
    # Calculate the distance matrix
    N = len(data)
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distance_matrix[i, j] = np.abs(data[i] - data[j])

    # Create the recurrence plot
    recurrence_plot = np.where(distance_matrix <= threshold, 1, 0)

    return recurrence_plot


def detect_cycle_lows(data_path, window_size=600, cycle_range=(32,52)):
    """
    分析日频金融数据，识别周期性低点
    参数：
    data_path - 数据文件路径（CSV格式，含date和price列）
    window_size - 局部低点检测窗口（交易日天数，默认600天≈2年）
    cycle_range - 目标周期范围（月数，默认32-52个月）
    """
    # 数据加载与预处理
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    
    # 平滑处理（三重指数平滑）
    df['smoothed'] = df['price'].ewm(span=180).mean().ewm(span=120).mean().ewm(span=60).mean()
    
    # 寻找局部低点
    min_idx = argrelextrema(df['smoothed'].values, np.less, order=window_size)[0]
    low_points = df.iloc[min_idx][['date', 'price']].reset_index(drop=True)
    
    # 计算周期长度并筛选
    low_points['months_since_last'] = low_points['date'].diff().dt.days / 30.437
    valid_cycles = low_points[
        (low_points['months_since_last'] >= cycle_range[0]) &
        (low_points['months_since_last'] <= cycle_range[1])
    ]
    
    # 标记周期起点
    valid_cycles['cycle_id'] = range(1, len(valid_cycles)+1)
    return valid_cycles[['cycle_id', 'date', 'price']]


def plot_cycles(df, cycles):
    plt.figure(figsize=(12,6))
    plt.plot(df['date'], df['price'], label='原始价格', alpha=0.3)
    plt.plot(df['date'], df['smoothed'], label='平滑曲线', color='orange')
    plt.scatter(cycles['date'], cycles['price'], 
                color='red', zorder=5, label='周期起点')
    
    for _, row in cycles.iterrows():
        plt.annotate(f'Cycle {row["cycle_id"]}\n{row["date"].strftime("%Y-%m")}',
                     (row['date'], row['price']),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center')
    
    plt.title('金融时间序列周期分析')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    plt.show() 