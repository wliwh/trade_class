"""
绘制递归图
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from index_get.get_index_value import other_index_getter
from common.smooth_tool import drawdown_series
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


def detect_cycle_lows(df, price_col='price', window_size=100, cycle_range=(33,49)):
    """
    分析日频金融数据，识别周期性低点
    参数：
    df - 数据文件路径（CSV格式，含date和price列）
    window_size - 局部低点检测窗口（交易日天数，默认600天≈2年）
    cycle_range - 目标周期范围（月数，默认33-49个月）
    """
    # 数据加载与预处理
    df['date'] = pd.to_datetime(df.index)
    # df.index = range(len(df))
    # df.index.name = 'idx'
    # df.sort_values('date', inplace=True)
    if isinstance(price_col, str):
        col1 = col2 = price_col
    else:
        col1, col2 = price_col

    # 寻找局部低点
    min_idx = argrelextrema(df[col1].values, np.less, order=window_size)[0]
    low_points = df.iloc[min_idx][['date', col1]].reset_index(drop=True)
    low_points['diff'] = (low_points['date']-df['date'].min()).dt.days

    # 找出第一个周期起点
    right_date = low_points.loc[low_points['diff']<=int(cycle_range[1]*30.4),col1].idxmin()
    first_cycle = low_points.iloc[right_date]
    low_points['diff'] = low_points['diff'] - first_cycle['diff']

    # 找出所有周期低点
    cycle_lows = [{'date':first_cycle['date'],'price':first_cycle[col1],'diff':0,'type':'L'}]
    while True:
        wks = low_points[(low_points['diff']>=int(cycle_range[0]*30.4)) & (low_points['diff']<=int(cycle_range[1]*30.4))]
        if wks.empty: break
        wks['diff'] = wks['diff'] / 30.4 / 42 -1
        wks['rank'] = 5*(wks[col1]/wks[col1].min()-1) + wks['diff'].apply(lambda x: x*2.5 if x>0 else -x)
        wks['rank'] = wks['rank'].apply(lambda x: 0 if x<0 else x)
        now_cycle = low_points.iloc[wks['rank'].idxmin()]
        cycle_lows.append({'date':now_cycle['date'],'price':now_cycle[col1],'diff':now_cycle['diff']/30.4,'type':'L'})
        low_points['diff'] = low_points['diff'] - now_cycle['diff']

    # 找出周期内的高点
    idx = df.loc[df['date'] <= cycle_lows[0]['date'],col2].idxmax()
    cycle_highs = [{'date':df.loc[idx,'date'],'price':df.loc[idx,col2],'diff':(df.loc[idx,'date']-cycle_lows[0]['date']).days/30.4,'type':'H'}]
    for t in range(len(cycle_lows)-1):
        left_date = cycle_lows[t]['date']
        right_date = cycle_lows[t+1]['date']
        idx = df.loc[(df['date']>=left_date) & (df['date']<=right_date),col2].idxmax()
        cycle_highs.append({'date':df.loc[idx,'date'],'price':df.loc[idx,col2],'diff':(df.loc[idx,'date']-left_date).days/30.4,'type':'H'})
    idx = df.loc[df['date'] >= cycle_highs[-1]['date'],col2].idxmax()
    cycle_highs.append({'date':df.loc[idx,'date'],'price':df.loc[idx,col2],'diff':(df.loc[idx,'date']-right_date).days/30.4,'type':'H'})

    # 依次寻找每个周期低点
    # end_date = df['date'].max()
    # now_date = first_cycle['date']
    # cycle_lows = [now_date]
    # while now_date < end_date:
    #     left_date = now_date+pd.Timedelta(days=int(cycle_range[0]*30.4))
    #     right_date = now_date+pd.Timedelta(days=int(cycle_range[1]*30.4))
    #     if left_date > end_date:
    #         break
    #     now_idx = df.loc[(df['date']>=left_date) & (df['date']<=right_date),col1].idxmin()
    #     now_date = df.iloc[now_idx]['date']
    #     cycle_lows.append(now_date)
    # print(cycle_lows)
    cycles = pd.DataFrame(cycle_lows + cycle_highs)
    cycles.sort_values('date', inplace=True)
    cycles.reset_index(drop=True, inplace=True)
    return cycles


def plot_cycles(df, cycles, df_name='price'):
    # 采用上下布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10))
    # 添加对数同比
    df['log'] = np.log(df[df_name])/(np.log(df[df_name]).shift(250))
    df['log'].fillna(1, inplace=True)
    cycles_low = cycles[cycles['type']=='L']
    cycles_high = cycles[cycles['type']=='H']
    ax1.plot(df['date'], df[df_name], label='Price', alpha=0.3)
    # plt.plot(df['date'], df['smoothed'], label='平滑曲线', color='orange')
    ax1.scatter(cycles_low['date'], cycles_low['price'], 
                color='red', zorder=5, label='Cycle Low')
    ax1.scatter(cycles_high['date'], cycles_high['price'], 
                color='green', zorder=5, label='Cycle High')
    ax2.plot(df['date'], df['log'], label='Log Ratio', alpha=0.7)
    for _, row in cycles.iterrows():
        ax2.scatter(row['date'], df.loc[row['date'] == df['date'], 'log'], 
                color='red' if row['type']=='L' else 'green', zorder=5)

    # for i, row in cycles_low.iterrows():
    #     plt.annotate(f'Cyc{(i+1)//2}: {row["date"].strftime("%Y-%m")}',
    #                  (row['date'], row['price']),
    #                  textcoords="offset points",
    #                  xytext=(0,-14),
    #                  ha='left')
    
    plt.title('Cycle Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show() 

if __name__ == '__main__':
    df = other_index_getter(Search_Index['标普500'],'2001-01-01')
    # df['date'] = pd.to_datetime(df.index)
    cycles = detect_cycle_lows(df, price_col='close')
    # print(cycles)
    print(drawdown_series(df.loc[df['date']>=cycles.iloc[-2,0],'close']))
    # plot_cycles(df, cycles, df_name='close')