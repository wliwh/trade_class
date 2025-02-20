"""
分析指数均线突破点位
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import pandas as pd


def find_ma_breakthrough_points(df: pd.DataFrame, ma_period: int, min_days_below: int = 5, 
                                window_days: int = None) -> pd.DataFrame:
    """
    查找指数跌破均线的关键点位
    
    参数:
        df: 包含 'close' 和 'low' 列的DataFrame，index为日期
        ma_period: 均线周期（如60，240）
        min_days_below: 连续低于均线的最小天数
        window_days: 前后寻找最低点的窗口天数
    
    返回:
        包含突破点信息的DataFrame
    """
    if window_days is None:
        window_days = ma_period // 4  # 默认窗口为均线周期的1/4
        
    # 计算移动平均线
    df['ma'] = df['close'].rolling(window=ma_period).mean()
    
    # 计算是否低于均线
    df['below_ma'] = df['close'] < df['ma']
    
    # 找出连续低于均线的区间
    below_periods = []
    current_period = []
    
    for date, row in df.iterrows():
        if row['below_ma']:
            current_period.append(date)
        else:
            if len(current_period) >= min_days_below:
                below_periods.append(current_period)
            current_period = []
    
    if len(current_period) >= min_days_below:
        below_periods.append(current_period)
    
    # print(below_periods)
        
    # 对每个区间找出最低点
    breakthrough_points = []
    for period in below_periods:
        start_idx = max(0, df.index.get_loc(period[0]) - window_days)
        end_idx = min(len(df), df.index.get_loc(period[-1]) + window_days + 1)
        high_start_idx = max(0, start_idx - window_days)
        
        window_data = df.iloc[start_idx:end_idx]
        min_low_idx = window_data['low'].idxmin()
        # print(high_start_idx, type(high_start_idx), period[0], type(period[0]))
        max_high_idx = df.iloc[high_start_idx:df.index.get_loc(period[0])]['high'].idxmax()
        
        # 重复元素不要添加进去
        if period[0] <= min_low_idx <= period[-1]:
            breakthrough_points.append({
                'start_date': period[0],
                'start_price': df.loc[period[0], 'ma'],
                'end_date': period[-1],
                'highest_date': max_high_idx,
                'highest_price': df.loc[max_high_idx, 'high'],
                'lowest_date': min_low_idx,
                'lowest_price': df.loc[min_low_idx, 'low'],
            })
        
    return pd.DataFrame(breakthrough_points)

def analyze_price_series(df: pd.DataFrame) -> dict:
    """
    分析价格序列，找出跌破240日和60日均线的关键点位
    
    参数:
        df: 包含 'close' 和 'low' 列的DataFrame，index为日期
        
    返回:
        包含两种均线突破点的字典
    """
    # 240日均线分析
    ma240_points = find_ma_breakthrough_points(
        df, 
        ma_period=240,
        min_days_below=5,
        window_days=120  # 前后半年
    )
    
    # 60日均线分析
    ma60_points = find_ma_breakthrough_points(
        df, 
        ma_period=60, 
        min_days_below=5,
        window_days=30  # 前后1.5个月
    )

    # 过滤掉ma60中与ma240重叠的突破点
    ma240_dates = set(ma240_points['lowest_date'])
    ma60_filtered = ma60_points[~ma60_points['lowest_date'].isin(ma240_dates)]
    
    return {
        'ma240_points': ma240_points,
        'ma60_points': ma60_filtered
    }

def analyze_index_ma_points(index_df: pd.DataFrame, start_date: str = None) -> dict:
    """
    分析指数的均线突破点位
    
    参数:
        index_df: 包含指数数据的DataFrame，需要包含'close'和'low'列
        start_date: 开始分析的日期，如果为None则使用全部数据
    """
    if start_date:
        index_df = index_df[index_df.index >= pd.to_datetime(start_date)]
        
    # 确保数据足够长
    if len(index_df) < 240:
        raise ValueError("数据长度不足240个交易日，无法进行240日均线分析")
        
    return analyze_price_series(index_df)

def format_breakthrough_points(points_df: pd.DataFrame) -> str:
    """格式化突破点信息"""
    if points_df.empty:
        return "未发现符合条件的突破点"
        
    result = []
    for _, row in points_df.iterrows():
        info = (f"突破区间: {row['start_date'].strftime('%Y-%m-%d')} 到 "
                f"{row['end_date'].strftime('%Y-%m-%d')}\n"
                f"前高处日期: {row['highest_date'].strftime('%Y-%m-%d')}\n"
                f"最高价: {row['highest_price']:.2f}\n"
                f"起点处均线: {row['start_price']:.2f}\n"
                f"最低点日期: {row['lowest_date'].strftime('%Y-%m-%d')}\n"
                f"最低价: {row['lowest_price']:.2f}\n"
                f"差异: {(row['start_price']-row['lowest_price'])/(row['highest_price']-row['start_price']):.2f}\n")
                # f"当时均线值: {row['ma_value']:.2f}\n"
                # f"偏离均线: {((row['lowest_price'] / row['ma_value'] - 1) * 100):.2f}%\n")
        result.append(info)
        
    return "\n".join(result)

if __name__ == '__main__':
    # 示例：获取指数数据并分析
    from get_index_value import other_index_getter
    
    # 获取指数数据
    index_getter = other_index_getter('IXIC','20160101')
    index_getter.index = pd.to_datetime(index_getter.index)
    
    # 分析最近一年的数据
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=8)).strftime('%Y-%m-%d')
    results = analyze_index_ma_points(index_getter, start_date)
    
    print("=== 指数均线分析结果 ===")
    print("\n240日均线突破点:")
    print(format_breakthrough_points(results['ma240_points']))
    print("\n60日均线突破点:")
    print(format_breakthrough_points(results['ma60_points'])) 