"""
分析指数均线突破点位
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import pandas as pd
from common.price_analyzer import PriceAnalyzer
from common.trade_date import get_trade_day_between

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
        
    return PriceAnalyzer.analyze_price_series(index_df)

def format_breakthrough_points(points_df: pd.DataFrame) -> str:
    """格式化突破点信息"""
    if points_df.empty:
        return "未发现符合条件的突破点"
        
    result = []
    for _, row in points_df.iterrows():
        info = (f"突破区间: {row['start_date'].strftime('%Y-%m-%d')} 到 "
                f"{row['end_date'].strftime('%Y-%m-%d')}\n"
                f"最低点日期: {row['lowest_date'].strftime('%Y-%m-%d')}\n"
                f"最低价: {row['lowest_price']:.2f}\n"
                f"当时均线值: {row['ma_value']:.2f}\n"
                f"偏离均线: {((row['lowest_price'] / row['ma_value'] - 1) * 100):.2f}%\n")
        result.append(info)
        
    return "\n".join(result)

if __name__ == '__main__':
    # 示例：获取指数数据并分析
    from get_index_value import other_index_getter
    
    # 获取指数数据
    index_getter = other_index_getter('标普500','20160101')
    index_getter.index = pd.to_datetime(index_getter.index)
    
    # 分析最近一年的数据
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=8)).strftime('%Y-%m-%d')
    results = analyze_index_ma_points(index_getter, start_date)
    
    print("=== 指数均线分析结果 ===")
    print("\n240日均线突破点:")
    print(format_breakthrough_points(results['ma240_points']))
    print("\n60日均线突破点:")
    print(format_breakthrough_points(results['ma60_points'])) 