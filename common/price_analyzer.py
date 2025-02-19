import pandas as pd
import numpy as np

class PriceAnalyzer:
    @staticmethod
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
            
        # 对每个区间找出最低点
        breakthrough_points = []
        for period in below_periods:
            start_idx = max(0, df.index.get_loc(period[0]) - window_days)
            end_idx = min(len(df), df.index.get_loc(period[-1]) + window_days + 1)
            
            window_data = df.iloc[start_idx:end_idx]
            min_low_idx = window_data['low'].idxmin()
            
            # 重复元素不要添加进去
            if period[0] < min_low_idx < period[-1]:
                breakthrough_points.append({
                    'start_date': period[0],
                    'end_date': period[-1],
                    'lowest_date': min_low_idx,
                    'lowest_price': df.loc[min_low_idx, 'low'],
                    'ma_value': df.loc[min_low_idx, 'ma']
                })
            
        return pd.DataFrame(breakthrough_points)

    @staticmethod
    def analyze_price_series(df: pd.DataFrame) -> dict:
        """
        分析价格序列，找出跌破240日和60日均线的关键点位
        
        参数:
            df: 包含 'close' 和 'low' 列的DataFrame，index为日期
            
        返回:
            包含两种均线突破点的字典
        """
        # 240日均线分析
        ma240_points = PriceAnalyzer.find_ma_breakthrough_points(
            df, 
            ma_period=240, 
            min_days_below=5,
            window_days=120  # 前后半年
        )
        
        # 60日均线分析
        ma60_points = PriceAnalyzer.find_ma_breakthrough_points(
            df, 
            ma_period=60, 
            min_days_below=5,
            window_days=45  # 前后1.5个月
        )
        
        return {
            'ma240_points': ma240_points,
            'ma60_points': ma60_points
        } 