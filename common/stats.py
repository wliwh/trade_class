"""
指数关键数据统计
1. zf(weekly, monthly)
2. wp(monthly, quarterly, yearly)
"""

import pandas as pd


def calculate_win_rate(data):
    previous_close = data['Close'].shift(1)
    data['Win'] = data['Close'] > previous_close
    wins = (data['Win'] == True).sum()
    total_trades = len(data) - 1
    win_rate = wins / total_trades if total_trades > 0 else 0
    return win_rate


