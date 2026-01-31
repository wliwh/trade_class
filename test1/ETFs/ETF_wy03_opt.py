# 原文网址：https://www.joinquant.com/post/42673
# 标题：【回顾3】ETF策略之核心资产轮动 (极速优化版)
# 作者：wywy1995
# 优化说明：
# 1. 使用 numpy 向量化计算替代了循环中的 polyfit，大幅提升了大量标的时的计算速度。
# 2. 批量获取历史数据，减少 IO 次数。

import numpy as np
import pandas as pd
import math
from jqdata import *

EXECUTION_TIME_PLACEHOLDER = '9:30'
EXECUTION_ETF_POOLS_PLACEHOLDER = ['518880.XSHG','513100.XSHG','159915.XSHE','510180.XSHG']

class Config:
    # ==================== 交易环境设置 ====================
    AVOID_FUTURE_DATA = True
    USE_REAL_PRICE = True

    BENCHMARK = "513100.XSHG"
    
    # 滑点与费率
    SLIPPAGE_FUND = 0.001
    SLIPPAGE_STOCK = 0.003
    
    COMMISSION_STOCK_OPEN = 0.0002
    COMMISSION_STOCK_CLOSE = 0.0002
    COMMISSION_MIN = 0

    # 策略核心参数
    M_DAYS = 25
    HOLD_COUNT = 1

def initialize(context):                   # 初始化函数 
    set_benchmark(Config.BENCHMARK)           # 设定基准
    set_option('use_real_price', Config.USE_REAL_PRICE)     # 用真实价格交易
    set_option("avoid_future_data", Config.AVOID_FUTURE_DATA)  # 打开防未来函数

    set_slippage(FixedSlippage(Config.SLIPPAGE_FUND), type="fund")
    set_slippage(FixedSlippage(Config.SLIPPAGE_STOCK), type="stock")
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=Config.COMMISSION_STOCK_OPEN, 
        close_commission=Config.COMMISSION_STOCK_CLOSE, 
        close_today_commission=0, min_commission=Config.COMMISSION_MIN
    ), type="fund")
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0, close_commission=0, 
        close_today_commission=0, min_commission=0
    ), type="mmf")

    log.set_level('system', 'error')       # 过滤一定级别的日志
    
    # 参数
    g.etf_pool = EXECUTION_ETF_POOLS_PLACEHOLDER
    g.m_days = Config.M_DAYS
    g.hold_count = Config.HOLD_COUNT
    
    run_daily(trade, EXECUTION_TIME_PLACEHOLDER)  # 每天运行确保即时捕捉动量变化

def get_rank_fast(etf_pool):
    """
    极速版排名计算：
    1. 批量获取数据
    2. 向量化计算斜率和R2
    """
    if not etf_pool: return []
    
    # 1. 批量获取数据 (返回 DataFrame, index=date, columns=etf_code)
    prices_df = history(g.m_days, '1d', 'close', etf_pool)
    
    # log 变换
    log_prices = np.log(prices_df)
    
    # 准备 X 轴 (0, 1, 2, ..., N-1)
    N = g.m_days
    x = np.arange(N)
    Y = log_prices.values # shape (N, M_etfs)
    x_mean = np.mean(x)
    y_mean = np.mean(Y, axis=0) # shape (M,)
    x_diff = x - x_mean # shape (N,)
    y_diff = Y - y_mean # shape (N, M)
    numerator = np.sum(x_diff[:, np.newaxis] * y_diff, axis=0)
    denominator = np.sum(x_diff**2)
    slope = numerator / denominator # shape (M,)
    intercept = y_mean - slope * x_mean
    y_pred = slope[np.newaxis, :] * x[:, np.newaxis] + intercept[np.newaxis, :] # (N, M)
    ss_res = np.sum((Y - y_pred)**2, axis=0)
    ss_tot = np.sum((Y - y_mean)**2, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_squared = 1 - (ss_res / ss_tot)
        r_squared = np.nan_to_num(r_squared) # 若 ss_tot为0，R2设为0
    annualized_returns = np.power(np.exp(slope), 250) - 1
    scores = annualized_returns * r_squared
    res_df = pd.DataFrame({'score': scores}, index=log_prices.columns)
    res_df = res_df.sort_values(by='score', ascending=False)
   
    return list(res_df.index)

# 交易操作
def trade(context):
    target_num = g.hold_count
    rank_list = get_rank_fast(g.etf_pool)
    target_list = rank_list[:target_num]
    
    # 卖出操作   
    hold_list = list(context.portfolio.positions)
    for etf in hold_list:
        if etf not in target_list:
            order_target_value(etf, 0)
            print('卖出' + str(etf))
        else:
            print('继续持有' + str(etf))
            
    # 买入操作
    hold_list = list(context.portfolio.positions)
    if len(hold_list) < target_num:
        if (target_num - len(hold_list)) > 0:
            value = context.portfolio.available_cash / (target_num - len(hold_list))
            for etf in target_list:
                if context.portfolio.positions[etf].total_amount == 0:
                    order_target_value(etf, value)
                    print('买入' + str(etf))
