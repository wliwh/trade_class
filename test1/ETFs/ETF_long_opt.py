# 风险及免责提示：该策略由聚宽用户在聚宽社区分享，仅供学习交流使用。
# 原文一般包含策略说明，如有疑问请到原文和作者交流讨论。
# 原文网址：https://www.joinquant.com/view/community/detail/44046
# 标题：【回顾3】ETF策略之核心资产轮动-添油加醋
# 
# 本程序为原策略的向量化/模块化重构版本，逻辑与原版完全一致。

import pandas as pd
import numpy as np
import math
from jqdata import *

EXECUTION_TIME_PLACEHOLDER = '9:30'
EXECUTION_ETF_POOLS_PLACEHOLDER = ['518880.XSHG','513100.XSHG','159915.XSHE','510180.XSHG']

class GlobalConfig:
    AVOID_FUTURE_DATA = True
    USE_REAL_PRICE = True

    BENCHMARK = "513100.XSHG"
    
    # 滑点与费率
    SLIPPAGE_FUND = 0.001
    SLIPPAGE_STOCK = 0.003
    COMMISSION_STOCK_OPEN = 0.0002
    COMMISSION_STOCK_CLOSE = 0.0002
    COMMISSION_MIN = 0
    
    # 动量参数
    MOMENTUM_DAYS = 25
    REVERSAL_DAYS = 200 # 25 * 8
    
    # 筛选参数
    TARGET_NUM = 1
    SCORE_DIFF_MIN = 0.1
    SCORE_DIFF_MAX = 15
    
    # RSRS参数
    RSRS_REG_WINDOW = 18    # 计算当前Beta的窗口
    RSRS_HIST_WINDOW = 250  # 计算Beta统计分布的历史长度
    RSRS_ROLLING_WINDOW = 20 # 历史统计中Rolling Beta的窗口
    RSRS_STD_MULTIPLIER = 2.0

class DataPreparation:
    @staticmethod
    def fetch_price_history(security_list, days):
        """
        一次性获取所有标的的历史收盘价，用于打分
        返回: DataFrame (index=date, columns=securities)
        """
        return history(days, '1d', 'close', security_list)

    @staticmethod
    def fetch_high_low_history(security, days):
        """
        获取单只标的的高低价，用于RSRS计算
        """
        return attribute_history(security, days, '1d', ['high', 'low'])

class Scoring:
    @staticmethod
    def calculate_slope_r2_vectorized(prices_df):
        """
        向量化计算多只ETF的斜率和R方
        输入: DataFrame (rows=time, cols=securities) - Log Prices
        输出: Series (index=securities) for slope, r2
        """
        n = len(prices_df)
        x = np.arange(n)
        x_mean = np.mean(x)
        x_var = np.var(x, ddof=1) 
        y_mean = prices_df.mean(axis=0)
        x_center = x - x_mean # (N,)
        y_center = prices_df - y_mean # (N, M)
        cov_xy = np.sum(x_center[:, np.newaxis] * y_center, axis=0) # (M,)
        sum_sq_diff_x = np.sum(x_center**2)
        slopes = cov_xy / sum_sq_diff_x
        intercepts = y_mean - slopes * x_mean
        sst = np.sum(y_center**2, axis=0)
        r_squareds = (cov_xy / np.sqrt(sum_sq_diff_x * sst)) ** 2
        
        return slopes, r_squareds

    @classmethod
    def get_scores(cls, prices_df):
        """
        计算综合得分
        prices_df: 包含足够长历史数据的DataFrame (max(momentum, reversal) days)
        """
        log_prices = np.log(prices_df)
        mom_df = log_prices.iloc[-GlobalConfig.MOMENTUM_DAYS:]
        mom_slopes, mom_r2 = cls.calculate_slope_r2_vectorized(mom_df)
        mom_annual_ret = np.power(np.exp(mom_slopes), 250) - 1
        mom_score = mom_annual_ret * mom_r2
        rev_df = log_prices.iloc[-GlobalConfig.REVERSAL_DAYS:]
        rev_slopes, rev_r2 = cls.calculate_slope_r2_vectorized(rev_df)
        rev_annual_ret = np.power(np.exp(rev_slopes), 250) - 1
        rev_score = rev_annual_ret * rev_r2
        final_scores = mom_score - rev_score / 6.0
        return final_scores.sort_values(ascending=False)

class PostFiltering:
    @staticmethod
    def filter_candidates(scores_series):
        """
        根据分差筛选
        """
        if len(scores_series) < 2:
             return [] if scores_series.empty else [scores_series.index[0]]
            
        max_score = scores_series.iloc[0]
        min_score = scores_series.iloc[-1]
        diff = max_score - min_score
        
        if GlobalConfig.SCORE_DIFF_MIN < diff < GlobalConfig.SCORE_DIFF_MAX:
            return list(scores_series.index[:GlobalConfig.TARGET_NUM])
        else:
            return []

class RiskControl:
    @staticmethod
    def check_rsrs(context, target_list):
        """
        对目标ETF进行RSRS择时检查
        """
        real_target_list = []
        for etf in target_list:
            if RiskControl._is_safe(context, etf):
                real_target_list.append(etf)
        return real_target_list

    @staticmethod
    def _is_safe(context, etf):
        # 1. 获取当前 Beta (18日)
        curr_data = DataPreparation.fetch_high_low_history(etf, GlobalConfig.RSRS_REG_WINDOW)
        curr_beta = np.polyfit(curr_data.low, curr_data.high, 1)[0]
        # 2. 获取阈值 (250日历史中的 RSRS_ROLLING_WINDOW 滚动beta)
        hist_data = DataPreparation.fetch_high_low_history(etf, GlobalConfig.RSRS_HIST_WINDOW)
        betas = []
        lows = hist_data['low'].values
        highs = hist_data['high'].values
        
        window = GlobalConfig.RSRS_ROLLING_WINDOW
        limit_idx = len(hist_data) - GlobalConfig.RSRS_ROLLING_WINDOW - 1 
        
        for i in range(limit_idx):
            y = highs[i:i+window]
            x = lows[i:i+window]
            beta = np.polyfit(x, y, 1)[0]
            betas.append(beta)
            
        beta_mean = np.mean(betas)
        beta_std = np.std(betas)
        threshold = beta_mean - GlobalConfig.RSRS_STD_MULTIPLIER * beta_std
        return curr_beta > threshold

class Execution:
    @staticmethod
    def execute(context, target_list):
        current_holdings = list(context.portfolio.positions.keys())

        for etf in current_holdings:
            if etf not in target_list:
                order_target_value(etf, 0)
                # log.info(f"卖出 {etf}")

        if not target_list:
            return
            
        current_holdings_after_sell = [e for e in context.portfolio.positions if context.portfolio.positions[e].total_amount > 0]
        if len(current_holdings_after_sell) < GlobalConfig.TARGET_NUM:
            slots_needed = GlobalConfig.TARGET_NUM - len(current_holdings_after_sell)
            cash_per_slot = context.portfolio.available_cash / slots_needed
            for etf in target_list:
                if context.portfolio.positions[etf].total_amount == 0:
                    order_target_value(etf, cash_per_slot)

# JQ 初始化与周期函数

def initialize(context):
    set_benchmark(GlobalConfig.BENCHMARK)
    set_option('use_real_price', GlobalConfig.USE_REAL_PRICE)
    set_option("avoid_future_data", GlobalConfig.AVOID_FUTURE_DATA)
    set_slippage(FixedSlippage(GlobalConfig.SLIPPAGE_FUND), type="fund")
    set_slippage(FixedSlippage(GlobalConfig.SLIPPAGE_STOCK), type="stock")
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=0,
        open_commission=GlobalConfig.COMMISSION_STOCK_OPEN,
        close_commission=GlobalConfig.COMMISSION_STOCK_CLOSE,
        close_today_commission=0,
        min_commission=GlobalConfig.COMMISSION_MIN), type='fund')
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0, close_commission=0, 
        close_today_commission=0, min_commission=0), type="mmf")
    log.set_level('system', 'error')
    run_daily(main_trade, EXECUTION_TIME_PLACEHOLDER)

def main_trade(context):
    # 1. 数据准备
    price_data = DataPreparation.fetch_price_history(EXECUTION_ETF_POOLS_PLACEHOLDER, GlobalConfig.REVERSAL_DAYS)
    # 2. 打分
    if price_data.empty: 
        return
    scores = Scoring.get_scores(price_data)
    # 3. 初步筛选 (Score Spread)
    candidates = PostFiltering.filter_candidates(scores)
    # 4. 风险控制 (RSRS)
    final_targets = RiskControl.check_rsrs(context, candidates)
    # 5. 执行
    Execution.execute(context, final_targets)