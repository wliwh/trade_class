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

class GlobalConfig:
    # ETF池
    ETF_POOL = [
        '518880.XSHG', # 黄金ETF（大宗商品）
        '513100.XSHG', # 纳指100（海外资产）
        '159915.XSHE', # 创业板100（成长股，科技股，中小盘）
        '510180.XSHG', # 上证180（价值股，蓝筹股，中大盘）
    ]
    
    # 动量参数
    MOMENTUM_DAYS = 25
    REVERSAL_DAYS = 200
    
    # 交易时间
    RUN_TIME = '9:30'
    
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
        """
        if prices_df.empty:
            return pd.Series(), pd.Series()

        n = len(prices_df)
        x = np.arange(n)
        
        # 计算X的方差和均值
        x_mean = np.mean(x)
        
        # 向量化计算
        y_mean = prices_df.mean(axis=0)
        
        # Centering
        x_center = x - x_mean
        y_center = prices_df - y_mean
        
        # Covariance (unscaled)
        cov_xy = np.sum(x_center[:, np.newaxis] * y_center, axis=0)
        
        # Slope
        sum_sq_diff_x = np.sum(x_center**2)
        if sum_sq_diff_x == 0:
             return pd.Series(0, index=prices_df.columns), pd.Series(0, index=prices_df.columns)

        slopes = cov_xy / sum_sq_diff_x
        
        # R-Squared
        sst = np.sum(y_center**2, axis=0)          
        r_squareds = (cov_xy / np.sqrt(sum_sq_diff_x * sst)) ** 2
        
        return slopes, r_squareds

    @classmethod
    def get_scores(cls, prices_df):
        """
        计算综合得分
        """
        # 取对数
        log_prices = np.log(prices_df)
        
        # 1. 动量得分 (最近 Momentum_Days)
        mom_df = log_prices.iloc[-GlobalConfig.MOMENTUM_DAYS:]
        mom_slopes, mom_r2 = cls.calculate_slope_r2_vectorized(mom_df)
        mom_annual_ret = np.power(np.exp(mom_slopes), 250) - 1
        mom_score = mom_annual_ret * mom_r2
        
        # 2. 反转得分 (最近 Reversal_Days)
        rev_df = log_prices.iloc[-GlobalConfig.REVERSAL_DAYS:]
        rev_slopes, rev_r2 = cls.calculate_slope_r2_vectorized(rev_df)
        rev_annual_ret = np.power(np.exp(rev_slopes), 250) - 1
        rev_score = rev_annual_ret * rev_r2
        
        # 3. 综合得分: Mom - Rev / 6
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
        # 简单检查数据长度
        if len(curr_data) < 2: return False
            
        curr_beta = np.polyfit(curr_data.low, curr_data.high, 1)[0]
        
        # 2. 获取阈值
        hist_data = DataPreparation.fetch_high_low_history(etf, GlobalConfig.RSRS_HIST_WINDOW)
        betas = []
        
        lows = hist_data['low'].values
        highs = hist_data['high'].values
        
        window = GlobalConfig.RSRS_ROLLING_WINDOW
        limit_idx = len(hist_data) - GlobalConfig.RSRS_ROLLING_WINDOW - 1
        
        if limit_idx < 1: return False # 数据不足

        for i in range(limit_idx):
            y = highs[i:i+window]
            x = lows[i:i+window]
            beta = np.polyfit(x, y, 1)[0]
            betas.append(beta)
            
        if not betas: return False
            
        beta_mean = np.mean(betas)
        beta_std = np.std(betas)
        
        threshold = beta_mean - GlobalConfig.RSRS_STD_MULTIPLIER * beta_std
        
        return curr_beta > threshold

class Execution:
    @staticmethod
    def execute(context, target_list):
        current_holdings = list(context.portfolio.positions.keys())
        
        # 1. 卖出
        for etf in current_holdings:
            if etf not in target_list:
                order_target_value(etf, 0)
                
        # 2. 买入
        if not target_list:
            return
            
        current_holdings_after_sell = [e for e in context.portfolio.positions if context.portfolio.positions[e].total_amount > 0]
        
        if len(current_holdings_after_sell) < GlobalConfig.TARGET_NUM:
            slots_needed = GlobalConfig.TARGET_NUM - len(current_holdings_after_sell)
            if slots_needed == 0: return

            cash_per_slot = context.portfolio.available_cash / slots_needed
            
            for etf in target_list:
                if context.portfolio.positions[etf].total_amount == 0:
                    order_target_value(etf, cash_per_slot)

# JQ 初始化与周期函数

def initialize(context):
    set_benchmark('513100.XSHG')
    set_option('use_real_price', True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(0.002))
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.0002, close_commission=0.0002, close_today_commission=0, min_commission=5), type='fund')
    log.set_level('system', 'error')
    
    run_daily(main_trade, GlobalConfig.RUN_TIME)

def main_trade(context):
    price_data = DataPreparation.fetch_price_history(GlobalConfig.ETF_POOL, GlobalConfig.REVERSAL_DAYS)
    
    if price_data.empty: return
        
    scores = Scoring.get_scores(price_data)
    candidates = PostFiltering.filter_candidates(scores)
    final_targets = RiskControl.check_rsrs(context, candidates)
    Execution.execute(context, final_targets)
