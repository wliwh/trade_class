# 策略名称：核心资产轮动-添油加醋版（模块化重构）
# 说明：
# 1. 核心逻辑：短期动量(25日) + 长期反转(200日) 结合打分。
# 2. 独特风控：
#    - 极值差离过滤 (Score Spread): 第一名和最后一名分差需在 [0.1, 15] 之间。
#    - 个股RSRS择时: 每个ETF单独计算RSRS，若Beta低于阈值则认为是顶部不买入。
# 3. 架构：Config -> Data -> Logic -> Execution

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
    
    # ==================== 策略核心参数 ====================
    HOLD_COUNT = 1          # 持仓数量
    
    # 动量/反转参数
    M_DAYS = 25             # 短期动量天数
    L_DAYS = 200            # 长期反转天数 (原策略 m_days * 8)
    L_WEIGHT = 6            # 长期因子权重分母 (Score = Short - Long / 6)
    
    # 极值差离过滤
    SPREAD_MIN = 0.1
    SPREAD_MAX = 15.0
    
    # RSRS 参数
    RSRS_N = 18             # RSRS回归周期
    RSRS_M = 250            # RSRS均值标准差回溯周期
    RSRS_STD_MULT = 2       # 阈值 = Mean - 2 * Std

# ==================== 初始化 ====================
def initialize(context):
    set_benchmark(Config.BENCHMARK)
    set_option('use_real_price', Config.USE_REAL_PRICE)
    set_option("avoid_future_data", Config.AVOID_FUTURE_DATA)
    
    set_slippage(FixedSlippage(Config.SLIPPAGE_FUND), type="fund")
    set_slippage(FixedSlippage(Config.SLIPPAGE_STOCK), type="stock")
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=Config.COMMISSION_STOCK_OPEN, 
        close_commission=Config.COMMISSION_STOCK_CLOSE, 
        close_today_commission=0, min_commission=Config.COMMISSION_MIN
    ), type="stock")
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0, close_commission=0, 
        close_today_commission=0, min_commission=0
    ), type="mmf")

    log.set_level('system', 'error')
    
    g.etf_pool = EXECUTION_ETF_POOLS_PLACEHOLDER
    
    # 每日 09:30 执行 (跟原策略一致)
    run_daily(trade, EXECUTION_TIME_PLACEHOLDER)

# ==================== 逻辑计算模块 ====================
def calculate_momentum_score(history_data):
    """计算单个ETF的动量得分: Annualized Return * R2"""
    try:
        # 预处理
        y = np.log(history_data['close'].values)
        x = np.arange(len(y))
        
        # 线性回归
        slope, intercept = np.polyfit(x, y, 1)
        
        # 计算年化收益
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        
        # 计算 R2
        # 原策略公式: 1 - (SS_res / ((n-1) * var(y)))
        # 注意: np.var(ddof=1) calculate unbiased variance
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred)**2)
        var_y = np.var(y, ddof=1)
        r_squared = 1 - (ss_res / ((len(y) - 1) * var_y)) if var_y > 0 else 0
        
        return annualized_returns * r_squared
    except:
        return 0

def get_combined_scores(etf_pool):
    """获取综合评分表 (短期动量 - 长期反转/6)"""
    scores = {}
    
    for etf in etf_pool:
        # 1. 获取短期数据
        df_short = attribute_history(etf, Config.M_DAYS, '1d', ['close'])
        if len(df_short) < Config.M_DAYS:
            scores[etf] = -999
            continue
        score_short = calculate_momentum_score(df_short)
        
        # 2. 获取长期数据 (用于反转)
        df_long = attribute_history(etf, Config.L_DAYS, '1d', ['close'])
        if len(df_long) < Config.L_DAYS:
            # 如果数据不够长，暂时用短期分代替或给低分，原逻辑隐含需要足够数据
            scores[etf] = -999
            continue
        score_long = calculate_momentum_score(df_long)
        
        # 3. 综合得分: 动量 - 反转
        final_score = score_short - (score_long / Config.L_WEIGHT)
        scores[etf] = final_score
        
    return scores

def calculate_beta(high_series, low_series):
    """计算 Beta (High vs Low 斜率)"""
    try:
        # polyfit(x, y, 1) -> fits y = kx + b
        # 这里 x=low, y=high
        slope, _ = np.polyfit(low_series, high_series, 1)
        return slope
    except:
        return 0

def check_rsrs_filter(etf, context):
    """
    RSRS 择时检查 (个股)
    Return: True(可买), False(到顶不可买)
    """
    try:
        # 获取足够长的历史数据来计算 Beta 分布
        # 需要 RSRS_M(250) 个样本，每个样本由 RSRS_N(18) 天数据回归得到
        # 实际上如果每次rolling算比较慢。
        # 原策略逻辑：
        # 1. 取 250 天数据
        # 2. 遍历每一天，取过去18天算beta，存入 betaList
        # 3. 计算 betaList 的 mean 和 std
        # 4. 当前 beta > mean - 2*std 则通过
        
        # 为了性能和数据完整性，我们取 M + 20 天以防万一
        total_days = Config.RSRS_M + Config.RSRS_N
        data = attribute_history(etf, total_days, '1d', ['high', 'low'])
        if len(data) < total_days: return True # 数据不足默认通过? 或者False? 原策略其实会报错，这里默认过
        
        highs = data['high'].values
        lows = data['low'].values
        
        beta_list = []
        # 计算过去 M 个 Beta 值
        # 比如今天要算 Beta(t), Beta(t-1) ... Beta(t-M)
        # 倒序或者正序遍历
        
        # 优化：只取最近250个点作为分布参考? 原策略是从 i=0 到 len-21
        # 原策略: attribute_history(250), loop (0, 250-21) -> 约230个样本
        # 实际上是拿过去一年的Beta分布来衡量当前的Beta
        
        # 重新实现原逻辑的精准复刻：
        # etf_data = attribute_history(etf, 250, '1d')
        # loop i from 0 to len-21 (即 229次)
        # slice i:i+20 (其实原代码写的是20，不是18? 让我们看Config)
        # 原代码: hl = attribute_history(etf, 18...) (行95)用于当前
        # 原代码: countBeta用的是 250天数据, slice 20天回归.
        # 此时出现 参数不一致: 行96用的是18天High/Low回归，行130(countBeta)用的是20天
        # 我们这里统一逻辑或严格照搬。
        # 照搬: getBeta(Current)用18天，Threshold用20天rolling。
        
        # 1. 计算 Threshold
        ref_data = attribute_history(etf, 250, '1d', ['high', 'low'])
        ref_betas = []
        for i in range(len(ref_data) - 21):
            sub_h = ref_data['high'].iloc[i:i+20]
            sub_l = ref_data['low'].iloc[i:i+20]
            b = calculate_beta(sub_h, sub_l)
            ref_betas.append(b)
            
        if not ref_betas: return True
        
        mu = np.mean(ref_betas)
        sigma = np.std(ref_betas)
        threshold = mu - Config.RSRS_STD_MULT * sigma
        
        # 2. 计算当前 Beta (用18天)
        cur_data = attribute_history(etf, 18, '1d', ['high', 'low'])
        cur_beta = calculate_beta(cur_data['high'], cur_data['low'])
        
        # 判决
        if cur_beta > threshold:
            return True
        else:
            # log.info(f"RSRS过滤 {etf}: Beta {cur_beta:.3f} <= Thr {threshold:.3f}")
            return False
            
    except Exception as e:
        log.warn(f"RSRS Error {etf}: {e}")
        return True # 出错默认放行

# ==================== 交易执行模块 ====================
def trade(context):
    # 1. 计算所有标的得分
    scores_dict = get_combined_scores(g.etf_pool)
    
    # 排序
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    if not sorted_items: return
    
    max_score = sorted_items[0][1]
    min_score = sorted_items[-1][1]
    score_spread = max_score - min_score
    
    # 2. 极值差离过滤 (亦即“枪打出头鸟”逻辑)
    target_list = []
    if Config.SPREAD_MIN < score_spread < Config.SPREAD_MAX:
        # 取前 N 名
        candidates = [x[0] for x in sorted_items[:Config.HOLD_COUNT]]
        target_list = candidates
        # log.info(f"分数差 {score_spread:.2f} 达标，初选: {target_list}")
    else:
        log.info(f"分数差 {score_spread:.2f} 异常，空仓")
        target_list = []
        
    # 3. RSRS 择时过滤
    final_targets = []
    for etf in target_list:
        if check_rsrs_filter(etf, context):
            final_targets.append(etf)
        else:
            log.info(f"RSRS风控触发，剔除 {etf}")
            
    # log.info(f"最终目标: {final_targets}")
    
    # 4. 执行交易
    current_holdings = list(context.portfolio.positions.keys())
    
    # 卖出不在目标里的
    for etf in current_holdings:
        if etf not in final_targets:
            order_target_value(etf, 0)
            
    # 买入目标
    buy_count = len(final_targets)
    if buy_count > 0:
        # 假设我们要持有 HOLD_COUNT 只，现在只有 buy_count 只达标
        # 原逻辑: value = cash / (target_num - len(hold)) -> 动态补仓
        # 简化逻辑: 平权分配现有资金
        
        # 原逻辑通过判断持仓数来补仓，比较复杂，我们采用标准的轮动买入逻辑：
        # 如果都在列表里，根据权重调整；如果有新增，买入。
        
        total_value = context.portfolio.total_value
        per_value = total_value / Config.HOLD_COUNT # 始终按目标持仓数分配仓位?
        # 原逻辑行 112: value = available_cash / (target_num - len(hold_list))
        # 这意味着它用剩余现金买新标的，不论旧标的涨跌，不做Rebalance。
        # 我们采用更稳健的做法：对目标列表里的进行 Target Value 调整。
        
        # 调整为: 
        # 如果通过筛选只有 0个 -> 空仓
        # 如果有 1个 -> 满仓 (因为 HOLD_COUNT=1)
        
        for etf in final_targets:
            order_target_value(etf, total_value / len(final_targets))
