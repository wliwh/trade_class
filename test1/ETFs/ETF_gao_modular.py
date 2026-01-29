# 策略名称：高收益ETF轮动策略（模块化重构版）
# 原始逻辑参考：ETF_gao.py
# 重构作者：Antigravity
# 说明：
# 1. 严格保留原策略逻辑：R2动量评分、均线过滤、成交量异常过滤、3日跌幅限制。
# 2. 严格保留原交易时点：10:29卖出，10:30买入。
# 3. 采用模块化架构：Config -> Data -> Logic -> Execution

import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta
from jqdata import *

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
    ETF_POOL = [
        # 境外
        "513100.XSHG",  # 纳指ETF
        "159509.XSHE",  # 纳指科技ETF
        "513520.XSHG",  # 日经ETF
        "513030.XSHG",  # 德国ETF
        # 商品
        "518880.XSHG",  # 黄金ETF
        "159980.XSHE",  # 有色ETF
        "159985.XSHE",  # 豆粕ETF
        "159981.XSHE",  # 能源化工ETF
        # "159870.XSHE", # 化工
        
        "501018.XSHG",  # 南方原油
        # 债券
        "511090.XSHG",  # 30年国债ETF
        # 国内
        "513130.XSHG",  # 恒生科技
        "513690.XSHG",  # 港股红利
        
        "510180.XSHG",   #上证180
        "159915.XSHE",   #创业板ETF
        
        "510410.XSHG",   #资源
        "515650.XSHG",   #消费50
        "512290.XSHG",   #生物医药
        "588120.XSHG",   #科创100
        "515070.XSHG",   #人工智能ETF
        
        "159851.XSHE",   #金融科技
        "159637.XSHE",   #新能源车
        "516160.XSHG",   #新能源
        
        "159550.XSHE",   #互联网ETF
        "512710.XSHG",   #军工ETF
        "159692.XSHE",   #证券
        "512480.XSHG",   #半导体
        "515250.XSHG",   #智能汽车
        "159378.XSHE",   #通用航空
        "516510.XSHG",   #云计算
        "515050.XSHG",   #5G通信
        "159995.XSHE",   #芯片 
        "515790.XSHG",   #光伏
        "515000.XSHG"    #科技
    ]
    
    # 策略参数
    HOLD_COUNT = 1          # 持仓数量
    M_DAYS = 25             # 动量参考天数
    MIN_MONEY = 500         # 最小交易额
    
    # 评分筛选参数
    MIN_SCORE = 0.0
    MAX_SCORE = 5.0
    DROP_3DAY_LIMIT = 0.95  # 3日跌幅限制
    
    # 均线过滤
    ENABLE_MA_FILTER = False # 原代码默认False，若需启用改为True
    MA_FILTER_DAYS = 20
    
    # 成交量检测
    ENABLE_VOLUME_CHECK = True
    VOLUME_LOOKBACK = 5
    VOLUME_THRESHOLD = 1.0


# ==================== 初始化 ====================
def initialize(context):
    set_benchmark(Config.BENCHMARK)
    set_option("avoid_future_data", Config.AVOID_FUTURE_DATA)
    set_option("use_real_price", Config.USE_REAL_PRICE)
    
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    
    set_slippage(FixedSlippage(Config.SLIPPAGE_FUND), type="fund")
    set_slippage(FixedSlippage(Config.SLIPPAGE_STOCK), type="stock")
    
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, # 0.001
        open_commission=Config.COMMISSION_STOCK_OPEN, 
        close_commission=Config.COMMISSION_STOCK_CLOSE, 
        close_today_commission=0, min_commission=Config.COMMISSION_MIN
    ), type="stock")
    
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0, close_commission=0, 
        close_today_commission=0, min_commission=0
    ), type="mmf")
    
    # 全局变量
    g.etf_pool = Config.ETF_POOL
    g.target_list = [] # 存储每日计算出的目标列表
    
    # 定时任务 (与原逻辑一致)
    run_daily(etf_trade_sell, "10:29") # 卖出
    run_daily(etf_trade_buy, "10:30")  # 买入
    run_daily(end_trade, "14:59")      # 盘后处理

# ==================== 数据获取模块 ====================
def get_safe_price(security, context):
    """获取防未来当前价"""
    try:
        current_dt = context.current_dt
        # 取最近1分钟bar的close，如果是回测，context.current_dt是分钟开始时间还是结束时间取决于平台
        # 稳妥起见，取前一分钟
        data = get_price(security, end_date=current_dt, frequency='1m', fields=['close'], count=2, panel=False)
        if data is not None and not data.empty:
            return data['close'].iloc[-1]
        return 0
    except:
        return 0

def get_current_vol_sum(security, context):
    """获取当日截止当前的累计成交量"""
    try:
        today = context.current_dt.date()
        df = get_price(security, start_date=today, end_date=context.current_dt, frequency='1m', fields=['volume'], panel=False)
        if df is not None and not df.empty:
            return df['volume'].sum()
        return 0
    except:
        return 0

# ==================== 逻辑计算模块 ====================
def calculate_ma_filter(stocks, days=20):
    if not stocks: return []
    filtered = []
    
    # 批量获取数据优化
    hists = attribute_history(stocks, days, '1d', ['close'], skip_paused=True)
    
    # 获取当前快照
    current_data = get_current_data()
    
    for stock in stocks:
        if stock not in hists or len(hists[stock]) < days: continue
        ma_price = hists[stock]['close'].mean()
        cur_price = current_data[stock].last_price
        
        if cur_price >= ma_price:
            filtered.append(stock)
            
    return filtered

def check_volume_anomaly(etf, context, lookback=5, threshold=1.0):
    """检测成交量是否异常放量"""
    try:
        # 1. 历史平均成交量
        hist = attribute_history(etf, lookback, '1d', ['volume'], skip_paused=True)
        if len(hist) < lookback: return False, 0
        avg_vol = hist['volume'].mean()
        if avg_vol == 0: return False, 0
        
        # 2. 当日实时成交量
        cur_vol = get_current_vol_sum(etf, context)
        
        ratio = cur_vol / avg_vol
        if ratio > threshold:
            return True, ratio
        return False, ratio
    except:
        return False, 0

def get_etf_score(etf, context):
    try:
        current_data = get_current_data()
        cur_price = current_data[etf].last_price
        if math.isnan(cur_price): return None
        
        # 获取历史数据
        hist = attribute_history(etf, Config.M_DAYS, '1d', ['close'])
        if len(hist) < Config.M_DAYS: return None
        
        prices = np.append(hist['close'].values, cur_price)
        
        # 1. 3日跌幅限制
        if len(prices) >= 4:
            # 只要有一天跌幅超过5% (ratio < 0.95)，就排除
            # 原逻辑：min(p[-1]/p[-2], ...) < 0.95
            drops = [prices[-1]/prices[-2], prices[-2]/prices[-3], prices[-3]/prices[-4]]
            if min(drops) < Config.DROP_3DAY_LIMIT:
                # log.info(f"排除 {etf}: 近3日有大跌")
                return None

        # 2. 计算 R2 和 年化收益
        y = np.log(prices)
        x = np.arange(len(y))
        w = np.linspace(1, 2, len(y)) # 线性加权
        
        slope, intercept = np.polyfit(x, y, 1, w=w)
        ann_ret = math.exp(slope * 250) - 1
        
        y_pred = slope * x + intercept
        ss_res = np.sum(w * (y - y_pred) ** 2)
        ss_tot = np.sum(w * (y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        
        score = ann_ret * r2
        
        if Config.MIN_SCORE < score < Config.MAX_SCORE:
            return {'etf': etf, 'score': score, 'r2': r2, 'ret': ann_ret}
        
        return None
        
    except Exception as e:
        log.warn(f"Error scoring {etf}: {e}")
        return None

def get_target_list(context):
    """筛选目标ETF列表"""
    pool = g.etf_pool
    
    # 1. 均线过滤
    if Config.ENABLE_MA_FILTER:
        pool = calculate_ma_filter(pool, Config.MA_FILTER_DAYS)
        log.info(f"均线过滤后剩余: {len(pool)}")
        
    # 2. 评分排序
    scored_list = []
    for etf in pool:
        res = get_etf_score(etf, context)
        if res: scored_list.append(res)
        
    # 按分数降序
    scored_list.sort(key=lambda x: x['score'], reverse=True)
    
    # 记录日志
    if scored_list:
        top3_info = [f"{x['etf']}:{x['score']:.3f}" for x in scored_list[:3]]
        log.info(f"Top 3: {top3_info}")
    
    # 返回纯代码列表
    return [x['etf'] for x in scored_list]


# ==================== 交易执行模块 ====================
def smart_order_target(security, value, context):
    """智能下单封装"""
    current_data = get_current_data()
    if current_data[security].paused: return False
    
    # 涨跌停检查
    price = current_data[security].last_price
    if price >= current_data[security].high_limit or price <= current_data[security].low_limit:
        return False
        
    # 计算目标股数 (向下取整到100)
    if price == 0: return False
    target_amount = (int(value / price) // 100) * 100
    
    # 最小交易额检查
    pos = context.portfolio.positions[security]
    current_val = pos.total_amount * price
    diff_val = abs(value - current_val)
    
    if diff_val < Config.MIN_MONEY and value > 0: # 只有开仓/调仓时检查，清仓不受限
        return False
        
    order_target_value(security, value)
    return True

# ---------- 卖出逻辑 (10:29) ----------
def etf_trade_sell(context):
    log.info("=== 开始执行卖出逻辑 (10:29) ===")
    
    # 1. 计算今日目标
    # 这里我们只计算 filter 结果，真正根据成交量剔除在买入时做，或者在这里如果持仓放量也卖
    raw_targets = get_target_list(context)
    g.target_list = raw_targets # 存给买入逻辑用
    
    # 截取前 N 名作为“理想持仓”
    ideal_targets = raw_targets[:Config.HOLD_COUNT]
    
    current_positions = list(context.portfolio.positions.keys())
    
    # 2. 检查持仓是否需要卖出
    for etf in current_positions:
        pos = context.portfolio.positions[etf]
        if pos.total_amount == 0: continue
        
        should_sell = False
        reason = ""
        
        # A. 成交量异常放量 -> 强制卖出
        if Config.ENABLE_VOLUME_CHECK:
            is_anomaly, ratio = check_volume_anomaly(etf, context, Config.VOLUME_LOOKBACK, Config.VOLUME_THRESHOLD)
            if is_anomaly:
                should_sell = True
                reason = f"放量异常(Ratio:{ratio:.2f})"
        
        # B. 不在目标池中
        if etf not in ideal_targets and not should_sell:
            should_sell = True
            reason = "不在目标列表"
            
        # C. 持仓超标且排名靠后 (针对持有多个标的的情况，本策略默认HOLD_COUNT=1)
        if etf in ideal_targets:
            # 如果在目标里，但排名在 HOLD_COUNT 之外 (例如持仓多了)
            if etf not in ideal_targets[:Config.HOLD_COUNT]:
                should_sell = True
                reason = "排名下降"
        
        if should_sell:
            log.info(f"卖出 {etf}: {reason}")
            smart_order_target(etf, 0, context)

# ---------- 买入逻辑 (10:30) ----------
def etf_trade_buy(context):
    log.info("=== 开始执行买入逻辑 (10:30) ===")
    
    # 1. 获取候选名单 (复用卖出时计算的，或者重新计算)
    # 建议重新读取 g.target_list，若为空则重新算
    targets = g.target_list
    if not targets:
        log.info("无目标标的")
        return

    # 2. 筛选最终买入目标 (剔除放量异常的)
    final_buy_targets = []
    
    # 只看前几名候选，甚至可以多看几个替补
    candidates = targets 
    
    for etf in candidates:
        if len(final_buy_targets) >= Config.HOLD_COUNT: break
        
        # 成交量检查
        if Config.ENABLE_VOLUME_CHECK:
            is_anomaly, ratio = check_volume_anomaly(etf, context, Config.VOLUME_LOOKBACK, Config.VOLUME_THRESHOLD)
            if is_anomaly:
                log.info(f"剔除买入目标 {etf}: 放量异常(Ratio:{ratio:.2f})")
                continue
        
        final_buy_targets.append(etf)
        
    if not final_buy_targets:
        log.info("无有效买入目标 (全被过滤)")
        return
        
    log.info(f"最终买入目标: {final_buy_targets}")
    
    # 3. 执行买入
    total_value = context.portfolio.total_value
    per_value = total_value / len(final_buy_targets)
    
    available_cash = context.portfolio.available_cash
    
    for etf in final_buy_targets:
        smart_order_target(etf, per_value, context)

# ---------- 盘后处理 ----------
def end_trade(context):
    pass
