# 策略名称：多因子宽基ETF择时轮动 (模块化重构版)
# 原始逻辑参考：ETF_yj15.py (作者: 养家大哥 v15.0)
# 重构作者：Antigravity
# 说明：
# 1. 核心逻辑：乖离率动量排名 + RSRS大盘择时 + 盘中60分钟动态止损。
# 2. 架构特点：将原策略的全局列表状态维护改为无状态计算(每次重算)，增强鲁棒性。
# 3. 包含特殊的盘中风控 (11:25, 11:27 运行)。

import numpy as np
import pandas as pd
from jqdata import *
from jqlib.technical_analysis import *

class Config:
    # ==================== 交易环境 ====================
    AVOID_FUTURE_DATA = True
    USE_REAL_PRICE = True
    
    # 费率 (参考前序优化的配置)
    SLIPPAGE_FUND = 0.001
    COMMISSION_FUND = 0.0001
    
    # ==================== 标的池 ====================
    # 宽基ETF为主
    ETF_POOL = [
        '510300.XSHG', # 沪深300
        '510050.XSHG', # 上证50
        '159949.XSHE', # 创业板500 (原代码注释如此，可能是创业板50?) 159949是创业板50
        '159928.XSHE', # 消费ETF
    ]
    REF_STOCK = '000300.XSHG' # 择时参考标的
    
    HOLD_NUM = 1
    
    # ==================== 动量排名参数 ====================
    MOMENTUM_DAY = 20       # 动量线性回归天数
    BIAS_N = 90             # 乖离率均线天数
    SWITCH_FACTOR = 1.04    # 换仓阈值 (新标的分数需超过旧标的 4%)
    MOTION_1DIFF_THR = 19   # 动量一阶导(变化率)阈值 (防急跌)
    RAISER_THR = 4.8        # 单日涨幅阈值 (防暴涨)
    
    # ==================== RSRS择时参数 ====================
    RSRS_N = 18             # 回归计算斜率的窗口
    RSRS_M = 600            # Z-Score 历史回溯窗口
    RSRS_K = 8              # RSRS得分的斜率窗口
    
    # 择时阈值
    SCORE_THR = -0.68       # RSRS得分买入阈值
    SCORE_FALL_THR = -0.43  # RSRS得分下跌趋势卖出阈值
    IDEX_SLOPE_RAISE_THR = 12 # 大盘斜率强势阈值
    
    # ==================== 盘中风损参数 ====================
    LOSS_N = 20             # 60分钟线 MA20
    LOSS_FACTOR = 1.005     # 相对昨日收盘价的下跌容忍度
    
    # ==================== 调度时间 ====================
    TIME_PREPARE = '09:00'  # 信号准备
    TIME_SELL = '09:30'     # 卖出执行
    TIME_BUY = '09:35'      # 买入执行
    TIME_INTRADAY = ['11:25', '11:27'] # 盘中检查

# ==================== 初始化 ====================
def initialize(context):
    set_benchmark(Config.REF_STOCK)
    set_option('use_real_price', Config.USE_REAL_PRICE)
    set_option("avoid_future_data", Config.AVOID_FUTURE_DATA)
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    
    set_slippage(FixedSlippage(Config.SLIPPAGE_FUND), type='fund')
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=Config.COMMISSION_FUND, 
        close_commission=Config.COMMISSION_FUND, 
        close_today_commission=0, min_commission=0
    ), type='fund')
    
    g.etf_pool = Config.ETF_POOL
    g.hold_stock = None  # 记录当前核心持仓
    
    # 信号存储
    g.timing_signal = 'KEEP' # BUY, SELL, KEEP
    g.check_out_list = []    # [code, score, adr]
    
    # 注册定时任务
    run_daily(task_prepare, time=Config.TIME_PREPARE)
    run_daily(task_sell, time=Config.TIME_SELL)
    run_daily(task_buy, time=Config.TIME_BUY)
    
    for t in Config.TIME_INTRADAY:
        run_daily(task_intraday_check, time=t)
    
    # 开盘前检查止损 (原策略有 check_lose time='open'，其实就是盘前/盘初)
    # 这里合并到 9:30 之前的 prepare 也没问题，或者单独列出
    # 原策略 check_lose 逻辑是：如果亏损 > 90% (ret <= -90)，止损。这几乎不可能触发，除非发生灾难。
    # 我们保留这个逻辑。
    run_daily(check_extreme_loss_protection, 'open')

# ==================== 基础计算函数 ====================
def get_ols_slope_r2(x, y):
    """
    计算线性回归斜率和R2
    x: np.array (比如 range)
    y: np.array (价格序列)
    """
    try:
        slope, intercept = np.polyfit(x, y, 1)
        # R2 计算
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        return slope, r2
    except:
        return 0, 0

def get_ols_slope(y):
    """快速计算斜率"""
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    return slope

# ==================== 择时逻辑模块 (RSRS) ====================
def calculate_rsrs_signal(context):
    """
    计算复杂的RSRS择时信号 (无状态版，每次回溯计算)
    返回: 'BUY', 'SELL', 'KEEP'
    """
    # 1. 准备数据
    # 需要足够计算 M 个 RSRS Score，每个 Score 需要 N 天，再加上算 Score 斜率的 K 天
    # 总长度 = M + N + K + 缓冲
    fetch_len = Config.RSRS_M + Config.RSRS_N + Config.RSRS_K + 20
    
    data = attribute_history(Config.REF_STOCK, fetch_len, '1d', ['high', 'low', 'close'], skip_paused=True)
    if len(data) < Config.RSRS_M + Config.RSRS_N:
        return 'KEEP' # 数据不足维持现状
        
    highs = data['high'].values
    lows = data['low'].values
    close = data['close'].values
    
    # 2. 计算斜率序列 (Slope Series)
    # 我们需要计算最近 M+K 个时间点的 RSRS Slope and R2
    # 也就是 rolling(window=N).apply(ols)
    
    # 优化：为了避免数千次循环，只计算我们需要的段
    # 我们最终需要计算 rsrs_score_history 的 zscore
    # rsrs_score_history 需要最近 K 个点来算斜率，以及最近1个点来判阈值
    # 但是 Z-Score 的标准化本身需要 M 个点的分布。
    # 所以我们需要最近 M+K 个 Slope值。
    
    needed_slopes_count = Config.RSRS_M + Config.RSRS_K
    slopes = []
    r2s = []
    
    # 从倒数第 needed_slopes_count 个点开始算
    start_idx = len(highs) - Config.RSRS_N - needed_slopes_count + 1
    if start_idx < 0: start_idx = 0
    
    # 滚动计算 N 日回归
    for i in range(start_idx, len(highs) - Config.RSRS_N + 1):
        h = highs[i : i + Config.RSRS_N]
        l = lows[i : i + Config.RSRS_N]
        s, r2 = get_ols_slope_r2(l, h) # RSRS是 Low vs High 回归
        slopes.append(s)
        r2s.append(r2)
        
    slopes = np.array(slopes)
    r2s = np.array(r2s)
    
    if len(slopes) < Config.RSRS_M: return 'KEEP'
    
    # 3. 计算 RSRS Score
    # Score = ZScore(Slope) * R2
    # ZScore 是相对于过去 M 天的分布
    
    rsrs_scores = []
    # 我们只需要最后 K 个 Score 来算 trend，以及当前最新的 Score
    # 但每个 Score 的 ZScore 依赖其之前的 M 个 Slope
    
    calc_range = range(len(slopes) - Config.RSRS_K, len(slopes)) # 最后 K 个
    
    for i in calc_range:
        # 取过去 M 个 (含当前 i)
        # history slice: slopes[i-M+1 : i+1]
        sub_slopes = slopes[i - Config.RSRS_M + 1 : i + 1]
        mean = np.mean(sub_slopes)
        std = np.std(sub_slopes)
        zscore = (slopes[i] - mean) / std if std != 0 else 0
        rsrs_scores.append(zscore * r2s[i])
        
    rsrs_scores = np.array(rsrs_scores)
    
    # 4. 计算指标衍生值
    current_rsrs_score = rsrs_scores[-1]
    
    # RSRS Slope (Score 的斜率)
    rsrs_slope = get_ols_slope(rsrs_scores) # 对最后 K 个分值回归
    
    # 大盘价格斜率 (Idex Slope) - 最近8天
    idex_slope = get_ols_slope(close[-8:])
    
    # log.info(f"RSRS: Score={current_rsrs_score:.3f}, Slope={rsrs_slope:.3f}, IdexSlope={idex_slope:.3f}")
    
    # 5. 威廉指标 (WR) 补充判断
    # 引入 jqlib 中的 WR，注意它需要 context.previous_date 可能需要调整
    # jqlib.technical_analysis.WR 默认 calculate for current end_date if include_now=True
    # 这里直接用数据算简版 WR 即可，或者调用库
    # 这里选择简单调用库函数，如果报错则 skip
    wr_flag = False
    try:
        dic_wr2, dic_wr1 = WR([Config.REF_STOCK], check_date=context.current_dt, N=21, N1=14, unit='1d', include_now=True)
        w1 = dic_wr1[Config.REF_STOCK]
        w2 = dic_wr2[Config.REF_STOCK]
        if w1 >= 97 and w2 >= 97:
            wr_flag = True
    except:
        pass
        
    # 6. 综合信号判定
    if wr_flag: return 'BUY'
    
    # 卖出条件
    if rsrs_slope < 0 and current_rsrs_score > 0: return 'SELL'
    if idex_slope < 0 and rsrs_slope > 0 and current_rsrs_score < Config.SCORE_FALL_THR: return 'SELL'
    
    # 买入条件
    if idex_slope > Config.IDEX_SLOPE_RAISE_THR and rsrs_slope > 0: return 'BUY'
    if current_rsrs_score > Config.SCORE_THR: return 'BUY'
    
    return 'SELL' # 默认 Keep or Sell? 原策略最后 else return "SELL"

# ==================== 选股排名模块 ====================
def calculate_stock_score(etf, context):
    """计算单个ETF的乖离动量分"""
    # 需要: BiasN + MomentumDay + 1 (for current) + 1 (for calc change rate)
    fetch_len = Config.BIAS_N + Config.MOMENTUM_DAY + 5
    data = attribute_history(etf, fetch_len, '1d', ['close'])
    if len(data) < fetch_len: return None
    
    closes = data['close']
    
    # 计算 Bias 序列
    # Bias = Close / MA(BiasN)
    ma = closes.rolling(Config.BIAS_N).mean()
    bias_series = (closes / ma)
    
    # 我们需要今天的动量分数，和昨天的动量分数（算变化率）
    
    def calc_mom(series_slice):
        # 归一化: bias / bias[0]
        # series_slice 长度为 MOMENTUM_DAY
        y = series_slice.values / series_slice.values[0]
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return slope * 10000
        
    # 今天的分数 (取最后 M 天)
    bias_today = bias_series.iloc[-Config.MOMENTUM_DAY:]
    score_today = calc_mom(bias_today)
    
    # 昨天的分数 (取 [-M-1 : -1])
    bias_yesterday = bias_series.iloc[-Config.MOMENTUM_DAY-1 : -1]
    score_yesterday = calc_mom(bias_yesterday)
    
    # 动量变化率 (一阶导)
    change_rate = score_today - score_yesterday
    
    # 单日涨幅 (ADR)
    adr = 100 * (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
    
    return {
        'etf': etf,
        'score': score_today,
        'change_rate': change_rate,
        'adr': adr
    }

def get_best_target(context):
    """获取综合排名第一的目标"""
    scores = []
    for etf in g.etf_pool:
        res = calculate_stock_score(etf, context)
        if res:
            # 换仓因子缓冲
            mult = Config.SWITCH_FACTOR if etf == g.hold_stock else 1.0
            res['final_score'] = res['score'] * mult
            scores.append(res)
            
    if not scores: return None
    
    # 排序
    scores.sort(key=lambda x: x['final_score'], reverse=True)
    best = scores[0]
    
    # 检查风控条件 (动量变化率过大 或 单日暴涨)
    if best['change_rate'] > Config.MOTION_1DIFF_THR or best['adr'] > Config.RAISER_THR:
        log.info(f"风控触发: {best['etf']} ChangeRate={best['change_rate']:.1f}, ADR={best['adr']:.1f}")
        return None # 触发风控，今日空仓
        
    return best

# ==================== 任务执行模块 ====================
def task_prepare(context):
    """09:00 - 计算信号"""
    # 1. 计算择时信号
    timing = calculate_rsrs_signal(context)
    g.timing_signal = timing
    
    # 2. 计算选股排名
    best_target = get_best_target(context)
    if best_target:
        g.check_out_list = [best_target['etf'], best_target['score'], best_target['adr']]
    else:
        g.check_out_list = []
        # 如果选股风控触发，强制 SELL
        g.timing_signal = 'SELL'
        
    log.info(f"信号准备: Timing={g.timing_signal}, Target={g.check_out_list}")

# ==================== 交易辅助模块 ====================
# ==================== 交易辅助模块 ====================
def smart_order_target_value(context, security, value):
    """
    智能下单: 检查交易数量是否满足最小单位 (100股)
    避免因微小调仓导致的 'Order amount < 100' 报错
    """
    if value == 0:
        # 清仓逻辑: 如果有持仓，直接卖完
        if context.portfolio.positions[security].total_amount > 0:
            return order_target_value(security, 0)
        return None
        
    current_data = get_current_data()
    price = current_data[security].last_price
    if price == 0: return None
    
    # 获取当前持仓市值
    current_value = context.portfolio.positions[security].value
    
    # 计算目标变动额
    delta_value = value - current_value
    
    # 变动股数 (绝对值)
    delta_amount = abs(delta_value / price)
    
    # 如果变动少于100股，忽略 (防止报错)
    if delta_amount < 100:
        log.info(f"忽略微小调仓: {security} 变动{delta_amount:.1f}股 < 100股")
        return None
        
    return order_target_value(security, value)

def task_sell(context):
    """09:30 - 执行卖出"""
    # 如果择时信号是 SELL，清仓
    if g.timing_signal == 'SELL':
        for etf in list(context.portfolio.positions.keys()):
            smart_order_target_value(context, etf, 0)
            log.info(f"择时卖出: {etf}")
            g.hold_stock = None

def task_buy(context):
    """09:35 - 执行买入/换仓"""
    if g.timing_signal not in ['BUY', 'KEEP']: return
    if not g.check_out_list: return
    
    target_etf = g.check_out_list[0]
    
    # 检查是否需要换仓
    current_holdings = list(context.portfolio.positions.keys())
    
    # 如果当前持有非目标标的，卖出
    for etf in current_holdings:
        if etf != target_etf:
            smart_order_target_value(context, etf, 0)
            log.info(f"换仓卖出: {etf}")
            
    # 买入目标
    # 全仓买入 (HOLD_NUM=1)
    if target_etf not in current_holdings:
        # 简单全仓买入
        smart_order_target_value(context, target_etf, context.portfolio.total_value)
        log.info(f"买入目标: {target_etf}")
        g.hold_stock = target_etf
    else:
        g.hold_stock = target_etf
        # 也可以做一次 rebalance 确保满仓
        smart_order_target_value(context, target_etf, context.portfolio.total_value)

def task_intraday_check(context):
    """11:25/27 - 盘中动态止损 (60分钟线)"""
    if not g.hold_stock: return
    
    etf = g.hold_stock
    # 如果已经卖了，就不管了
    if context.portfolio.positions[etf].total_amount == 0: return

    # 获取60分钟数据
    # 需要 LOSS_N + 2 个 60m bar
    try:
        bars = attribute_history(etf, Config.LOSS_N + 5, '60m', ['close'])
        if len(bars) < Config.LOSS_N: return
        
        closes = bars['close']
        ma = closes.rolling(Config.LOSS_N).mean()
        
        current_close = closes.iloc[-1] 
        current_ma = ma.iloc[-1]
        
        # 相比昨日收盘价
        yesterday_close = attribute_history(etf, 1, '1d', ['close'])['close'].iloc[-1]
        
        cur_price = get_current_data()[etf].last_price
        
        # 止损条件:
        # 1. 60分钟收盘价 < 60分钟MA20 (趋势破位)
        cond1 = current_close < current_ma
        
        # 11:25 只预警
        if context.current_dt.minute == 25:
            if cond1:
                log.info(f"盘中可能止损预警: {etf}, 60m_Price={current_close:.3f} < MA={current_ma:.3f}")
            return
            
        # 11:27 执行
        # 2. 当前价 * 1.005 <= 昨日收盘价 (今日未上涨或下跌)
        cond2 = cur_price * Config.LOSS_FACTOR <= yesterday_close
        
        if cond1 and cond2:
            smart_order_target_value(context, etf, 0)
            log.info(f"盘中动态止损触发: {etf}, 60m_Price={current_close:.3f}, MA={current_ma:.3f}, DayRet={(cur_price/yesterday_close - 1):.2%}")
            g.hold_stock = None
    except Exception as e:
        pass

def check_extreme_loss_protection(context):
    """极端亏损止损 (原策略ret <= -90%逻辑)"""
    for etf, pos in context.portfolio.positions.items():
        if pos.total_amount > 0:
            ret = (pos.price / pos.avg_cost) - 1
            if ret <= -0.9:
                smart_order_target_value(context, etf, 0)
                log.warn(f"极端止损触发: {etf} 亏损90%")
