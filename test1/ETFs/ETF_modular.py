# 风险及免责提示：该策略由聚宽用户在聚宽社区分享，仅供学习交流使用。
# 标题：模块化ETF轮动策略 (Modular ETF Rotation)
# 基于 ETF_wy.py，整合了 ETF_long, ETF_std_score, ETF_vol, ETF_yj15 的核心功能。
# 策略作者：Antigravity (Based on work by wywy1995, 养家大哥, 慕长风等)

import numpy as np
import pandas as pd
import math
from jqdata import *

class Config:
    # --- 功能开关 ---
    # 1. 启用长周期反转逻辑 (来自 ETF_long.py)
    # 逻辑：在短期得分基础上，减去 1/6 的长周期(200天)得分，避免追高长线已透支的品种。
    ENABLE_LONG_TERM_REVERSAL = False
    REVERSAL_FACTOR = 6.0 # 长周期反转因子

    # 6. 启用分歧度过滤 (来自 ETF_long.py)
    # 逻辑：如果资产池得分极差过小(无主线)或过大(极端分化)，则空仓。
    ENABLE_DISPERSION_FILTER = False
    DISPERSION_MIN = 0.1  # 分歧度最小阈值 (long)
    DISPERSION_MAX = 15   # 分歧度最大阈值 (long)

    # 2. 启用 RSRS 择时 (来自 ETF_yj15 / ETF_long)
    # 逻辑：使用 RSRS 指标 (斜率) 判断大盘趋势。如果低于阈值，则视为下跌趋势，空仓。
    ENABLE_RSRS_TIMING = False
    RSRS_THRESHOLD = 0.9  # RSRS 固定阈值

    # 3a. 启用动态 Beta RSRS 择时 (来自 ETF_long.py)
    # 逻辑：使用 250 天滚动窗口计算 RSRS 的动态阈值 (均值 - 2倍标准差)，而非固定阈值。
    ENABLE_DYNAMIC_BETA_RSRS = False
    BETA_STD_M = 2        # 动态 Beta 标准差倍数
    BETA_DEFAULT = 0.8    # 默认 Beta 阈值
    # 3b. 启用动量一阶导 RSRS 择时 (来自 ETF_yj15)
    # 逻辑：使用动量一阶导作为 RSRS 指标，判断大盘趋势。如果低于阈值，则视为下跌趋势，空仓。
    MEAN_DAY = 20         # RSRS均线周期
    RSRS_N = 18           # RSRS回归周期
    RSRS_M = 600          # RSRS历史统计周期
    RSRS_SCORE_THR = 0.7  # RSRS 标准分阈值 (by yj15/wy)
    MOTION_1DIFF_THR = 19 # 动量一阶导阈值 (yj15)

    # 3. 启用多周期加权评分 (来自 ETF_std_score.py)
    # 逻辑：综合 3, 5, 10, 25(或m_days) 多个周期的得分。若开启，将覆盖默认的单周期评分逻辑。
    ENABLE_MULTI_PERIOD_SCORE = False 
    MULTI_PERIODS = [3, 5, 10] # 多周期评分类别(最后会自动加上M_DAYS)
    MULTI_WEIGHTS = [0.1, 0.2, 0.3, 0.4] # 多周期权重

    # 4. 启用波动率调整 (来自 ETF_vol.py)
    # 逻辑：在得分计算中考虑夏普比率因素，优先选择波动率较小的上涨品种。
    ENABLE_VOLATILITY_ADJUST = False

    # 5. 启用趋势加速过滤 (来自 ETF_yj15.py)
    # 逻辑：计算动量的一阶导数 (加速度)。如果变化率过大，视为短期过热或情绪不稳，进行过滤。
    ENABLE_TREND_ACCELERATION = False    
    INIT_LEN = 5          # 初始化动量窗口长度

    # --- 基础参数 ---
    BENCHMARK = "513100.XSHG"
    M_DAYS = 25           # 动量参考天数
    STOCK_NUM = 1         # 持仓数量
    TREND_DAY = 250       # 交易日数

    ETF_POOLS = {
        4: {
            '518880.XSHG': '黄金', #黄金ETF（大宗商品）
            '513100.XSHG': '纳指', #纳指100（海外资产）
            '159915.XSHE': '成长', #创业板100（成长股，科技股，中小盘）
            '510180.XSHG': '价值', #上证180（价值股，蓝筹股，中大盘）
        },
        7: {
            '518880.XSHG':'黄金',	# 2013-07-29 || 黄金基金,黄金ETF,黄金ETF基金
            '513100.XSHG':'纳指',	# 2013-05-15 || 纳指ETF
            '159907.XSHE':'小盘',	# 2006-09-05 || TMTETF,信息ETF,通信ETF,计算机,科技100,科技ETF,科技50,5GETF,AI智能,中证科技,信息技术ETF
            '510050.XSHG':'大盘',	# 2005-02-23 || 工银上50,MSCI基金,上证50,上50ETF,A50ETF,MSCIA股,景顺MSCI,MSCI中国,天弘300,长三角,添富300,100ETF,ZZ800ETF,800ETF,A50基金,MSCI易基,沪50ETF,工银300,华夏300,HS300ETF,300ETF,综指ETF,180ETF,SZ50ETF,平安300,深100ETF银华,沪深300ETF南方,沪深300ETF,深红利ETF,深证100ETF,广发300
            '511010.XSHG':'国债',	# 2013-03-25 || 5年地债ETF,招商快线ETF,豆粕ETF,货币ETF,5年地债,城投ETF,十年国债,10年地债
            '159920.XSHE':'恒生',	# 2012-10-22 || H股ETF,中概互联,恒生通,恒指ETF,港股100,H股ETF
            '510880.XSHG':'红利',	# 2007-01-18 || 能源ETF基金,中证红利,100红利,能源ETF,有色ETF
        },
        9: {
            '518880.XSHG':'黄金',	# 2013-07-29 || 黄金9999,黄金基金,黄金ETF,工银黄金,上海金,金ETF,黄金ETF基金
            '513100.XSHG':'纳指',	# 2013-05-15 || 法国ETF,东证ETF,纳指生物,日经225,纳斯达克,德国ETF,日经ETF,标普ETF,标普500,标普500ETF,亚太精选ETF,225ETF,日经ETF,纳指ETF,纳斯达克ETF
            '159901.XSHE':'深证100',	# 2006-04-24 || 有色60ETF,深证成指ETF,ZZ500ETF,300成长,广发500,500ETF,稀土ETF,有色金属ETF,化工ETF,钢铁ETF,500指增,国泰500,化工ETF,矿业ETF,畜牧ETF,碳中和E,双碳ETF,畜牧养殖,家电ETF,家电基金,红利质量ETF,化工龙头,中证500ETF博时,增强ETF,中药ETF,稀土ETF,有色ETF,中证500,ETF500,中证500ETF鹏华,家电ETF,养殖ETF,汽车ETF,农业ETF,中证500ETF,化工50,500ETF增强,稀土基金,深成ETF,中药ETF,双碳ETF,碳中和ETF南方,中小100ETF
            '159920.XSHE':'恒生',	# 2012-10-22 || 恒生股息,恒生通,恒生国企ETF,港股100,港股红利,恒指ETF,港股通50,恒生红利ETF,H股ETF,H股ETF
            '510050.XSHG':'50',	# 2005-02-23 || 消费50,国信价值,沪深300ETF南方,HS300,深红利ETF,800ETF,沪深300ETF泰康,MSCI易基,金融ETF,添富300,中证A100ETF基金,红利ETF,HS300E,中国A50,交运ETF,万家50,300增强,上50ETF,上证50,上海国企,工银上50,中国A50ETF,A50ETF,100ETF,上证ETF,天弘300,广发300,工银300,物流ETF,A50基金,SZ50ETF,300增ETF,180ETF,MSCIA50,300ETF,HS300ETF,国货ETF,华夏300,综指ETF,沪深300ETF
            '159907.XSHE':'2000',	# 2011-08-10 || ZZ1000,机器人ETF,1000ETF,中证1000ETF,中证1000ETF易方达,1000基金,1000ETF增强,机器人,教育ETF,1000增强ETF,中证1000,工业母机ETF,机床ETF,1000ETF,国证2000ETF
            '510880.XSHG':'红利',	# 2007-01-18 || 能源ETF,银行指基,银行ETF,红利低波,资源ETF,红利博时,银行ETF,银行基金,电力ETF,绿色电力ETF,银行股基,华夏银基,中证红利,红利100,共赢ETF,100红利,中国国企,红利50,煤炭ETF,银行ETF天弘,红利300,国企共赢ETF,电力ETF
            '511010.XSHG':'国债',	# 2013-03-25 || 汇添富快钱ETF,招商快线ETF,国开ETF,国开债券ETF,国开债ETF,货币ETF,政金债,豆粕ETF,能源化工ETF,有色ETF,5年地债ETF,活跃国债,5年地债,公司债,城投ETF,十年国债,10年地债,短融ETF,0-4地债ETF,国债政金,上证转债,转债ETF
            '512960.XSHG':'央调',	# 2019-01-18 || 央企改革,基建ETF,基建50ETF,基建50,央企创新,创新央企,央创ETF,基建ETF,央企ETF
        }
    }

def initialize(context):
    set_benchmark(Config.BENCHMARK)
    set_option('use_real_price', True)
    set_option("avoid_future_data", True)
    set_slippage(FixedSlippage(0.002))
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.0002, close_commission=0.0002, close_today_commission=0, min_commission=5), type='fund')
    log.set_level('system', 'error')

    g.etf_pool = Config.ETF_POOLS[4]
    g.m_days = Config.M_DAYS
    
    # 初始化动量记录 (用于趋势加速过滤)
    if Config.ENABLE_TREND_ACCELERATION:
        g.stock_motion = initial_stock_motion(g.etf_pool, g.m_days)

    run_daily(trade, '10:00')

# --- 辅助函数：初始化动量 (来自 ETF_yj15) ---
def initial_stock_motion(stock_pool, m_days):
    stock_motion = {}
    init_len = Config.INIT_LEN 
    
    # 一次性获取 max_len 数据，防止多次 I/O
    # Max Need: Day0 (Today-1) needed m_days
    # Day -1 needed m_days ...
    # Day -(init_len-1) needed m_days
    # Total history: m_days + init_len - 1
    # Add buffer 1
    
    fetch_len = m_days + init_len + 5
    
    for stock in stock_pool:
        motion_que = []
        data = attribute_history(stock, fetch_len, '1d', ['close'])
        
        # 从最早需要的那个时间点开始算
        # 我们需要得到 [S_(t-init_len+1), ..., S_(t)] (对应 t=yesterday)
        # 也就是列表最后应有 init_len 个元素
        
        # data index: 0, 1, ..., N-1
        # range logic:
        # i represents the END index of the window
        # First window end: len(data) - init_len
        # Last window end: len(data)
        
        total_data_len = len(data)
        for i in range(init_len):
            # i=0 -> want earliest score
            # end_idx = total_data_len - (init_len - 1 - i)
            # if i=0, end_idx = total - init + 1 
            # if i=init-1 (last), end_idx = total
            
            end_idx = total_data_len - (init_len - 1 - i)
            start_idx = end_idx - m_days
            
            if start_idx < 0: continue
            
            curr_close = data['close'].iloc[start_idx : end_idx]
            
            if len(curr_close) == m_days:
                 score = get_single_momentum_score(curr_close, m_days) * 100
                 motion_que.append(score)
                 
        stock_motion[stock] = motion_que
    return stock_motion

def update_stock_motion(context, stock_pool, data_cache):
    for stock in stock_pool:
        # 使用缓存数据
        if stock in data_cache:
            # 需要最后 m_days + 1 天
            df = data_cache[stock].iloc[-(g.m_days + 1):]
            if len(df) < g.m_days + 1: continue
            
            score = get_single_momentum_score(df['close'], g.m_days) * 100 
            g.stock_motion[stock].append(score)
            if len(g.stock_motion[stock]) > 5:
                g.stock_motion[stock].pop(0)

def get_single_momentum_score(close_series, days):
    # 统一的基础动量计算：年化收益 * R2
    if len(close_series) < days: return 0
    y = np.log(close_series[-days:])
    x = np.arange(len(y))
    
    # 使用 get_ols 统一计算
    intercept, slope, r2 = get_ols(x, y)
    
    ann_ret = math.pow(math.exp(slope), Config.TREND_DAY) - 1
    return ann_ret * r2

# --- 核心逻辑 ---

def get_rank(etf_pool, data_cache):
    score_list = []
    
    if Config.ENABLE_MULTI_PERIOD_SCORE:
        # --- 多周期评分逻辑 (ETF_std_score) ---
        periods = Config.MULTI_PERIODS + [g.m_days]
        weights = Config.MULTI_WEIGHTS 
        
        for etf in etf_pool:
            if etf not in data_cache: 
                score_list.append(-999)
                continue
            
            total_score = 0
            df = data_cache[etf]['close']
            for p, w in zip(periods, weights):
                s = get_single_momentum_score(df, p)
                total_score += s * w
            score_list.append(total_score)
            
    else:
        # --- 单周期基础逻辑 (ETF_wy) ---
        for etf in etf_pool:
            if etf not in data_cache: 
                score_list.append(-999)
                continue
                
            df_close = data_cache[etf]['close']
            
            # 1. 基础得分
            score = get_single_momentum_score(df_close, g.m_days)
            
            # 2. 长周期反转修正 (ETF_long)
            if Config.ENABLE_LONG_TERM_REVERSAL:
                long_days = g.m_days * 8
                if len(df_close) >= long_days:
                    score_long = get_single_momentum_score(df_close, long_days)
                    # 反转修正：减去长周期得分的一部分
                    score = score - score_long / Config.REVERSAL_FACTOR
            
            # 3. 波动率调整 (ETF_vol)
            if Config.ENABLE_VOLATILITY_ADJUST:
                # 简单实现：score = score + sharpe * 0.2 (参考 vol 代码)
                # Vol 代码逻辑： annualized_returns * r_squared (and explicitly no sharpe in some versions, but let's add logic)
                # 我们这里采用一种惩罚高波动的逻辑：
                if len(df_close) >= g.m_days:
                    y = np.log(df_close[-g.m_days:])
                    volatility = np.std(np.diff(y)) * np.sqrt(Config.TREND_DAY)
                    if volatility > 0:
                        # 如果波动率过大，降低得分
                        score = score / (1 + volatility) 

            score_list.append(score)

    df = pd.DataFrame(index=etf_pool, data={'score': score_list})
    df = df.sort_values(by='score', ascending=False)
    
    # 打印 Record
    for etf,ename in etf_pool.items():
       # 限制显示的最大值为 20，防止极端值(如年化几百倍)压缩图表
       disp_score = df.loc[etf]['score']
       if disp_score > 10: disp_score = 10
       elif disp_score < -10: disp_score = -10
       record(**{ename: round(disp_score, 2)})
        
    return df

def get_beta(context, etf, data_cache):
    # 计算动态 Beta (ETF_long)
    if etf not in data_cache: return Config.BETA_DEFAULT
    
    etf_data = data_cache[etf] # 包含 high, low
    # 只需要最近 Config.TREND_DAY 天
    etf_data = etf_data.iloc[-Config.TREND_DAY:]
    
    if len(etf_data) < Config.TREND_DAY: return Config.BETA_DEFAULT 
    
    beta_list = []
    # 滚动计算 20 日斜率
    # 这里也可以优化，但暂时保留循环结构，因为数据已经在内存中
    for i in range(0, len(etf_data)-21, 5): 
        sub_df = etf_data.iloc[i:i+20]
        slope = np.polyfit(sub_df.low, sub_df.high, 1)[0]
        beta_list.append(slope)
        
    if not beta_list: return Config.BETA_DEFAULT
    # 均值 - 2倍标准差
    return np.mean(beta_list) - Config.BETA_STD_M * np.std(beta_list)


# --- RSRS 辅助函数 ---
def get_ols(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    # 修复除零错误
    var_y = np.var(y, ddof=1)
    if var_y == 0:
        r2 = 0
    else:
        r2 = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * var_y))
    return (intercept, slope, r2)

def get_zscore(slope_series):
    mean = np.mean(slope_series)
    std = np.std(slope_series)
    return (slope_series[-1] - mean) / std

def get_rsrs_zscore(stock_data, N, M):
    # data need to be at least N + M
    if len(stock_data) < N + M: return 0, 0
    
    # Calculate historical slopes
    # This can be slow if done every day for 600 points window
    # Optimization: maintain a global queue if performance is critical
    # For now, we compute on the fly using numpy striding or just loop for modularity
    
    # Fast implementation using stride for rolling window OLS is complex
    # We use a simple loop for the last M points
    # Need high/low
    high = stock_data['high'].values
    low = stock_data['low'].values
    
    slopes = []
    # We need M slopes. Each slope needs N days.
    # Total data needed: M + N - 1
    # i ranges from 0 to M-1. 
    # data slice: [i : i+N]
    # To get M slopes ending at current day:
    # We take last M+N-1 days
    
    subset_high = high[-(N+M-1):]
    subset_low = low[-(N+M-1):]
    
    if len(subset_high) < N+M-1: return 0, 0

    for i in range(M):
        h = subset_high[i:i+N]
        l = subset_low[i:i+N]
        slope = np.polyfit(l, h, 1)[0]
        slopes.append(slope)
        
    r2 = np.polyfit(subset_low[-N:], subset_high[-N:], 1)[0] # Just a placeholder/recalc
    # Actually we need r2 of the LAST window
    intercept, slope, r2 = get_ols(subset_low[-N:], subset_high[-N:])
    
    zscore = get_zscore(slopes)
    return zscore, r2


def trade(context):
    # --- 0. 统一数据获取 (Optimization) ---
    # fetch_len 需要满足所有逻辑的最大需求
    # RSRS Z-Score 逻辑需要 RSRS_M + RSRS_N
    rsrs_need = Config.RSRS_M + Config.RSRS_N + 10
    
    max_days = max(Config.TREND_DAY, rsrs_need)
    if Config.ENABLE_LONG_TERM_REVERSAL:
        max_days = max(max_days, g.m_days * 8)
    # 加上一点 Buffer
    fetch_len = max_days + 10
    
    data_cache = {}
    for etf in g.etf_pool:
        # 获取 close, high, low
        data_cache[etf] = attribute_history(etf, fetch_len, '1d', ['close', 'high', 'low'])

    # --- 1. 更新动量 ---
    if Config.ENABLE_TREND_ACCELERATION:
        update_stock_motion(context, g.etf_pool, data_cache)

    # --- 2. 获取排名 ---
    rank_df = get_rank(g.etf_pool, data_cache)
    target_num = Config.STOCK_NUM
    
    # 分歧度过滤 (ETF_long)
    force_empty = False
    if Config.ENABLE_DISPERSION_FILTER and not rank_df.empty:
        scores = rank_df['score'].tolist()
        c = max(scores) - min(scores)
        if not (Config.DISPERSION_MIN < c < Config.DISPERSION_MAX):
            force_empty = True
            log.info(f"分歧度过滤触发: c={c:.4f}, 空仓观望")

    if force_empty:
        target_list = []
    else:
        # 初步选出 Top N
        target_list = list(rank_df.index)[:target_num]
        
        # 3. 趋势加速过滤 (ETF_yj15)
        if Config.ENABLE_TREND_ACCELERATION and target_list:
            top_etf = target_list[0]
            if top_etf in g.stock_motion and len(g.stock_motion[top_etf]) >= 2:
                recent_motion = g.stock_motion[top_etf]
                change_rate = recent_motion[-1] - recent_motion[-2]
                if change_rate > Config.MOTION_1DIFF_THR:
                    target_list = []
                    log.info(f"趋势加速过滤触发: {top_etf} change_rate={change_rate:.2f}")

    # 4. RSRS 择时过滤 (ETF_long)
    if Config.ENABLE_RSRS_TIMING:
        final_target_list = []
        for etf in target_list:
            if etf not in data_cache: continue
            
            # 使用 cache 中的数据 (至少需要 M+N or Trend_Day)
            # 统一取足够长的数据
            df = data_cache[etf]
            
            # --- 逻辑 A: 动态 Beta (ETF_long) ---
            if Config.ENABLE_DYNAMIC_BETA_RSRS:
                hl = df.iloc[-Config.RSRS_N:]
                if len(hl) < Config.RSRS_N: 
                     final_target_list.append(etf)
                     continue
                slope = np.polyfit(hl.low, hl.high, 1)[0]
                threshold = get_beta(context, etf, data_cache)
                
                if slope > threshold:
                    final_target_list.append(etf)
                else:
                    log.info(f"RSRS(Beta)未通过: {etf} slope={slope:.3f} thr={threshold:.3f}")

            # --- 逻辑 B: 标准 RSRS Z-Score (ETF_yj15) ---
            else:
                # 1. RSRS Score Check
                # 需要 Config.RSRS_M + Config.RSRS_N 的数据
                zscore, r2 = get_rsrs_zscore(df, Config.RSRS_N, Config.RSRS_M)
                rsrs_score = zscore * r2
                
                # 2. MA Trend Check
                # Mean Day MA
                if len(df) < Config.MEAN_DAY + 2:
                    current_ma = 0
                    prev_ma = 0
                else:
                    current_ma = df['close'].iloc[-Config.MEAN_DAY:].mean()
                    prev_ma = df['close'].iloc[-(Config.MEAN_DAY+1):-1].mean()
                
                # Condition
                if rsrs_score > Config.RSRS_SCORE_THR and current_ma > prev_ma:
                    final_target_list.append(etf)
                else:
                    log.info(f"RSRS(Z)未通过: {etf} score={rsrs_score:.3f} MA_Trend={current_ma>prev_ma}")

        target_list = final_target_list

    # --- 执行交易 ---
    # 获取当前真实持仓 (amount > 0)
    current_holdings = [etf for etf in context.portfolio.positions if context.portfolio.positions[etf].total_amount > 0]
    
    # 如果目标与当前持仓完全一致 (无变化)，直接跳过
    if set(target_list) == set(current_holdings):
        log.info("持仓与目标一致，无操作")
        return

    # 1. 卖出不在目标列表中的
    for etf in current_holdings:
        if etf not in target_list:
            order_target_value(etf, 0)
            log.info(f"卖出 {etf}")
            
    # 2. 买入逻辑
    # 统计卖出后预计剩余的持仓数量
    held_etfs_count = 0
    for etf in current_holdings:
        if etf in target_list:
            held_etfs_count += 1
            
    # 需要新买入的标的
    to_buy_list = [etf for etf in target_list if etf not in current_holdings]
    
    if len(to_buy_list) > 0:
        # 资金分配：可用资金 / 需要买入的数量
        # 注意：这里假设卖出释放的资金已回流到 available_cash (ETF通常T+0可用)
        if len(to_buy_list) > 0:
            value = context.portfolio.available_cash / len(to_buy_list)
            for etf in to_buy_list:
                order_target_value(etf, value)
                log.info(f"买入 {etf}")
