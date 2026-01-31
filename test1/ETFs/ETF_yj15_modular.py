# 克隆自聚宽文章：https://www.joinquant.com/post/41718
# 标题：多因子宽基ETF择时轮动改进版-高收益大资金低回撤
# 作者：养家大哥
# 重构说明：逻辑不变，将程序重构为“参数配置”、“前期过滤”，“排名计算”、“后期过滤”、”盘中止损“、”买卖执行“等模块

from jqdata import *
import numpy as np
import math
from jqlib.technical_analysis import *

EXECUTION_SOLD_TIME_PLACEHOLDER = '9:30'
EXECUTION_BUY_TIME_PLACEHOLDER = '9:35'
EXECUTION_ETF_POOLS_PLACEHOLDER = ['510300.XSHG','510050.XSHG','159949.XSHE','159928.XSHE']

# ==============================================================================
# 1. 参数配置模块 (Parameter Configuration)
# ==============================================================================
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

def initialize(context):
    set_benchmark(Config.BENCHMARK)
    set_option('use_real_price', Config.USE_REAL_PRICE)
    set_option("avoid_future_data", Config.AVOID_FUTURE_DATA)  # 避免引入未来信息
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
    log.set_level('order', 'error')
    
    # 策略参数
    g.stock_num = 1         # 买入评分最高的前stock_num只股票
    g.momentum_day = 20     # 最新动量参考最近momentum_day的
    g.ref_stock = '000300.XSHG' # 用ref_stock做择时计算的基础数据
    g.N = 18                # 计算最新斜率slope，拟合度r2参考最近N天
    g.M = 600               # 计算最新标准分zscore，rsrs_score参考最近M天(600)
    g.K = 8                 # 计算 zscore 斜率的窗口大小
    g.biasN = 90            # 乖离动量的时间天数
    g.lossN = 20            # 止损MA20---60分钟
    g.lossFactor = 1.005    # 下跌止损的比例，相对前一天的收盘价
    g.SwitchFactor = 1.04   # 换仓位的比例，待换股相对当前持股的分数
    g.Motion_1diff = 19     # 股票前一天动量变化速度门限
    g.raiser_thr = 4.8      # 股票前一天上涨的比例门限
    g.hold_stock = 'null'
    g.score_thr = -0.68          # rsrs标准分指标阈值
    g.score_fall_thr = -0.43     # 当股票下跌趋势时候， 卖出阀值rsrs
    g.idex_slope_raise_thr = 12  # 判断大盘指数强势的斜率门限
    
    # 初始化筛选池
    g.stock_pool = EXECUTION_ETF_POOLS_PLACEHOLDER
    # 沪深300ETF, 上证50ETF, 创业板500, 中证消费ETF    
    
    # 数据预处理（前期准备）
    g.slope_series, g.rsrs_score_history = initial_slope_series() # 除去回测第一天的slope
    g.stock_motion = initial_stock_motion(g.stock_pool)           # 除去回测第一天的动量

    # 定时运行
    run_daily(my_trade_prepare, time='7:00', reference_security=g.ref_stock) # 准备信号
    run_daily(check_lose, time='open', reference_security=g.ref_stock)       # 盘中检查(原版保留)
    run_daily(my_trade, time=EXECUTION_SOLD_TIME_PLACEHOLDER, reference_security=g.ref_stock)         # 交易执行(卖出)
    run_daily(my_sell2buy, time=EXECUTION_BUY_TIME_PLACEHOLDER, reference_security=g.ref_stock)      # 交易执行(买入)
    
    # 盘中风控
    run_daily(pre_hold_check, time='11:25')
    run_daily(hold_check, time='11:27')

# ==============================================================================
# 2. 前期过滤/数据准备模块 (Data Preparation & Filter)
# ==============================================================================
# 初始化准备数据,除去回测第一天的slope,zscores
def initial_slope_series():
    length = g.N + g.M + g.K
    data = attribute_history(g.ref_stock, length, '1d', ['high', 'low', 'close'])
    multe_data = [get_ols(data.low[i:i+g.N], data.high[i:i+g.N]) for i in range(length-g.N)]
    slopes = [i[1] for i in multe_data]
    r2s = [i[2] for i in multe_data]
    zscores = [(get_zscore(slopes[i+1:i+1+g.M]) * r2s[i+g.M]) for i in range(g.K)]
    return (slopes, zscores)

# 获取初始化动量因子，除去回测第一天
def initial_stock_motion(stock_pool):
    stock_motion = {}
    for stock in stock_pool:
        motion_que = []
        data = attribute_history(stock, g.biasN + g.momentum_day + 1, '1d', ['close'])
        data = data[:-1]
        bias = (data.close/data.close.rolling(g.biasN).mean())[-g.momentum_day:] 
        score = np.polyfit(np.arange(g.momentum_day), bias/bias[0], 1)[0].real*10000 
        motion_que.append(score)
        stock_motion[stock] = motion_que
    return stock_motion

# 线性回归工具函数
def get_ols(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
    return (intercept, slope, r2)

# 因子标准化工具函数
def get_zscore(slope_series):
    mean = np.mean(slope_series)
    std = np.std(slope_series)
    return (slope_series[-1] - mean) / std

def get_zscore_slope(z_scores):
    y = z_scores
    x = np.arange(len(z_scores))
    slope, intercept = np.polyfit(x, y, 1)
    return slope

# ==============================================================================
# 3. 排名计算模块 (Ranking Calculation)
# ==============================================================================
# 动量因子：由收益率动量改为相对MA90均线的乖离动量
def get_rank(context, stock_pool):
    rank = []
    for stock in stock_pool:
        data = attribute_history(stock, g.biasN + g.momentum_day, '1d', ['close'])
        bias = (data.close/data.close.rolling(g.biasN).mean())[-g.momentum_day:] # 乖离因子
        score = np.polyfit(np.arange(g.momentum_day), bias/bias[0], 1)[0].real*10000 # 乖离动量拟合
        adr = 100*(data.close[-1] - data.close[-2])/data.close[-2] # 股票的涨跌幅度
        
        # 换仓因子
        if stock == g.hold_stock: 
            raise_x = g.SwitchFactor
        else: 
            raise_x = 1
            
        rank.append([stock, score*raise_x, adr])
        
        # 更新动量记录
        g.stock_motion[stock].append(score)
        if len(g.stock_motion[stock]) > 5:
            g.stock_motion[stock].pop(0)
            
    # 日志输出
    log_str = ''
    for item in rank:
        log_str += "%s:%.2f:%.2f; "%(item[0], item[1], item[2])
    log.info(log_str)
    
    # 排序
    rank = [i for i in rank if math.isnan(i[1]) == False]
    rank.sort(key=lambda x: x[1], reverse=True)
    return rank[0]

# ==============================================================================
# 4. 后期过滤/择时模块 (Post-filtering & Timing)
# ==============================================================================
# 计算RSRS择时信号
def get_timing_signal(context, stock):
    data = attribute_history(g.ref_stock, g.N, '1d', ['high', 'low', 'close'])
    intercept, slope, r2 = get_ols(data.low, data.high)
    g.slope_series.append(slope)
    rsrs_score = get_zscore(g.slope_series[-g.M:]) * r2
    g.rsrs_score_history.append(rsrs_score)
    rsrs_slope = get_zscore_slope(g.rsrs_score_history[-g.K:])
    
    # 大盘指数收盘价趋势
    idex_slope = np.polyfit(np.arange(8), data.close[-8:], 1)[0].real
    
    g.slope_series.pop(0)
    g.rsrs_score_history.pop(0)
    
    log.info('rsrs_slope {:.3f} rsrs_score {:.3f} idex_slope {:.3f}'.format(rsrs_slope, rsrs_score, idex_slope))
    
    # WR指标辅助判断
    WR2, WR1 = WR([g.ref_stock], check_date=context.previous_date, N=21, N1=14, unit='1d', include_now=True)
    if WR1[g.ref_stock] >= 97 and WR2[g.ref_stock] >= 97: return "BUY"
    
    # RSRS信号逻辑
    if (rsrs_slope < 0 and rsrs_score > 0):
        return "SELL"
    if (idex_slope < 0) and (rsrs_slope > 0) and (rsrs_score < g.score_fall_thr): 
        return "SELL"
    if (idex_slope > g.idex_slope_raise_thr) and (rsrs_slope > 0): 
        return "BUY"
    if (rsrs_score > g.score_thr): 
        return "BUY"
    
    return "SELL"

# 综合信号准备及后期个股过滤
def my_trade_prepare(context):
    # 1. 获取排名第一的股票
    g.check_out_list = get_rank(context, g.stock_pool)
    
    # 2. 获取大盘择时信号
    g.timing_signal = get_timing_signal(context, g.ref_stock)
    log.info('今日自选及择时信号:{} {}'.format(g.check_out_list[0], g.timing_signal))
    
    # 3. 后期过滤：判断个股动量变化一阶导数和涨幅
    cur_stock = g.check_out_list[0]
    cur_adr = g.check_out_list[2] # 股票价格相对前一天涨跌比例
    change_rate = g.stock_motion[cur_stock][-1] - g.stock_motion[cur_stock][-2]
    
    if (change_rate > g.Motion_1diff) or (cur_adr > g.raiser_thr):
        g.timing_signal = 'SELL'
        log.info("由于涨跌:%.2f, 动量变化%.2f，今日空仓" % (cur_adr, change_rate))
    
    # 4. 生成交易意向消息
    if g.timing_signal == 'SELL':
        for stock in context.portfolio.positions:
            send_message("准备卖出ETF [%s]" % stock)
    elif g.timing_signal == 'BUY' or g.timing_signal == 'KEEP':
        if g.check_out_list[0] not in context.portfolio.positions:
            if len(context.portfolio.positions) > 0:
                stock_tmps = list(context.portfolio.positions.keys())
                send_message("准备卖ETF [%s], 买入ETF [%s]" % (stock_tmps[0], g.check_out_list[0]))
            else:
                send_message("准备买入ETF [%s]" % g.check_out_list[0])
    else:
        send_message("保持原来仓位")

# ==============================================================================
# 5. 盘中止损模块 (Intraday Stop-Loss)
# ==============================================================================
# 盘中动态止损前置检查
def pre_hold_check(context):
    if context.portfolio.positions:
        for stk in context.portfolio.positions:
            dt = attribute_history(stk, g.lossN+2, '60m', ['close'])
            dt['man'] = dt.close / dt.close.rolling(g.lossN).mean()
            if dt.man[-1] < 1.0:
                log.info("盘中可能止损，卖出：{}".format(stk))
                send_message("盘中可能止损，卖出：{}".format(stk))

# 盘中正式止损检查
def hold_check(context):
    current_data = get_current_data()
    if context.portfolio.positions:
        for stk in context.portfolio.positions:
            yesterday_di = attribute_history(stk, 1, '1d', ['close'])
            dt = attribute_history(stk, g.lossN+2, '60m', ['close'])
            dt['man'] = dt.close / dt.close.rolling(g.lossN).mean()
            
            # 止损条件：60分钟跌破MA20 且 当前价格低于昨日收盘价一定比例(或持平)
            if (dt.man[-1] < 1.0) and (current_data[stk].last_price * g.lossFactor <= yesterday_di['close'][-1]):
                stk_dict = context.portfolio.positions[stk]
                log.info('准备平仓，总仓位:{}, 可卖出：{}, '.format(stk_dict.total_amount, stk_dict.closeable_amount))
                send_message("盘中止损，卖出：{}".format(stk))
                if stk_dict.closeable_amount:
                    order_target_value(stk, 0)
                    log.info('盘中止损', stk)
                else:
                    log.info('无法止损', stk)

# 原版保留的简单止损，似乎未被重度使用，但保留以防万一
def check_lose(context):
    for position in list(context.portfolio.positions.values()):
        security = position.security
        cost = position.avg_cost
        price = position.price
        ret = 100 * (price / cost - 1)
        if ret <= -90:
            order_target_value(position.security, 0)
            print("！！！！！！触发止损信号: 标的={}, 浮动盈亏={}% ！！！！！！".format(security, format(ret, '.2f')))

# ==============================================================================
# 6. 买卖执行模块 (Execution)
# ==============================================================================
# 9:30 执行卖出
def my_trade(context):
    if g.timing_signal == 'SELL':
        for stock in context.portfolio.positions:
            position = context.portfolio.positions[stock]
            close_position(position)
    elif g.timing_signal == 'BUY' or g.timing_signal == 'KEEP':
        adjust_position(context, g.check_out_list)

# 9:35 执行买入
def my_sell2buy(context):
    hour = context.current_dt.hour
    if hour == 9:
        if g.timing_signal == 'BUY' or g.timing_signal == 'KEEP':
            buy_stocks(context, g.check_out_list)

# 调仓逻辑：卖出非目标持仓
def adjust_position(context, buy_stocks):
    for stock in context.portfolio.positions:
        if stock not in buy_stocks:
            position = context.portfolio.positions[stock]
            close_position(position)
            g.hold_stock = 'null'
            return
    
    # 如果已经在 my_trade 中卖出了，或者本来就没持仓，这里只负责卖出逻辑
    # 买入在 my_sell2buy 或者 下面的逻辑中处理 (原版逻辑如此)
    # 原版 adjust_position 同时也包含了买入逻辑，这里保留原版逻辑
    buy_exec(context, buy_stocks)

# 买入执行
def buy_stocks(context, buy_stocks):
    buy_exec(context, buy_stocks)

# 统一的买入执行函数
def buy_exec(context, buy_stocks):
    position_count = len(context.portfolio.positions)
    if g.stock_num > position_count:
        value = context.portfolio.cash / (g.stock_num - position_count)
        for stock in buy_stocks:
            if context.portfolio.positions[stock].total_amount == 0:
                if open_position(stock, value):
                    if len(context.portfolio.positions) == g.stock_num:
                        g.hold_stock = stock
                        break

# 开仓工具函数
def open_position(security, value):
    order = order_target_value(security, value)
    if order != None and order.filled > 0:
        return True
    return False

# 平仓工具函数
def close_position(position):
    security = position.security
    order = order_target_value(security, 0)
    if order != None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False