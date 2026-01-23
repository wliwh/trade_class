# 导入必要的库
from jqdata import *
import pandas as pd
import numpy as np
import datetime

# 初始化函数，设置基本参数
def initialize(context):
    # 开启真实价格回测模式，避免未来函数
    set_option('use_real_price', True)
    
    # 设置回测的开始和结束时间
    set_param(context)
    
    # 设置基准指数
    set_benchmark('000300.XSHG')  # 沪深300指数
    
    # 设置滑点和手续费
    set_slippage(PriceRelatedSlippage(0.002))  # 设置滑点为0.2%
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0001, open_commission=0.0003, close_commission=0.0003, close_today_commission=0, min_commission=0.5), type='stock')
    
    # 设置要交易的ETF池
    g.etfs = [
        '510050.XSHG',  # 上证50ETF
        '510300.XSHG',  # 沪深300ETF
        '510500.XSHG',  # 中证500ETF
        '159915.XSHE',  # 创业板ETF
        '159919.XSHE'   # 沪深300ETF(深)
    ]
    
    # 设置均线参数
    g.ma_period = 20   # 均线周期
    g.lookback_days = 30  # 回看天数，用于计算均线历史值
    
    # 当前持仓的ETF
    g.current_holdings = []
    
    # 记录上一日净值用于计算收益率
    g.previous_portfolio_value = context.portfolio.total_value
    
    # 设置定时运行函数
    run_daily(trade, time='9:30')

# 设置策略参数
def set_param(context):
    # 设置回测的开始日期和结束日期
    g.start_date = '2020-01-01'  # 回测开始时间
    g.end_date = '2020-12-31'    # 回测结束时间
    log.info('策略开始日期: %s, 结束日期: %s' % (g.start_date, g.end_date))

# 计算均线能量
def calculate_ma_energy(security, ma_period, lookback_days):
    # 获取历史数据，多获取一些数据以计算均线
    prices = attribute_history(security, lookback_days + ma_period, '1d', ['close'])
    
    # 计算每日的均线值
    ma_values = []
    for i in range(lookback_days):
        ma_value = prices['close'][i:i+ma_period].mean()
        ma_values.append(ma_value)
    
    # 计算均线能量（连续N日均线值大于或小于前一日的值）
    energy = 0
    for i in range(1, len(ma_values)):
        if ma_values[i] > ma_values[i-1]:  # 当日均线值大于前一日
            if energy > 0:  # 如果已经是正向趋势，则累加
                energy += 1
            else:  # 如果是负向趋势，则重置为1
                energy = 1
        elif ma_values[i] < ma_values[i-1]:  # 当日均线值小于前一日
            if energy < 0:  # 如果已经是负向趋势，则累减
                energy -= 1
            else:  # 如果是正向趋势，则重置为-1
                energy = -1
        # 如果相等，能量保持不变
    
    return energy

# 交易函数
def trade(context):
    # 计算所有ETF的均线能量
    etf_energies = []
    for etf in g.etfs:
        energy = calculate_ma_energy(etf, g.ma_period, g.lookback_days)
        etf_energies.append((etf, energy))
        log.info("ETF: %s, 均线能量: %d" % (etf, energy))
    
    # 按均线能量降序排序
    etf_energies.sort(key=lambda x: x[1], reverse=True)
    
    # 筛选均线能量大于0的ETF
    positive_energy_etfs = [item for item in etf_energies if item[1] > 0]
    
    # 清空当前所有持仓
    for etf in g.current_holdings:
        order_target(etf, 0)
        log.info("清空持仓: %s" % etf)
    
    g.current_holdings = []
    
    # 根据均线能量情况决定交易策略
    if len(positive_energy_etfs) >= 2:
        # 至少有2个ETF的均线能量为正，选择能量最大的两个均仓买入
        target_etfs = positive_energy_etfs[:2]
        position_ratio = 0.5  # 每个ETF分配50%的资金
        
        for etf, energy in target_etfs:
            order_value(etf, context.portfolio.cash * position_ratio)
            g.current_holdings.append(etf)
            log.info("买入ETF: %s, 均线能量: %d, 仓位: %.0f%%" % (etf, energy, position_ratio * 100))
    
    elif len(positive_energy_etfs) == 1:
        # 只有1个ETF的均线能量为正，以50%仓位买入
        etf, energy = positive_energy_etfs[0]
        position_ratio = 0.5
        order_value(etf, context.portfolio.cash * position_ratio)
        g.current_holdings.append(etf)
        log.info("买入ETF: %s, 均线能量: %d, 仓位: %.0f%%" % (etf, energy, position_ratio * 100))
    
    else:
        # 所有ETF的均线能量都为负，空仓
        log.info("所有ETF均线能量均为负，保持空仓")

# 收盘后处理函数
def after_trading_end(context):
    # 记录每日收益
    current_value = context.portfolio.total_value
    
    # 计算日收益率
    if hasattr(g, 'previous_portfolio_value') and g.previous_portfolio_value > 0:
        daily_returns = (current_value - g.previous_portfolio_value) / g.previous_portfolio_value
    else:
        daily_returns = 0
    
    # 计算总收益率
    starting_value = context.portfolio.starting_cash
    total_returns = (current_value - starting_value) / starting_value
    
    # 更新上一日净值
    g.previous_portfolio_value = current_value
    
    # 记录信息
    log.info("今日收益率: %.2f%%" % (daily_returns * 100))
    log.info("总收益率: %.2f%%" % (total_returns * 100))
    log.info("当前持仓: %s" % g.current_holdings)
