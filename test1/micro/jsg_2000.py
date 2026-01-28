# 克隆自聚宽文章：https://www.joinquant.com/post/56890
# 标题：中证2000搅屎棍指数增强策略，年化40%回撤15%
# 作者：MarioC

# 四大搅屎棍策略
# 众所周知，A股就是一坨SI，做量化策略就是屎上雕花，那么屎坑里面就会有搅屎棍，这个雕花过程尽量要避开搅屎棍。
# 这就是我的《四大搅屎棍策略》，避开搅屎棍，妈妈再也不担心我回撤了！
# 经过我一系列的数据分析，我发现了这四大搅屎棍。
# 《银行、有色金属、钢铁、煤炭》。
# 这四个如果任意一个出现在市场宽度TOP1，行情准崩。

from jqdata import *
import numpy as np
import pandas as pd


#初始化函数 
def initialize(context):
    # 设定基准
    set_benchmark('563300.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 设置交易成本万分之三，不同滑点影响可在归因分析中查看
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003, close_today_commission=0, min_commission=5),type='stock')
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    #初始化全局变量
    g.no_trading_today_signal = False
    g.num = 1  # 判断买卖点的行业数量
    g.hold_list = [] #当前持仓的全部股票    
    g.limit_up_list = [] #记录持仓中昨日涨停的股票
    g.pass_months = [1,4,12]
    g.etf = '563300.XSHG'  # 要买入的ETF
    
    # 设置交易运行时间
    run_daily(prepare_stock_list, '9:05') #准备股票池
    run_monthly(adjust, 1, '09:35')
    run_daily(print_position_info, '15:10')

#1-1 准备股票池
def prepare_stock_list(context):
    #获取已持有列表
    g.hold_list = []
    for position in list(context.portfolio.positions.values()):
        stock = position.security
        g.hold_list.append(stock)

#1-3 整体调整持仓
# 调仓
def adjust(context):
    should_hold = select(context)
    if should_hold:
        if g.etf not in context.portfolio.positions:
            order_value(g.etf, context.portfolio.available_cash)
    else:
        if g.etf in context.portfolio.positions:
            order_target_value(g.etf, 0)

# 判断今天是否在空仓月
def is_empty_month(context):
    month = context.current_dt.month
    return month in g.pass_months

# 获取市场宽度
def get_market_breadth(context):
    yesterday = context.previous_date
    stocks = get_index_stocks("000300.XSHG")  # 获取中证全指成分股
    count = 1
    h = get_price(
        stocks,
        end_date=yesterday,
        frequency="1d",
        fields=["close"],
        count=count + 20,
        panel=False,
    )
    h["date"] = pd.DatetimeIndex(h.time).date
    df_close = h.pivot(index="code", columns="date", values="close").dropna(axis=0)
    df_ma20 = df_close.rolling(window=20, axis=1).mean().iloc[:, -count:]
    df_bias = df_close.iloc[:, -count:] > df_ma20
    df_bias["industry_name"] = pd.Series(getStockIndustry(stocks))
    df_ratio = (
        (df_bias.groupby("industry_name").sum() * 100.0)
        / df_bias.groupby("industry_name").count()
    ).round()
    top_industries = df_ratio.loc[:, yesterday].nlargest(g.num)
    I = top_industries.index.tolist()
    
    return I

# 获取股票所属行业
def getStockIndustry(stocks):
    industry = get_industry(stocks)
    return {
        stock: info["sw_l1"]["industry_name"]
        for stock, info in industry.items()
        if "sw_l1" in info
    }

# 择时
def select(context):
    I = get_market_breadth(context) #获取了市场宽度
    industries = {"银行I", "有色金属I", "煤炭I", "钢铁I", "采掘I"}
    
    if industries.intersection(I):
        log.info(f"触发空仓条件，命中特定行业：{industries.intersection(I)}")
        return False
    elif is_empty_month(context):
        log.info(f"触发空仓条件，当前为空仓月份：{context.current_dt.month}月")
        return False
    else:
        log.info("未触发空仓条件，正常持仓")
        return True

#4-3 打印每日持仓信息
def print_position_info(context):
    #打印当天成交记录
    trades = get_trades()
    for _trade in trades.values():
        print('成交记录：'+str(_trade))
    #打印账户信息
    for position in list(context.portfolio.positions.values()):
        securities=position.security
        cost=position.avg_cost
        price=position.price
        ret=100*(price/cost-1)
        value=position.value
        amount=position.total_amount    
        print('代码:{}'.format(securities))
        print('成本价:{}'.format(format(cost,'.2f')))
        print('现价:{}'.format(price))
        print('收益率:{}%'.format(format(ret,'.2f')))
        print('持仓(股):{}'.format(amount))
        print('市值:{}'.format(format(value,'.2f')))