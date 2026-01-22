# 克隆自聚宽文章：https://www.joinquant.com/post/65000
# 标题：核心ETF和行业ETF轮动策略
# 作者：Alexsaesh

##
# 标题：核心ETF-精练版
# 参照：
# （1）克隆自聚宽文章：https://www.joinquant.com/post/42673
# （2）标题：【回顾3】ETF策略之核心资产轮动
# （3）作者：wywy1995
# 修改内容：
#       1. 为了在模拟环境修改代码，增加了do_schedule函数
#       2. 分开卖出和买入的时间，并调试到合适的时间
#       3. 增加收盘前14:55货币基金买入的操作
#       4. 合理优化了ETF池
#       5. 为减少过度拟合，删除以前增加的诸多设计
# 状态：在实盘中
# 作者：王巨明
##
import numpy as np
import pandas as pd
from jqdata import *
##
##初始化函数 
def initialize(context):
    # 设定基准，选用沪深300指数
    set_benchmark('513100.XSHG')    
    # 开启避免未来数据的模式
    set_option("avoid_future_data", True)
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 设置滑点 https://www.joinquant.com/view/community/detail/a31a822d1cfa7e83b1dda228d4562a70
    set_slippage(FixedSlippage(0.001))
    # 调试时必须打开下面语句；调试完成后，开启下面语句，减少日志内容
    log.set_level('order', 'error')
    # 设置成交比例，即每分钟内的成交量的10%(原来为5%)
    set_option('order_volume_ratio', 0.10)
    # 设置交易费用
    set_order_cost(OrderCost(
        close_tax=0, open_commission=0.00005, 
        close_commission=0.00005, min_commission=0), 
        type='fund')
    # 设置参数
    set_parameters(context)
    do_schedule(context)
##
def set_parameters(context):
    g.etf_pool = [
        '159949.XSHE', #创业板50，  成立2016-06-30，基金规模286.7亿元 (截止2025-9-30)
        '513100.XSHG', #纳指ETF，　 成立2013-04-25，基金规模171.6亿元 (截止2025-9-30)
        #'513090.XSHG', #香港证券，  成立2020-03-13，基金规模341.8亿元 (截止2025-9-30)
        '518880.XSHG', #黄金ETF， 　成立2013-07-18，基金规模686.2亿元 (截止2025-9-30)
        ]
    g.MF_etf='511880.XSHG'  #货币基金｜银华日利ETF成立2013-04-01，基金规模688.4亿元 (截止2025-09-30)
    g.target_etf = ""   #选择符合条件用于交易的ETF（唯一）
##
def do_schedule(context): 
    run_daily(trade_prep, '5:00')           
    run_daily(trade_sell, '10:30') 
    run_daily(trade_buy,  '10:30')
    # 收盘前，货币基金的交易
    run_daily(trade_MFetf,  '14:55')  
##
def after_code_changed(context):
    #1、设置参数
    set_parameters(context)
    #2、取消和重启所有定时运行
    unschedule_all()
    do_schedule(context)
##
## 开盘前准备函数
def trade_prep(context):
    #【1】保留成立超过400天的ETF【至少250个交易日】
    etf_list=[]
    for etf in g.etf_pool:
        start_date = get_security_info(etf).start_date
        one_years_ago = context.current_dt.date() - pd.Timedelta(days=400)
        if start_date < one_years_ago:
            etf_list.append(etf) 
    #【2】计算score，存放于g.etf_df中
    etf_df=get_rank(etf_list)
    #【3】计算real_target_list：将g.etf_df拷贝到etf_df，并得到real_target_list
     #生成经排序的real_target_list，得到最终的g.target_etf
    etf_df.sort_values(by="score", ascending=False, inplace=True)
    ## 保留score>0的ETF
    etf_df=etf_df[etf_df["score"]>0]
    real_target_list = etf_df.index.tolist()
    if len(real_target_list)>0:
        g.target_etf =real_target_list[0]
    else:
        g.target_etf =g.MF_etf
    #【4】显示且微信提示所选择的etf、持有的etf
    target_etf=g.target_etf
    target_name=get_security_info(target_etf).display_name
    hold_list = list(context.portfolio.positions)
    if len(hold_list)>0:
        hold_etf=hold_list[0]
        hold_name=get_security_info(hold_etf).display_name
        print("=== 已持etf：%s【%s】,  现选etf：%s【%s】" % (hold_etf[:6],hold_name,target_etf[:6],target_name))
        send_message("=== 已持etf：%s【%s】, 现选etf：%s【%s】" % (hold_etf[:6],hold_name,target_etf[:6],target_name))
    else:
        print("=== 现选etf：%s【%s】" % (target_etf[:6],target_name))
        send_message("=== 现选etf：%s【%s】" % (target_etf[:6],target_name))
##
## 卖出交易
def trade_sell(context):
    hold_list = list(context.portfolio.positions)
    if len(hold_list)>0:
        hold_etf=hold_list[0]
        name=get_security_info(hold_etf).display_name
        if hold_etf!=g.target_etf and hold_etf!=g.MF_etf:
            if context.portfolio.positions[hold_etf].closeable_amount>0: 
                order_sell=order_target_value(hold_etf, 0)
                if order_sell != None and order_sell.filled > 0:
                    log.info("已成功卖出%s【%s】%d股。" % (name,hold_etf,order_sell.filled))
##
## 买入交易
def trade_buy(context):
    current_data = get_current_data()
    ## 先进行卖出g.MF_etf操作
    hold_list = list(context.portfolio.positions)
    if len(hold_list) > 0 :
        hold_etf=hold_list[0]
        if hold_etf!=g.target_etf:
            if hold_etf==g.MF_etf:
                etf=hold_etf
                name=get_security_info(etf).display_name
                order_sell=order_target_value(etf, 0)
                if order_sell != None and order_sell.filled > 0:
                    log.info("已成功卖出%s【%s】%d股。" % (name,etf,order_sell.filled))
    ## 真正的买入操作
    value = context.portfolio.available_cash
    # 金额不小于10000元时才进行买入操作
    if value>=10000:
        etf=g.target_etf
        name=get_security_info(etf).display_name
        last_price=current_data[etf].last_price   
        high_limit=current_data[etf].high_limit
        # 买入时，在基础价加0.5%。对于黄金ETF，限价过大而影响资金使用。
        tmp_price=min(max(last_price*1.005,last_price+0.001),last_price+0.005)
        limit_price=min(tmp_price,high_limit)                   
        order_buy=order_target_value(etf,value,LimitOrderStyle(limit_price))
        if order_buy != None and order_buy.filled > 0:
            log.info("已成功买入%s【%s】%s股。" % (name,etf,order_buy.filled))
##
## 收盘前，货币基金的交易
def trade_MFetf(context):
    current_data = get_current_data()
    value = context.portfolio.available_cash
    # 金额不小于10000元时才进行买入操作
    if value>=10000:
        etf=g.MF_etf
        name=get_security_info(etf).display_name
        set_order_cost(OrderCost(close_tax=0, open_commission=0, 
            close_commission=0, min_commission=0), type='fund')
        last_price=current_data[etf].last_price   
        # 买入时，在基础价加0.1%。
        limit_price=last_price*1.001
        order_buy=order_target_value(etf,value,LimitOrderStyle(limit_price))
        if order_buy != None and order_buy.filled > 0:
            log.info("已成功买入%s【%s】%s股。" % (name,etf,order_buy.filled))
##
## 基于年化收益和判定系数打分的动量因子轮动 https://www.joinquant.com/post/26142
def get_rank(etf_pool):
    score_list = []
    slope_list = []
    for etf in etf_pool:
        m_days = 25 #动量参考天数
        df = attribute_history(etf, m_days, '1d', ['close'])
        y = df['log'] = np.log(df.close)
        x = df['num'] = np.arange(df.log.size)
        slope, intercept = np.polyfit(x, y, 1)
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        r_squared = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
        score = annualized_returns * r_squared
        score_list.append(score)
        slope_list.append(slope)
    df = pd.DataFrame(index=etf_pool, data={'slope':slope_list,'score':score_list})
    # 给索引命名为'ETF'
    df.index.name = 'ETF'  
    return df
# end
