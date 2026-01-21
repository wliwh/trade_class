# 克隆自聚宽文章：https://www.joinquant.com/post/42673
# 标题：【回顾3】ETF策略之核心资产轮动
# 作者：wywy1995

import numpy as np
import pandas as pd


#初始化函数 
def initialize(context):
    # 设定基准
    set_benchmark('000300.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 设置滑点 https://www.joinquant.com/view/community/detail/a31a822d1cfa7e83b1dda228d4562a70
    set_slippage(FixedSlippage(0.003))
    # 设置交易成本
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0001, open_commission=0.0002, close_commission=0.0002, close_today_commission=0, min_commission=5), type='fund')
    # 过滤一定级别的日志
    log.set_level('system', 'error')
    # 参数
    g.etf_pool = {
        '518880.XSHG': '黄金', #黄金ETF（大宗商品）
        '513100.XSHG': '纳指', #纳指100（海外资产）
        '159915.XSHE': '成长', #创业板100（成长股，科技股，中小盘）
        '510180.XSHG': '价值', #上证180（价值股，蓝筹股，中大盘）
        '159628.XSHE': '小盘',
        # '511010.XSHG': '十债',
        '513180.XSHG':'恒生',	# 2012-10-22 || H股ETF,中概互联,恒生通,恒指ETF,港股100,H股ETF

    }
    # g.etf_pool = {
    #     '518880.XSHG':'黄金',	# 2013-07-29 || 黄金基金,黄金ETF,黄金ETF基金
    #     '159907.XSHE':'小盘',	# 2006-09-05 || TMTETF,信息ETF,通信ETF,计算机,科技100,科技ETF,科技50,5GETF,AI智能,中证科技,信息技术ETF
    #     '510050.XSHG':'大盘',	# 2005-02-23 || 工银上50,MSCI基金,上证50,上50ETF,A50ETF,MSCIA股,景顺MSCI,MSCI中国,天弘300,长三角,添富300,100ETF,ZZ800ETF,800ETF,A50基金,MSCI易基,沪50ETF,工银300,华夏300,HS300ETF,300ETF,综指ETF,180ETF,SZ50ETF,平安300,深100ETF银华,沪深300ETF南方,沪深300ETF,深红利ETF,深证100ETF,广发300
    #     '511010.XSHG':'国债',	# 2013-03-25 || 5年地债ETF,招商快线ETF,豆粕ETF,货币ETF,5年地债,城投ETF,十年国债,10年地债
    #     '159920.XSHE':'恒生',	# 2012-10-22 || H股ETF,中概互联,恒生通,恒指ETF,港股100,H股ETF
    #     '513100.XSHG':'纳指',	# 2013-05-15 || 纳指ETF
    #     '510880.XSHG':'红利',	# 2007-01-18 || 能源ETF基金,中证红利,100红利,能源ETF,有色ETF
    # }
    # g.etf_pool = {
    #     '512880.XSHG': '证券',
    #     '515030.XSHG': '新能',
    #     '513050.XSHG': '中概',
    #     '512690.XSHG': '白酒',
    #     '512170.XSHG': '医疗',
    #     '159905.XSHE': '深红利',
    # }
    # g.etf_pool = ['512660.XSHG', '511010.XSHG', '510880.XSHG', '159915.XSHE', '513050.XSHG', '510050.XSHG', 
    #               '588100.XSHG', '512100.XSHG', '518800.XSHG', '513060.XSHG', '512980.XSHG', '512010.XSHG',
    #               '513100.XSHG', '512720.XSHG', '512070.XSHG', '515880.XSHG', '159920.XSHE', '159922.XSHE',
    #               '513520.XSHG', '515000.XSHG', '515790.XSHG', '515700.XSHG', '159825.XSHE', '512400.XSHG',
    #               '512200.XSHG', '513360.XSHG', '512480.XSHG', '510230.XSHG', '159647.XSHE', '159928.XSHE']

    g.m_days = 25 #动量参考天数
    g.ma_energy_days = 20
    run_daily(trade, '9:30') #每天运行确保即时捕捉动量变化


# 基于年化收益和判定系数打分的动量因子轮动 https://www.joinquant.com/post/26142
def get_rank(etf_pool):
    score_list = []
    for etf in etf_pool:
        df = attribute_history(etf, g.m_days, '1d', ['close'])
        y = df['log'] = np.log(df.close)
        x = df['num'] = np.arange(df.log.size)
        slope, intercept = np.polyfit(x, y, 1)
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        r_squared = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
        # score = annualized_returns * r_squared
        # 计算波动率（标准差）
        volatility = np.std(np.diff(y)) * np.sqrt(250)
        
        # 计算夏普比率（简化版，无无风险利率）
        sharpe = annualized_returns / volatility if volatility > 0 else 0
        
        # 优化评分公式：结合年化收益、R²和夏普比率
        # 给予年化收益更高的权重，同时考虑趋势稳定性和风险调整后收益
        score = annualized_returns * r_squared # + sharpe * 0.2
        
        # 对于负收益，根据下跌幅度增加惩罚
        # if annualized_returns < 0:
        #     score = score * (1 + annualized_returns)  # 下跌越多，惩罚越大
        score_list.append(score)
    df = pd.DataFrame(index=etf_pool, data={'score':score_list})
    df = df.sort_values(by='score', ascending=False)
    kk = dict(df.iloc[0])['score']
    if kk<0: print(df.index[0], kk)
    rank_list = list(df.index)    
    print(df)
    record(**{v:round(df.loc[k],2) for k,v in g.etf_pool.items()})
    # record(黄金 = round(df.loc['518880.XSHG'], 2))
    # record(纳指 = round(df.loc['513100.XSHG'], 2))
    # record(成长 = round(df.loc['159915.XSHE'], 2))
    # record(价值 = round(df.loc['510180.XSHG'], 2))
    # record(证券 = round(df.loc['512880.XSHG'], 2))
    # record(中概 = round(df.loc['513050.XSHG'], 2))
    return rank_list# if kk>0 else []
    
def calculate_ma_energy(stock, window=20):
    """计算20日均线能量[2,4](@ref)"""
    # 获取60天数据确保均线计算稳定性
    df = attribute_history(stock, window+400, '1d', ['close'])
    closes = df.close
    
    # 计算20日均线
    ma20 = closes.rolling(window=20).mean().dropna()
    
    # 统计连续上涨周期数
    energy = 0
    for i in range(2, len(ma20)):
        if ma20[i] > ma20[i-1]:
            energy = energy +1 if energy >=0 else -1
        elif ma20[i] < ma20[i-1]:
            energy = energy -1 if energy <=0 else 1
    return energy
    
def min_corr(stocks):
    nday = 729
    p = history(nday, '1d', 'close', stocks).dropna(axis=1)
    r = np.log(p).diff()[1:]
    v = r.std()*math.sqrt(243)
    v = v[(v>0.05) & (v<0.33)]
    r = r[v.index]
    m_corr = r.corr()
    corr_mean = {}
    for stock in m_corr.columns:
        corr_mean[stock] = m_corr[stock].abs().mean()
    etf_pool = sorted(corr_mean, key=corr_mean.get)[:4]
    return etf_pool

def get_rank2(etf_pool):
    # etf_pool = min_corr(etf_pool)
    # print(f"ETF池: {etf_pool}")
    score_list = []
    for etf in etf_pool:
        df = attribute_history(etf, g.m_days, '1d', ['close'])
        y = df['log'] = np.log(df.close)
        x = df['num'] = np.arange(df.log.size)
        slope, intercept = np.polyfit(x, y, 1)
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        r_squared = 1 - (sum((y - (slope * x + intercept))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
        score = annualized_returns * r_squared
        score_list.append(score)
    df = pd.DataFrame(index=etf_pool, data={'score':score_list})
    df = df[(df['score']>-0.5) & (df['score']<4.5)]
    df = df.sort_values(by='score', ascending=False)
    rank_list = list(df.index)
    return rank_list
    
def get_rank3(etf_pool):
    score_list = []
    for etf in etf_pool:
        score = calculate_ma_energy(etf, g.ma_energy_days)
        score_list.append(score)
    df = pd.DataFrame(index=etf_pool, data={'score':score_list})
    df = df.sort_values(by='score', ascending=False)
    kk = dict(df.iloc[0])['score']
    if kk<0: print(df.index[0], kk)
    rank_list = list(df.index)    
    print(df)
    record(**{v:round(df.loc[k],2) for k,v in g.etf_pool.items()})
    return rank_list if kk>0 else []

# 交易
def trade(context):
    # 获取动量最高的一只ETF
    target_list = get_rank2(g.etf_pool)
    if target_list:
        target_num = 1    
        target_list = target_list[:target_num]
    else:
        target_num = 0
    # print(target_list)
    # 卖出    
    hold_list = list(context.portfolio.positions)
    for etf in hold_list:
        if etf not in target_list:
            order_target_value(etf, 0)
            print('卖出' + str(etf))
        else:
            print('继续持有' + str(etf))
    # 买入
    hold_list = list(context.portfolio.positions)
    if len(hold_list) < target_num:
        value = context.portfolio.available_cash / (target_num - len(hold_list))
        for etf in target_list:
            if context.portfolio.positions[etf].total_amount == 0:
                order_target_value(etf, value)
                print('买入' + str(etf))

