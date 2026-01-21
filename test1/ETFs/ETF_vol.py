# 风险及免责提示：该策略由聚宽用户在聚宽社区分享，仅供学习交流使用。
# 原文一般包含策略说明，如有疑问请到原文和作者交流讨论。
# 原文网址：https://www.joinquant.com/post/49615
# 标题：波动率过滤后相关性最小etf轮动
# 作者：开心果
# 原回测条件：2017-01-01 到 2024-08-23, ￥100000, 每天

import numpy as np
import pandas as pd
#初始化函数 
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option("avoid_future_data", True)
    log.set_level('system', 'error')
    g.etf_pool = ['512660.XSHG', '511010.XSHG', '510880.XSHG', '159915.XSHE', '513050.XSHG', '510050.XSHG', '588100.XSHG', '512100.XSHG', '518800.XSHG', '513060.XSHG', '512980.XSHG', '512010.XSHG', '513100.XSHG', '512720.XSHG', 
                    '512070.XSHG', '515880.XSHG', '159920.XSHE', '159922.XSHE', '513520.XSHG', '515000.XSHG', '515790.XSHG', '515700.XSHG', '159825.XSHE', '512400.XSHG', '512200.XSHG', '513360.XSHG', '512480.XSHG', '510230.XSHG', '159647.XSHE', '159928.XSHE']
    # g.etf_pool = {
    #     '518880.XSHG':'黄金',	# 2013-07-29 || 黄金基金,黄金ETF,黄金ETF基金
    #     '159907.XSHE':'小盘',	# 2006-09-05 || TMTETF,信息ETF,通信ETF,计算机,科技100,科技ETF,科技50,5GETF,AI智能,中证科技,信息技术ETF
    #     # '510050.XSHG':'大盘',	# 2005-02-23 || 工银上50,MSCI基金,上证50,上50ETF,A50ETF,MSCIA股,景顺MSCI,MSCI中国,天弘300,长三角,添富300,100ETF,ZZ800ETF,800ETF,A50基金,MSCI易基,沪50ETF,工银300,华夏300,HS300ETF,300ETF,综指ETF,180ETF,SZ50ETF,平安300,深100ETF银华,沪深300ETF南方,沪深300ETF,深红利ETF,深证100ETF,广发300
    #     '511010.XSHG':'国债',	# 2013-03-25 || 5年地债ETF,招商快线ETF,豆粕ETF,货币ETF,5年地债,城投ETF,十年国债,10年地债
    #     # '159920.XSHE':'恒生',	# 2012-10-22 || H股ETF,中概互联,恒生通,恒指ETF,港股100,H股ETF
    #     '513100.XSHG':'纳指',	# 2013-05-15 || 纳指ETF
    #     '510880.XSHG':'红利',	# 2007-01-18 || 能源ETF基金,中证红利,100红利,能源ETF,有色ETF
    # }
    # g.etf_pool = {
    #     #'159901.XSHE':'深证100',	# 2006-04-24 || 有色60ETF,深证成指ETF,ZZ500ETF,300成长,广发500,500ETF,稀土ETF,有色金属ETF,化工ETF,钢铁ETF,500指增,国泰500,化工ETF,矿业ETF,畜牧ETF,碳中和E,双碳ETF,畜牧养殖,家电ETF,家电基金,红利质量ETF,化工龙头,中证500ETF博时,增强ETF,中药ETF,稀土ETF,有色ETF,中证500,ETF500,中证500ETF鹏华,家电ETF,养殖ETF,汽车ETF,农业ETF,中证500ETF,化工50,500ETF增强,稀土基金,深成ETF,中药ETF,双碳ETF,碳中和ETF南方,中小100ETF
    #     '518880.XSHG':'黄金',	# 2013-07-29 || 黄金9999,黄金基金,黄金ETF,工银黄金,上海金,金ETF,黄金ETF基金
    #     '159920.XSHE':'恒生',	# 2012-10-22 || 恒生股息,恒生通,恒生国企ETF,港股100,港股红利,恒指ETF,港股通50,恒生红利ETF,H股ETF,H股ETF
    #     '510050.XSHG':'50',	# 2005-02-23 || 消费50,国信价值,沪深300ETF南方,HS300,深红利ETF,800ETF,沪深300ETF泰康,MSCI易基,金融ETF,添富300,中证A100ETF基金,红利ETF,HS300E,中国A50,交运ETF,万家50,300增强,上50ETF,上证50,上海国企,工银上50,中国A50ETF,A50ETF,100ETF,上证ETF,天弘300,广发300,工银300,物流ETF,A50基金,SZ50ETF,300增ETF,180ETF,MSCIA50,300ETF,HS300ETF,国货ETF,华夏300,综指ETF,沪深300ETF
    #     '513100.XSHG':'纳指',	# 2013-05-15 || 法国ETF,东证ETF,纳指生物,日经225,纳斯达克,德国ETF,日经ETF,标普ETF,标普500,标普500ETF,亚太精选ETF,225ETF,日经ETF,纳指ETF,纳斯达克ETF
    #     #'159907.XSHE':'2000',	# 2011-08-10 || ZZ1000,机器人ETF,1000ETF,中证1000ETF,中证1000ETF易方达,1000基金,1000ETF增强,机器人,教育ETF,1000增强ETF,中证1000,工业母机ETF,机床ETF,1000ETF,国证2000ETF
    #     '510880.XSHG':'红利',	# 2007-01-18 || 能源ETF,银行指基,银行ETF,红利低波,资源ETF,红利博时,银行ETF,银行基金,电力ETF,绿色电力ETF,银行股基,华夏银基,中证红利,红利100,共赢ETF,100红利,中国国企,红利50,煤炭ETF,银行ETF天弘,红利300,国企共赢ETF,电力ETF
    #     '511010.XSHG':'国债',	# 2013-03-25 || 汇添富快钱ETF,招商快线ETF,国开ETF,国开债券ETF,国开债ETF,货币ETF,政金债,豆粕ETF,能源化工ETF,有色ETF,5年地债ETF,活跃国债,5年地债,公司债,城投ETF,十年国债,10年地债,短融ETF,0-4地债ETF,国债政金,上证转债,转债ETF
    #     #'512960.XSHG':'央调',	# 2019-01-18 || 央企改革,基建ETF,基建50ETF,基建50,央企创新,创新央企,央创ETF,基建ETF,央企ETF
    # }
    g.m_days = 25  
    run_daily(trade, '10:00')

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

def get_rank(etf_pool):
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

# 交易
def trade(context):
    target_num = 1
    etf_pool = min_corr(g.etf_pool)
    target_list = get_rank(etf_pool)[:target_num]
    # 卖出    
    hold_list = list(context.portfolio.positions)
    for etf in hold_list:
        if etf not in target_list:
            order_target_value(etf, 0)
    # 买入
    hold_list = list(context.portfolio.positions)
    if len(hold_list) < target_num:
        value = context.portfolio.available_cash / (target_num - len(hold_list))
        for etf in target_list:
            if context.portfolio.positions[etf].total_amount == 0:
                order_target_value(etf, value)