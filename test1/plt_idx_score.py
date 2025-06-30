'''
绘制常见ETF相应指数的评分
价格对一次/二次等多项式的拟合，得出其斜率和误差，进而得出评分
'''

import math
import sys
from pathlib import Path
from typing import Optional
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import tkinter as tk
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from common.trade_date import get_trade_day, get_delta_trade_day
from common.smooth_tool import LLT_MA, HMA, TREND_FLEX, RE_FLEX, ema_tan
from index_get.get_index_value import other_index_getter, basic_index_getter, future_index_getter

Selected_Code_Name = ('IXIC','HSTECH','GDAXI','N225','SENSEX',\
                      '159985','518880','162411','501018','159980','159981',\
                      '000015', '399006', '000688', '399303')
Selected_Basic = Selected_Code_Name[:3] + ('518880','501018','159985') + ('000015', '399006', '399303')
Selected_Basic3 = ('518880','501018') + ('IXIC', 'HSTECH', '000015', '399006')
Future_Basic = ('FG0','V0','P0','JM0','m0','RB0','lc0','T0')

def get_screen_size():
    root = tk.Tk()  # 创建临时窗口
    width = root.winfo_screenwidth()  # 获取宽度
    height = root.winfo_screenheight()  # 获取高度
    root.destroy()  # 销毁窗口
    return width, height

def get_index_table(selected:tuple = Selected_Code_Name):
    index_dt = pd.read_csv(Path(Path(__file__).parents[1],'common','csi_index.csv'))
    index_dt = index_dt[index_dt.code.isin(selected)]
    index_dt.reset_index(drop=True,inplace=True)
    return index_dt

def get_index_prices(df:pd.DataFrame, end:Optional[str]=None, count:Optional[int]=None):
    date_fmt = '%Y-%m-%d'
    end_date = get_delta_trade_day(end,-1,date_fmt=date_fmt) if end else get_trade_day(date_fmt=date_fmt)
    beg_date = get_delta_trade_day(end_date,(-(count+1) if count else -301), date_fmt=date_fmt)
    prices = dict()
    for _, row in df.iterrows():
        if row['type_name'].startswith(('other-','nother-')):
            prices[row['code']] = other_index_getter(row['code'], beg_date, end_date)['close']
        elif row['type_name'].startswith('base-'):
            prices[row['code']] = basic_index_getter(row['code'], beg_date, end_date, False)['close']
        elif row['type_name'].startswith('future-'):
            prices[row['code']] = future_index_getter(row['code'], beg_date, end_date)['close']
    prices = pd.DataFrame(prices).iloc[1:]
    prices.interpolate('linear',inplace=True)
    prices = prices.round(3)
    return prices

def calculate_weighted_slope_simplified(llt, weight_factor=0.05):
    weights = np.exp(-weight_factor * np.arange(len(llt))[::-1])
    weights /= weights.sum()
    slopes = np.diff(llt)
    slopes = np.insert(slopes, 0, 0)
    weighted_slope = np.sum(weights * slopes)
    return weighted_slope

def calculate_weighted_r_squared_simplified(close, llt, weight_factor=0.05, llt_penalty=1.0):
    """
    计算加权 R 平方值，衡量 LLT 均线对价格的拟合程度，并对位于LLT下的价格加大权重
    """
    weights = np.exp(-weight_factor * np.arange(len(close))[::-1])
    weights /= weights.sum()

    # 根据 close 与 llt 的关系调整权重
    for i in range(len(close)):
        if close[i] < llt[i]:
            weights[i] *= llt_penalty  # 增加低于 LLT 的权重
        else:
            weights[i] /= llt_penalty # 减少高于 LLT 的权重

    weights /= weights.sum()  # 重新归一化

    # 计算残差
    residuals = close - llt

    # 计算加权残差平方和
    ss_res = np.sum(weights * residuals**2)

    # 计算加权总平方和
    weighted_mean = np.sum(weights * close)
    ss_tot = np.sum(weights * (close - weighted_mean)**2)

    # 计算加权 R 平方值
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared

def calc_poly(y, windows:int, degree:int=1):
    y1 = y.iloc[-windows:].values
    x = np.arange(len(y1))
    coeffs = np.polyfit(x, y1, degree)
    p = np.poly1d(coeffs)
    y_predicted = p(x)
    return y_predicted

def calc_rolling_score_with_llt(prices:pd.DataFrame, window:int=21, llt_penalty:float=1.0):
    scores = dict()
    weights = np.exp(-(1.0/window) * np.arange(window)[::-1])
    weights /= weights.sum()
    # weights = np.ones(window)/window
    rweights = np.empty((window, prices.shape[1]))
    rweights[:] = weights[:, np.newaxis]
    for p in prices.rolling(window=window+4):
        if len(p)<window+4: continue
        p = np.log(p/p.iloc[0])
        p1 = p.apply(lambda x: calc_poly(x, window, 2), axis=0)
        # slope
        ps = p1.diff(axis=0).iloc[-window:]
        ps = ps.apply(lambda x:x*weights, axis=0)
        ps = ps.sum(axis=0).values
        # R^2
        pcrop = p.iloc[-window:].values
        residuals = pcrop-(p1.iloc[-window:].values)
        rweights[residuals>0] *= llt_penalty
        rweights = rweights/(rweights.sum(axis=0))
        ss_res = np.sum(rweights * (residuals**2), 0)
        rweighted_mean = np.sum(rweights * pcrop, axis=0)
        ss_tot = np.sum(rweights * ((pcrop - rweighted_mean)**2), axis=0)
        r_squared = 1 - ss_res / ss_tot
        r_squared[r_squared<0] = np.nan
        scores[p.index[-1]] = ps*r_squared*1000.0
    scores = pd.DataFrame(scores,index=prices.columns).T
    min_score = np.floor(scores.min().min())
    # scores.fillna(method='ffill',inplace=True)
    scores.ffill(inplace=True)
    return scores

def create_plotly_figure(rows:int, row_heights:list, plt_shape: dict={}):
    fig = make_subplots(
        rows=rows, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.0,
        row_heights=row_heights,
        specs=[[{"secondary_y": True}]] * rows
    )

    _w, _h = get_screen_size()
    
    fig.update_layout(
        width=plt_shape.get('plt_width', int(0.9*_w)),
        height=plt_shape.get('plt_height', int(0.9*_h)),
        # margin=dict(l=50, r=50, t=80, b=50),
        margin=dict(t=30, b=30, pad=0),
        autosize=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    return fig


def plot_index(names:tuple = Selected_Basic3):
    from plotly.colors import sample_colorscale

    g = get_index_table(names)
    p1 = get_index_prices(g, count=200)
    code2names = {p:g.loc[g.code==p,'name_zh'].values[0] for p in p1.columns}
    score = calc_rolling_score_with_llt(p1, llt_penalty=1.0)
    dates = score.index
    beg_, end_ = dates.min(), dates.max()
    all_days = set(x.strftime('%Y-%m-%d') for x in pd.date_range(start=beg_,end=end_,freq='D'))
    rm_days = sorted(list(all_days - set(dates)))
    p1 = p1[p1.index>=beg_]
    p1 = p1/p1.iloc[0]
    # adjust score
    max_score, min_score = score.max().max(), score.min().min()
    score = (score - min_score) / (max_score - min_score) * 0.5 + 0.25 # p1.min().min() - 0.5

    fig = create_plotly_figure(1, [1.0])
    fig.update_xaxes(
        # rangeslider_visible=False,
        # rangeselector_visible=False,
        tickformat=r'%m-%d',
        rangebreaks=[dict(values=rm_days)],  # hide holidays (Christmas and New Year's, etc)
        type='date'
    )
    sampled_colors = sample_colorscale('Rainbow', np.linspace(0, 1, p1.shape[1]))
    for c_, n in zip(sampled_colors, p1.columns):
        fig.add_trace(go.Scatter(x=p1.index, y=round(p1[n],3), name=code2names[n], line=dict(color=c_, width=3)),row=1, col=1)
        fig.add_trace(go.Scatter(x=p1.index, y=score[n],name=n, line=dict(color=c_, dash='dash'), showlegend=False), row=1, col=1)
    fig.update_layout(
        title=f"{beg_} --- {end_}",
        showlegend=True,
        # template='plotly_white'
    )
    fig.show()


def get_flex(names:tuple = Selected_Basic3, days:int=400, trend:bool=True):
    g = get_index_table(names)
    p1 = get_index_prices(g, count=days+80)
    p1.bfill(inplace=True)
    sco = p1.apply(TREND_FLEX if trend else RE_FLEX, axis=0)
    # sm, sn = sco.tail(days-40).max().max(), sco.tail(days-40).min().min()
    # sm, sn = math.ceil(sm), math.floor(sn)
    # sco = (sco-sn)/(sm-sn)
    # sco[sco>1.0] = 1.0; sco[sco<0.0]=0.0
    return sco


def get_emas(names:tuple = Selected_Basic3, days:int=200):
    g = get_index_table(names)
    p1 = get_index_prices(g, count=days+80)
    p1.bfill(inplace=True)
    sco = p1.apply(lambda x:ema_tan(x, 10, 21), axis=0)
    return sco


def calc_index_rank(names:tuple = Selected_Basic3, days:int = 200):

    # 获取数据（与原函数相同）
    g = get_index_table(names)
    p1 = get_index_prices(g, count=days)
    code2names = {p:g.loc[g.code==p,'name_zh'].values[0] for p in p1.columns}
    score = calc_rolling_score_with_llt(p1, llt_penalty=1.0)
    dates = score.index
    beg_, end_ = dates.min(), dates.max()
    all_days = set(x.strftime('%Y-%m-%d') for x in pd.date_range(start=beg_,end=end_,freq='D'))
    rm_days = sorted(list(all_days - set(dates)))
    p1 = p1[p1.index>=beg_]
    p1 = p1/p1.iloc[0]
    max_score, min_score = score.max().max(), score.min().min()
    score = (score - min_score) / (max_score - min_score)

    flex_score = get_flex(names, days, True)
    flex_score = flex_score[flex_score.index>=beg_]
    score.columns = list(range(len(score.columns)))
    flex_score.columns = list(range(len(flex_score.columns)))
    g1 = (~score.idxmax(axis=1).diff().isin((0,))).sum()
    g2 = (~flex_score.idxmax(axis=1).diff().isin((0,))).sum()
    print(g1, g2)


def plot_index_separate(names:tuple = Selected_Basic3, days:int = 200):
    from plotly.colors import sample_colorscale
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # 获取数据（与原函数相同）
    g = get_index_table(names)
    p1 = get_index_prices(g, count=days)
    code2names = {p:g.loc[g.code==p,'name_zh'].values[0] for p in p1.columns}
    score = calc_rolling_score_with_llt(p1, llt_penalty=1.0)
    dates = score.index
    beg_, end_ = dates.min(), dates.max()
    all_days = set(x.strftime('%Y-%m-%d') for x in pd.date_range(start=beg_,end=end_,freq='D'))
    rm_days = sorted(list(all_days - set(dates)))
    p1 = p1[p1.index>=beg_]
    p1 = p1/p1.iloc[0]
    max_score, min_score = score.max().max(), score.min().min()
    score = (score - min_score) / (max_score - min_score)

    flex_score = get_flex(names, days, True)
    flex_score = flex_score[flex_score.index>=beg_]
    # print(pd.concat([score.idxmax(axis=1), flex_score.idxmax(axis=1)], axis=1).tail(40))
    
    # 创建两个独立的图表
    fig = create_plotly_figure(rows=3, row_heights=[0.6, 0.2, 0.2])
    # fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    
    # 设置共享的x轴配置
    xaxis_config = dict(
        tickformat='%m-%d',
        rangebreaks=[dict(values=rm_days)],
        type='date'
    )
    
    # 设置颜色
    sampled_colors = sample_colorscale('Rainbow', np.linspace(0, 1, p1.shape[1]))
    
    # 添加数据到第一个图表（价格）
    for c_, n in zip(sampled_colors, p1.columns):
        fig.add_trace(go.Scatter(
            x=p1.index, 
            y=round(p1[n], 3), 
            name=code2names[n], 
            line=dict(color=c_, width=3),
            legendgroup=n,
            showlegend=True
        ),
        row=1,
        col=1
    )
    
    # 添加数据到第二个图表（分数）
    for c_, n in zip(sampled_colors, p1.columns):
        fig.add_trace(go.Scatter(
            x=p1.index, 
            y=round(score[n],3),
            name=code2names[n],  # 显示相同的名称
            line=dict(color=c_, dash='solid', width=1.5),
            legendgroup=n,
            showlegend=False  # 避免图例重复
        ),
        row=2,
        col=1
    )

    for c_, n in zip(sampled_colors, p1.columns):
        fig.add_trace(go.Scatter(
            x=p1.index, 
            y=round(flex_score[n],3),
            name=code2names[n],  # 显示相同的名称
            line=dict(color=c_, dash='solid', width=1.5),
            legendgroup=n,
            showlegend=False  # 避免图例重复
        ),
        row=3,
        col=1
    )

    fig.update_layout(
        title=f"指标分析: {beg_} - {end_}",
        xaxis=xaxis_config,
        hovermode='x unified',
        hoverdistance=-1,       # 取消悬停距离限制
    )
    fig.update_xaxes(
        showspikes=True, 
        spikemode='across',  # 跨所有子图显示
        spikesnap='cursor', 
        spikecolor='gray',
        spikethickness=1
    )
    # fig.update_yaxes(
    #     title_text="分数", 
    #     range=[-0.5, 0.5],
    #     row=2, 
    #     col=1
    # )
    # 显示图表
    fig.show()


if __name__ == '__main__':
    # g = get_index_table(Selected_Basic3)
    # p1 = get_index_prices(g, count=250)
    # print(calc_rolling_score_with_llt(p1,21).tail(15))
    calc_index_rank(Selected_Basic, 500)
    # print(get_emas(Selected_Basic3, 200))
    pass