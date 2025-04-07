
import os
import sys
from pathlib import Path
from typing import Optional
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from common.trade_date import get_trade_day, get_delta_trade_day
from common.smooth_tool import LLT_MA, HMA, ema_tan
from index_get.get_index_value import other_index_getter, basic_index_getter

Selected_Code_Name = ('IXIC','HSTECH','GDAXI','N225','SENSEX',\
                      '159985','518880','162411','501018','159980','159981',\
                      '000015', '399006', '000688', '399303')
Selected_Basic = Selected_Code_Name[:3] + ('518880','501018','159985') + ('000015', '399006', '399303')
Selected_Basic3 = ('518880','501018') + ('IXIC', 'HSTECH', '000015', '399006')

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

def calc_rolling_score_with_llt(prices:pd.DataFrame, window:int=21, llt_penalty:float=1.0, poly_n:int=2):
    scores = dict()
    weights = np.exp(-(1.0/window) * np.arange(window)[::-1])
    weights /= weights.sum()
    # weights = np.ones(window)/window
    rweights = np.empty((window, prices.shape[1]))
    rweights[:] = weights[:, np.newaxis]
    for p in prices.rolling(window=window*2):
        if len(p)<window*2: continue
        p = np.log(p/p.iloc[0])
        p1 = p.apply(lambda x: calc_poly(x, window, poly_n), axis=0)
        # p1 = p.apply(lambda x: ema_tan(x, 11, window), axis=0)
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
    scores.fillna(min_score,inplace=True)
    return scores

def find_max_score():
    g = get_index_table(Selected_Basic3)
    p1 = get_index_prices(g, count=800)
    hold_dic = dict()
    for n in (1,2):
        score = calc_rolling_score_with_llt(p1, llt_penalty=1.0, poly_n=n)
        hold_dic[n] = score.idxmax(axis=1)
    p = pd.DataFrame(hold_dic)
    p.columns = ('n1','n2')
    p['dif'] = (p['n1']==p['n2'])
    print(p.head())
    p.to_csv('hold_etf.csv')

def create_plotly_figure(rows: int, row_heights: list, plt_shape: dict={}):
    fig = make_subplots(
        rows=rows, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.0,
        row_heights=row_heights,
        specs=[[{"secondary_y": True}]] * rows
    )
    
    fig.update_layout(
        width=plt_shape.get('plt_width', 1400),
        height=plt_shape.get('plt_height', 800),
        # margin=dict(l=50, r=50, t=80, b=50),
        margin=dict(t=30, b=30, pad=0),
        autosize=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    return fig

def plot_index():
    from plotly.colors import sample_colorscale

    g = get_index_table(Selected_Basic3)
    p1 = get_index_prices(g, count=200)
    score = calc_rolling_score_with_llt(p1, llt_penalty=1.0)
    dates = score.index
    beg_, end_ = dates.min(), dates.max()
    all_days = set(x.strftime('%Y-%m-%d') for x in pd.date_range(start=beg_,end=end_,freq='D'))
    rm_days = sorted(list(all_days - set(dates)))
    p1 = p1[p1.index>=beg_]
    p1 = p1/p1.iloc[0]
    # adjust score
    max_score, min_score = score.max().max(), score.min().min()
    score = (score - min_score) / (max_score - min_score) * 2 - 1
    print('Prepare data OK...')

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
        fig.add_trace(go.Scatter(x=p1.index, y=p1[n],name=n, line=dict(color=c_, width=3)),row=1, col=1)
        fig.add_trace(go.Scatter(x=p1.index, y=score[n],name=n, line=dict(color=c_, dash='dash')), row=1, col=1)
    fig.update_layout(
        title=f"{beg_} --- {end_}",
        showlegend=True,
        # template='plotly_white'
    )
    fig.show()


if __name__ == '__main__':
    # g = get_index_table(Selected_Basic)
    # p1 = get_index_prices(g, count=250)
    # print(calc_rolling_score_with_llt(p1,21))
    plot_index()
    # print(find_max_score())
    pass