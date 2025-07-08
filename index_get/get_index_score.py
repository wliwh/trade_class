import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from typing import Optional
from index_get.config import _logger
from common.smooth_tool import TREND_FLEX, RE_FLEX
from common.trade_date import get_delta_trade_day, get_trade_day
from index_get.core import IndicatorGetter
from index_get.get_index_value import other_index_getter, basic_index_getter, future_index_getter

Selected_Code_Name = ('IXIC','HSTECH','GDAXI','N225','SENSEX',\
                      '000015', '399006', '000688', '399303')
Selected_Future_ETF = ('159985','518880','162411','501018','159980','159981')
Selected_Future_Name = ('FG0','V0','P0','JM0','m0','RB0','lc0','T0')

Code_Future_Dict = {
    'world': Selected_Code_Name,
    'futetf': Selected_Future_ETF,
    'future': Selected_Future_Name,
}

def get_index_table(selected:tuple = Selected_Code_Name):
    index_dt = pd.read_csv(Path(Path(__file__).parents[1],'common','csi_index.csv'))
    index_dt = index_dt[index_dt.code.isin(selected)]
    index_dt.reset_index(drop=True,inplace=True)
    return index_dt

def calc_price_with_name(df:pd.DataFrame, get_name:str = 'close'):
    if get_name in ('close', 'open', 'high', 'low'):
        return df[get_name]
    elif get_name.lower() in ('hl','lh','h/l','l/h'):
        return 0.5*(df['high'] + df['low'])
    elif get_name.lower() in ('co','oc','c/o','o/c'):
        return 0.5*(df['close'] + df['open'])
    else:
        raise ValueError(f"Invalid price name: {get_name}")

def get_index_prices(df:pd.DataFrame, end:Optional[str]=None, count:Optional[int]=None, get_name:str = 'close'):
    date_fmt = '%Y-%m-%d'
    end_date = get_delta_trade_day(end,-1,date_fmt=date_fmt) if end else get_trade_day(cut_hour=32,date_fmt=date_fmt)
    beg_date = get_delta_trade_day(end_date,(-(count+1) if count else -301), date_fmt=date_fmt)
    prices = dict()
    for _, row in df.iterrows():
        if row['type_name'].startswith(('other-','nother-')):
            pp = other_index_getter(row['code'], beg_date, end_date)
            prices[row['code']] = calc_price_with_name(pp, get_name)
        elif row['type_name'].startswith('base-'):
            pp = basic_index_getter(row['code'], beg_date, end_date, False)
            prices[row['code']] = calc_price_with_name(pp, get_name)
        elif row['type_name'].startswith('future-'):
            pp = future_index_getter(row['code'], beg_date, end_date)
            prices[row['code']] = calc_price_with_name(pp, get_name)
    prices = pd.DataFrame(prices).iloc[1:]
    prices.interpolate('linear',inplace=True)
    prices = prices.round(3)
    return prices


def calc_poly(y, windows:int, degree:int=1):
    y1 = y.iloc[-windows:].values
    x = np.arange(len(y1))
    coeffs = np.polyfit(x, y1, degree)
    p = np.poly1d(coeffs)
    y_predicted = p(x)
    return y_predicted

def calc_rolling_score(prices:pd.Series, window:int=21, llt_penalty:float=1.0):
    """
    计算单个时间序列的滚动评分，使用LLT惩罚机制
    
    参数:
    prices: pd.Series - 价格时间序列
    window: int - 滚动窗口大小，默认21
    llt_penalty: float - LLT惩罚因子，默认1.0
    
    返回:
    pd.Series - 滚动评分序列
    """
    scores = dict()
    weights = np.exp(-(1.0/window) * np.arange(window)[::-1])
    weights /= weights.sum()
    
    for i in range(window+4, len(prices)+1):
        p = prices.iloc[i-window-4:i]
        if len(p) < window+4: 
            continue
            
        p_log = np.log(p / p.iloc[0])
        p1_values = calc_poly(p_log, window, 2)
        p1 = pd.Series(p1_values, index=p_log.iloc[-window:].index)
        ps = np.diff(p1_values)
        ps = ps * (weights[1:])  # weights需要去掉最后一个元素以匹配diff后的长度
        ps_sum = np.sum(ps)
        pcrop = p_log.iloc[-window:].values
        residuals = pcrop - p1_values
        rweights = weights.copy()
        rweights[residuals > 0] *= llt_penalty
        rweights = rweights / rweights.sum()
        ss_res = np.sum(rweights * (residuals**2))
        rweighted_mean = np.sum(rweights * pcrop)
        ss_tot = np.sum(rweights * ((pcrop - rweighted_mean)**2))
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        r_squared = r_squared if r_squared >= 0 else np.nan
        scores[p.index[-1]] = ps_sum * r_squared * 1000.0
    
    # 转换为Series并处理缺失值
    scores_series = pd.Series(scores)
    scores_series.ffill(inplace=True)
    
    return scores_series


def calc_index_scores(names:tuple = Selected_Code_Name, beg_date:str = None, days:int = 200, type_name:str = 'world', price_name:str = 'close'):
    df_lst = list()
    g = get_index_table(names)
    p1 = get_index_prices(g, count=days+100, get_name=price_name)
    code2names = {p:g.loc[g.code==p,'name_zh'].values[0] for p in p1.columns}
    # score = calc_rolling_score_with_llt(p1, llt_penalty=1.0)
    for n in p1.columns:
        p3 = p1[n].bfill()
        score_flex = pd.Series(TREND_FLEX(p3), index=p3.index)
        score_reflex = pd.Series(RE_FLEX(p3), index=p3.index)
        if beg_date:
            score1 = calc_rolling_score(p1[n], llt_penalty=1.0)
            df1 = pd.DataFrame({'name':n, 'name_zh':code2names[n], 'type_name':type_name,
                                'value':p1.loc[p1.index>beg_date,n],
                                'score':score1[score1.index>beg_date],
                                'trendflex':score_flex[score_flex.index>beg_date],
                                'reflex':score_reflex[score_reflex.index>beg_date]})
        else:
            p2 = p1.iloc[-days:][n]
            score1 = calc_rolling_score(p2, llt_penalty=1.0)
            beg_ = score1.index[0]
            df1 = pd.DataFrame({'name':n, 'name_zh':code2names[n], 'type_name':type_name,
                                'value':p2[p2.index>=beg_], 'score':score1,
                                'trendflex':score_flex[score_flex.index>=beg_],
                                'reflex':score_reflex[score_reflex.index>=beg_]})
        df1.dropna(inplace=True)
        df_lst.append(df1)
    df_all = pd.concat(df_lst,axis=0)
    return df_all

def calc_index_scores_begin(max_date_idx:str):
    from datetime import datetime
    df_lst = list()
    beg_date = datetime.strptime(max_date_idx,'%Y-%m-%d').date()
    end_date = get_trade_day(cut_hour=20)
    days = (end_date-beg_date).days+1
    for n,t in Code_Future_Dict.items():
        df = calc_index_scores(t, beg_date=max_date_idx, days=days+60, type_name=n, price_name='close')
        df_lst.append(df)
    df_all = pd.concat(df_lst,axis=0)
    df_all.sort_index(inplace=True)
    return df_all


def test_merge_index_scores():
    df_lst = list()
    for n,t in Code_Future_Dict.items():
        df = calc_index_scores(t, beg_date=None, days=3000, type_name=n, price_name='close')
        df_lst.append(df)
    df_all = pd.concat(df_lst,axis=0)
    df_all.sort_index(inplace=True)
    return df_all


class index_score_indicator(IndicatorGetter):
    def __init__(self, cator_name: str = 'index_score') -> None:
        super().__init__(cator_name)
        self.update_fun = calc_index_scores_begin

    def set_warn_info(self):
        conf = self.cator_conf
        ws = pd.read_csv(conf['fpath'], index_col=0)
        near_date = conf['max_date_idx']
        # read_date = conf['warning_info'][0]
        now_data = ws[ws.index==near_date]
        infos = [near_date]
        for k in Code_Future_Dict.keys():
            codes = Code_Future_Dict[k]
            filtered_data = now_data.loc[now_data.name.isin(codes)]
            if not filtered_data.empty:
                info = {
                    "type": k,
                    "name": list(codes),
                    "name_zh": filtered_data['name_zh'].to_list(),
                    "score": filtered_data['score'].round(4).to_list(),
                    "trendflex": filtered_data['trendflex'].round(4).to_list(),
                    "reflex": filtered_data['reflex'].round(4).to_list()
                }
                infos.append(info)
        if len(infos) > 1: #and read_date != near_date:
            _logger.info(f"{self.cator_name} warning info updating to {near_date}.")
            self.set_cator_conf(True, warning_info=infos)
        else:
            self.set_cator_conf(True, warning_info=False)
        return infos

if __name__ == '__main__':
    # df = merge_index_scores()
    # df.to_csv('data_save/index_scores.csv')
    # print(get_trade_day(cut_hour=32))

    sco = index_score_indicator()
    sco.update_data()
    sco.set_warn_info()
    pass