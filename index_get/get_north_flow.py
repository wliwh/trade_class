"""
获取北向流入数据
note: akshare==1.12.1
"""
import akshare as ak
import pandas as pd
import talib
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from common.smooth_tool import MM_Dist_Ser
from core import IndicatorGetter

MA_List = dict(ma=talib.MA,
               ema=talib.EMA,
               wma=talib.WMA)

try:
    hsgt_north_acc_flow = ak.stock_hsgt_north_acc_flow_in_em
except AttributeError as e:
    import requests
    import json
    def hsgt_north_acc_flow(symbol: str = "沪股通") -> pd.DataFrame:
        url = "http://push2his.eastmoney.com/api/qt/kamt.kline/get"
        params = {
            "fields1": "f1,f3,f5",
            "fields2": "f51,f54",
            "klt": "101",
            "lmt": "5000",
            "ut": "b2884a393a59ad64002292a3e90d46a5",
            "cb": "jQuery18305732402561585701_1584961751919",
            "_": "1584962164273",
        }
        r = requests.get(url, params=params)
        data_text = r.text
        data_json = json.loads(data_text[data_text.find("{") : -2])
        if symbol == "沪股通":
            temp_df = (
                pd.DataFrame(data_json["data"]["hk2sh"])
                .iloc[:, 0]
                .str.split(",", expand=True)
            )
            temp_df.columns = ["date", "value"]
            temp_df['value'] = pd.to_numeric(temp_df['value'])
            return temp_df
        if symbol == "深股通":
            temp_df = (
                pd.DataFrame(data_json["data"]["hk2sz"])
                .iloc[:, 0]
                .str.split(",", expand=True)
            )
            temp_df.columns = ["date", "value"]
            temp_df['value'] = pd.to_numeric(temp_df['value'])
            return temp_df
        if symbol == "北上":
            temp_df = (
                pd.DataFrame(data_json["data"]["s2n"])
                .iloc[:, 0]
                .str.split(",", expand=True)
            )
            temp_df.columns = ["date", "value"]
            temp_df['value'] = pd.to_numeric(temp_df['value'])
            return temp_df

def calc_winds_bias(p:pd.Series, N:int, MA_Fun:str, windows:tuple):
    ma_fun = MA_List.get(MA_Fun.lower(), 'ma')
    ret = pd.DataFrame({'value':p.values}, index=p.index)
    ret['mean'] = ma_fun(p, N)
    ret['bias'] = p / ret['mean'] * 100 - 100
    for w in windows:
        ret[f'Q{w}'] = MM_Dist_Ser(ret['bias'],w)
    return ret

def get_north_acc_flow():
    ''' inflow data '''
    # TODO: stock_hsgt_north_acc_flow_in_em 被移除
    north_acc_flow = hsgt_north_acc_flow('北上')
    north_acc_flow.set_index('date', inplace=True)
    north_acc_flow /= 1e4
    north_acc_flow.fillna(method='ffill', inplace=True)
    north_acc_flow.sort_index(inplace=True)
    return north_acc_flow['value']


def get_hsgt_north():
    ''' purchase data '''
    sh_acc_flow = ak.stock_hsgt_hist_em('沪股通').set_index('日期')
    sz_acc_flow = ak.stock_hsgt_hist_em('深股通').set_index('日期')
    north_pad_flow = sh_acc_flow['历史累计净买额'].add(
        sz_acc_flow['历史累计净买额'], fill_value=0)
    north_pad_flow.index = [d.strftime('%Y-%m-%d')
                            for d in north_pad_flow.index]
    north_pad_flow.index.name = 'date'
    # 2020-10-13 数据空缺
    north_pad_flow.loc['2020-10-13'] = north_pad_flow['2020-10-12']
    north_pad_flow.sort_index(inplace=True)
    return north_pad_flow


def getter_north_flow(N:int, MA_Fun:str, windows:tuple):
    ''' 北向流入数据, 包含偏移量及其分位 '''
    p1 = calc_winds_bias(get_north_acc_flow().iloc[:-1], N, MA_Fun, windows)
    p1.insert(0, 'item', 'inflow')
    p2 = calc_winds_bias(get_hsgt_north(), N, MA_Fun, windows)
    p2.insert(0, 'item', 'purchase')
    p = pd.concat([p1,p2],axis=0)
    p.sort_index(inplace=True)
    if p.index[-1]!=p.index[-2]: p.drop(p.index[-1],inplace=True)
    return p


class north_flow_indicator(IndicatorGetter):
    def __init__(self,
                 cator_name='north_flow',
                 update_fun=getter_north_flow) -> None:
        super().__init__(cator_name, update_fun)


if __name__=='__main__':
    # print(get_north_acc_flow().iloc[:-1])
    # print(getter_north_flow(20,'ema',(60,120,200)).tail(10))
    p1 = north_flow_indicator()
    p1.update_data()
    # p1.set_warn_info()
    print(p1.get_warn_info())
    pass