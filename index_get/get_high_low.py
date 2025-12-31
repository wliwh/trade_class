"""
获取新高新低数据
"""

import time
from typing import Union
import pandas as pd
import requests
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from common.trade_date import Trade_List, get_trade_day_between
from core import IndicatorGetter

# os.chdir(os.path.dirname(__file__))

Legu_headers = {
    # 'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE'
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
}

High_Low_Legu_Indexs = {
    'all':'sh000985','sz50':'sh000016', 'hs300':'sh000300', 
    'zz500':'sh000905','cyb':'sz399006', 'cy50':'sz399673', 'kc50':'sh000688'}

def high_low_from_legu(symbol: str = "all") -> pd.DataFrame:
    """
    乐咕乐股-创新高、新低的股票数量
    https://www.legulegu.com/stockdata/high-low-statistics
    :param symbol: choice of {'all', 'sz50', 'hs300', 'zz500', 'cyb', 'cy50', 'kc50'}
    :type symbol: str
    :return: 创新高、新低的股票数量
    :rtype: pandas.DataFrame
    """
    if symbol in High_Low_Legu_Indexs:
        url = f"https://www.legulegu.com/stockdata/member-ship/get-high-low-statistics/{symbol}"
    r = requests.get(url,headers=Legu_headers)
    data_json = r.json()
    temp_df = pd.DataFrame(data_json)
    temp_df["date"] = pd.to_datetime(temp_df["date"], unit="ms").dt.date
    del temp_df["indexCode"]
    temp_df.sort_values(['date'], inplace=True, ignore_index=True)
    return temp_df

def get_high_low_legu(date:Union[str,list]):
    """
    根据给定日期获取高低

    该函数从High_Low_Legu_Indexs字典中提取所有symbol的高低指数, 
    并将它们整理成一个DataFrame, 它支持以字符串或列表形式提供日期。

    参数:
    - date: Union[str,list] 一个日期字符串或日期字符串列表，用于检索对应的指数数据。

    返回:
    - DataFrame 包含每个symbol在指定日期的high20和low20指数的DataFrame。
    """
    hl_lst = list()
    sym_lst = High_Low_Legu_Indexs.keys()
    for symbol in sym_lst:
        rs = high_low_from_legu(symbol).set_index('date')
        if isinstance(date, str):
            hl_lst.append(rs.loc[pd.to_datetime(date).date(),'high20':])
        else:
            o_date_idx = pd.to_datetime(rs.index)
            rs.insert(1,'symbol', symbol)
            hl_lst.append(rs.loc[o_date_idx.intersection(pd.to_datetime(date)),'symbol':])
    if isinstance(date, str):
        hl_pd = pd.DataFrame(hl_lst).astype('int')
        hl_pd.insert(0, 'symbol', sym_lst)
    else:
        hl_pd = pd.concat(hl_lst, axis=0)
        hl_pd.loc[:,'high20':] = hl_pd.loc[:,'high20':].astype('int')
    return hl_pd


def check_hl_legu_file_dates(fpth:str):
    ''' 检查HL文件的日期是否完整 '''
    hl_days = set(pd.read_csv(fpth,index_col=0).index)
    all_days = Trade_List.copy()
    all_days['trade_date'] = all_days['trade_date'].apply(lambda x:x.strftime('%Y-%m-%d'))
    rg_days = set(all_days.loc[(all_days['trade_date']>=min(hl_days)) & (all_days['trade_date']<=max(hl_days)),'trade_date'])
    return max(hl_days), rg_days - hl_days

def getter_legu_high_low(fpth:str):
    ''' 添加新日期的HL情况 '''
    left_date, append_dates = check_hl_legu_file_dates(fpth)
    all_dates = list(append_dates) + get_trade_day_between(left_date,left=False,date_fmt='%Y-%m-%d')
    hl_tables = list()
    # for d in all_dates:
    #     hl_tables.append(get_high_low_legu(d))
    hl_tables.append(get_high_low_legu(all_dates))
    return pd.concat(hl_tables,axis=0)


class high_low_legu_indicator(IndicatorGetter):
    def __init__(self, cator_name: str='high_low_legu') -> None:
        super().__init__(cator_name)
        self.update_fun = getter_legu_high_low

if __name__=='__main__':
    # os.chdir(os.path.dirname(__file__))
    # tt = getter_legu_high_low('../data_save/high_low_legu.csv')
    # print(tt)
    p1 = high_low_legu_indicator()
    p1.update_data()
    print(p1.cator_conf['max_date_idx'])
    pass