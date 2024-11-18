"""
获取每日QVIX数据
"""
import re
import time
import numpy as np
import akshare as ak
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from common.trade_date import get_trade_day_between
from core import IndicatorGetter

SymbolLists = ('50ETF','300ETF','500ETF','CYB','1000ETF','KCB')

def parse_symbol_str(symbol: str, minute: bool = False):
    ''' symbol 输出 '''
    rcp = re.compile(r'(index|cyb|kcb)?(1000|500|300|50)?(etf)?', re.IGNORECASE)
    c_pre, c_mid, _ = rcp.match(symbol).groups()
    if c_pre and c_pre.lower() == 'index' and c_mid == '1000':
        return 'index1000' if minute else 'Index1000'
    elif c_pre and symbol.lower() == 'cyb':
        return 'cyb' if minute else 'CYB'
    elif c_pre and symbol.lower() == 'kcb':
        return 'kcb' if minute else 'KCB'
    elif not c_pre and c_mid:
        if symbol.lower() in ('1000', '1000etf'):
            return 'index1000' if minute else 'Index1000'
        elif symbol[len(c_mid):].lower() in ('', 'etf'):
            return c_mid if minute else c_mid+'ETF'
    else:
        return None


def _test_parse():
    symbol_lst = [
        'index1000', 'Index1000', 'index', 'index10',
        'cyb', 'CYB', 'Cyb', 'cyb300', 'cyb12', '300cyb',
        '50', '300', '500', '1000', '200', '1000etf',
        '50Etf', 'cc300ETF', 'indexETF', '50k'
    ]
    for s in symbol_lst:
        print(s, parse_symbol_str(s))


def option_qvix(symbol: str = '50ETF') -> pd.DataFrame:
    """
    期权波动率指数 QVIX
    http://1.optbbs.com/s/vix.shtml?{symbol}
    [可选: 50ETF, 300ETF, 500ETF, CYB, Index1000]
    :return: 期权波动率指数 QVIX
    :rtype: pandas.DataFrame
    """
    symbol = parse_symbol_str(symbol, False)
    url = "http://1.optbbs.com/d/csv/d/k.csv"
    index_col_dict = {
        '50ETF': [0, 1, 2, 3, 4],
        '300ETF': [0, 9, 10, 11, 12],
        '500ETF': [0, 67, 68, 69, 70],
        'CYB': [0, 71, 72, 73, 74],
        'Index1000': [0, 25, 26, 27, 28],
        'KCB': [0, 83, 84, 85, 86]
    }
    temp_df = pd.read_csv(url).iloc[:, index_col_dict[symbol]]
    temp_df.columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
    ]
    temp_df["date"] = pd.to_datetime(temp_df["date"]).dt.date
    temp_df.dropna(inplace=True)
    temp_df.replace({'#NUM!':np.nan},inplace=True)
    temp_df.fillna(method='ffill',inplace=True)
    return temp_df


def _option_call_put_positon(p1:pd.DataFrame,ename:str='510050'):
    tt = p1.copy()
    s1,s2,s3,s4 = [],[],[],[]
    # tt[['call_pos','call5','put_pos','put5']] = np.nan
    for t in tt.date:
        s_t = t.strftime('%Y%m%d')
        # print(s_t)
        try:
            time.sleep(0.5)
            rr=ak.option_lhb_em(ename,'期权持仓情况-认购持仓量',s_t)
            amts=rr.loc[rr['机构']=='总成交量','持仓量'].values[0]
            amt5=rr.loc[rr['机构']=='前五名合计','持仓量'].values[0]
            s1.append(amts); s2.append(amt5)
        except:
            s1.append(np.nan); s2.append(np.nan)
            print('call',s_t)
        try:
            time.sleep(0.5)
            r2=ak.option_lhb_em(ename,'期权持仓情况-认沽持仓量',s_t)
            amps=r2.loc[r2['机构']=='总成交量','持仓量'].values[0]
            amp5=r2.loc[r2['机构']=='前五名合计','持仓量'].values[0]
            s3.append(amps); s4.append(amp5)
        except:
            s3.append(np.nan); s4.append(np.nan)
            print('put', s_t)
    # print(s1,s3)
    tt['call_pos'] = s1; tt['call5'] = s2
    tt['put_pos'] = s3; tt['put5'] = s4
    tt['call_put'] = tt['call_pos'] / tt['put_pos']
    return tt


def _make_option_day_pds():
    symb_lst = SymbolLists
    ss_pds = list()
    for s in symb_lst:
        print('>>>', s)
        p1 = option_qvix(s)
        p1 = p1.tail(200)
        p1.insert(1,'code',s)
        if s in ('50ETF','300ETF'):
            p1 = _option_call_put_positon(p1,'510050' if s=='50ETF' else '510300')
        ss_pds.append(p1)
    ss_tb = pd.concat(ss_pds,axis=0)
    ss_tb.set_index('date',inplace=True)
    ss_tb.sort_index(inplace=True)
    # ss_tb.to_csv('qvix_day1.csv')
    return ss_tb


def getter_qvix_day(max_date_idx:str) -> pd.DataFrame:
    ''' 根据时间列表填充qvix数据 '''
    dt_lst = get_trade_day_between(max_date_idx, left=False)
    ss_pds = list()
    for s in SymbolLists:
        p1 = option_qvix(s)
        p1 = p1[p1['date'].apply(lambda x:x in dt_lst)]
        p1.insert(1,'code',s)
        if s in ('50ETF','300ETF'):
            p1 = _option_call_put_positon(p1,'510050' if s=='50ETF' else '510300')
        ss_pds.append(p1)
    # print(dt_lst, p1['date'])
    ss_tb = pd.concat(ss_pds,axis=0)
    ss_tb.set_index('date',inplace=True)
    ss_tb.sort_index(inplace=True)
    return ss_tb

class qvix_day_indicator(IndicatorGetter):
    def __init__(self, cator_name='qvix_day') -> None:
        super().__init__(cator_name, getter_qvix_day)

if __name__=='__main__':
    # append_qvix_minute_file()
    # print(option_qvix('50ETF'))
    # option_call_put_positon(option_qvix('50'),'510050')
    # p1 = pd.DataFrame(pd.date_range('20240520',periods=10,freq='b'))
    # p1['date'] = p1[0]
    # print(_option_call_put_positon(p1,'510300'))
    
    # getter_qvix_day('2024-05-20')
    p1 = qvix_day_indicator()
    p1.update_data()
    # p1.set_warn_info()
    print(p1.get_warn_info())
    pass

