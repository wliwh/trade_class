import re
import pandas as pd
import akshare as ak
from typing import Optional, Union, Tuple, List, AnyStr, TypeVar
from functools import wraps
from datetime import datetime, date, time, timedelta
import pandas_market_calendars as mcal

Trade_List = ak.tool_trade_date_hist_sina()

F = TypeVar('F')

def _date_fmt(func: F) -> F:

    @wraps(func)
    def run(*args, **kwargs):
        date_fmt = kwargs.get('date_fmt', None)
        val = func(*args, **kwargs)
        if date_fmt is None: return val
        if val is None: return val
        if isinstance(val, str):
            return val
        if isinstance(val, (date, time)):
            return val.strftime(date_fmt)
        elif isinstance(val, (List,Tuple)):
            val = [v.strftime(date_fmt) for v in val]
            return val
        elif isinstance(val, pd.Series):
            val = val.apply(lambda x:x.strftime(date_fmt))
            return val
    
    return run

@_date_fmt
def get_trade_list(wind:int=400, date_fmt:Union[str, None]=None)->pd.Series:
    tlst = Trade_List.copy()
    return tlst['trade_date'].tail(wind)

@_date_fmt
def get_trade_day(cut_hour:int=16,
                  date_fmt:Union[str, None]=None) -> Union[date, str] :
    ''' 获取最近的交易日, new_hour 为判断新交易日的分割小时 '''
    ntime = datetime.now()
    # ntime = datetime.strptime('2025-07-05 9:00:00','%Y-%m-%d %H:%M:%S')
    n_idx = (Trade_List.trade_date<=ntime.date()).argmin() - 1
    n_day = Trade_List.loc[n_idx, 'trade_date']
    delts = ntime - datetime.combine(n_day, time())
    delts_hours =delts.days*24 + delts.seconds/3600
    # if (ntime.date() > n_day) or (ntime.hour>=cut_hour):
    #     return n_day
    if delts_hours>=cut_hour:
        return n_day
    return Trade_List.loc[n_idx-1,'trade_date']

@_date_fmt
def get_delta_trade_day(day:str='',
                        delta:int=1,
                        date_fmt:Union[str, None]=None) -> Union[date, str, None]:
    ''' 获取某一日的相隔若干个交易日的日期 '''
    if day:
        ndate = pd.to_datetime(day).date()
    else:
        ndate = datetime.today().date()
    n_idx = (Trade_List.trade_date>ndate).argmax()
    if delta>0: 
        return Trade_List.loc[n_idx+delta-1,'trade_date']
    n_trade = day if ndate in Trade_List.trade_date.values else None
    if delta==0:
        return n_trade
    else: 
        return Trade_List.loc[n_idx+delta-(1 if n_trade!=None else 0),'trade_date']

@_date_fmt
def get_next_update_time(day:str, sft_tm:Union[int,str],
                         date_fmt:Union[str, None]=None):
    ''' 获取下一次更新数据的时间, 该时间后可更新下一交易日的数据

        @param day 表示当前数据最新的日期
        @param sft_tm 为整数或表示时间的字符, 整数表示最新日期的下一交易日的某个整点后
        sft_tm 若为字符, 格式为(+|#|^)<h>(:<m>), 表示某时某分并且在小时前面可以添加+/#/^符号. 分别表示最新日期下个交易日当日的某个时点, 或其第二个自然日/交易日的某个时点, 或最新日期的下两个自然日的某个时点
    '''
    trade_tm = datetime.combine(get_delta_trade_day(day,1), time())
    if isinstance(sft_tm, int):
        return trade_tm+timedelta(hours=sft_tm)
    else:
        tm_cpr = r'^(\+|#|\^)?(\d{1,2})[^\d\w]?(\d{1,2})?'
        p, h, m = re.match(tm_cpr, sft_tm).groups()
        h, m = int(h), (int(m) if m else 0)
        if p is None:
            return trade_tm+timedelta(hours=h,minutes=m)
        elif p=='+':
            return trade_tm+timedelta(hours=24+h, minutes=m)
        elif p=='#':
            trade_tm = datetime.combine(get_delta_trade_day(day,2), time())
            return trade_tm+timedelta(hours=h, minutes=m)
        elif p=='^':
            trade_tm = pd.to_datetime(day)
            return trade_tm+timedelta(hours=48+h, minutes=m)

@_date_fmt
def get_trade_day_between(fday:str,
                          eday:Optional[str]=None,
                          cut_hour:int=16,
                          left:bool=True,
                          date_fmt:Union[str, None]=None) -> Union[List,None]:
    ''' 获取fday至最近交易日或指定日期间的所有交易日, 包括两端 '''
    f_idx = (Trade_List.trade_date>=pd.to_datetime(fday).date()).argmax()
    if eday is not None:
        n_idx = (Trade_List.trade_date<=pd.to_datetime(eday).date()).argmin()-1
    else:
        ntime = datetime.now()
        n_idx = (Trade_List.trade_date<=ntime.date()).argmin() - 1
        if (ntime.date() <= Trade_List.loc[n_idx, 'trade_date']) and (ntime.hour<cut_hour):
            n_idx -= 1
    if f_idx>n_idx: return None
    if not left and f_idx==n_idx: return None
    return Trade_List.loc[(f_idx if left else f_idx+1):n_idx,'trade_date'].to_list()

def get_next_weekday(day:str,
                     weekday:int=6,
                     date_fmt:Union[str, None]=None) -> Union[date, str, None]:
    ''' TODO:获取某一日的下个为指定weekday的交易日 '''
    nday = get_delta_trade_day(day,1)
    if nday is None: return None
    return (nday+pd.offsets.Week(1,weekday=weekday-1)).date()
    # return next_wday.strftime(_Date_FMT) if is_str else next_wday

# print(get_trade_day_between('2024-01-02',left=False))

if __name__ == '__main__':
    # t1 = get_delta_trade_day('2024-01-22',-1)
    # print(t1, type(t1))
    # t2 = get_delta_trade_day('2024-01-20',0, date_fmt='%Y-%m-%d')
    # print(t2, type(t2))
    # t3 = get_delta_trade_day('2024-01-21',2, date_fmt='%Y/%m/%d')
    # print(t3, type(t3))

    # t4 = get_trade_list(5)
    # print(t4.values[0], type(t4.values[0]))
    # t5 = get_trade_list(5,date_fmt='%Y%m%d')
    # print(t5.values[1], type(t5.values[1]))

    # t6 = get_trade_day(32)
    # print(t6)
    # t7 = get_trade_day(date_fmt='%y/%m/%d')
    # print(t7, type(t7))

    # t6 = get_trade_day_between('2024-01-19', left=True)
    # print(t6[0], t6[-1], type(t6[-1]))
    # t7 = get_trade_day_between('2024-01-11', left=False, date_fmt='%Y#%m#%d')
    # print(t7[0], t7[-1],type(t7[-3]))

    t8 = get_next_update_time('2025-07-03','+8:30',date_fmt='%Y-%m-%d %H:%M')
    print(t8)
    pass