"""
获取每日指数(基金)净值
"""

import akshare as ak
import numpy as np
from pytdx.hq import TdxHq_API
import pandas as pd
import efinance as ef
from pathlib import Path
import os, sys
from typing import Optional, Union, List

sys.path.append(Path(__file__).parents[1])
from common.trade_date import (
    get_trade_day,
    get_trade_day_between, 
    get_delta_trade_day)
from common.smooth_tool import drawdown_details
from common.chart_core import make_candle_echarts, make_line_echarts

os.chdir(Path(__file__).parent)
Tdx_Getter = TdxHq_API()

from pytdx.config.hosts import hq_hosts

Tdx_Connect_List = (
    ('119.147.212.81', 7709),
    ('112.74.214.43', 7727),
    ('221.231.141.60', 7709),
    ('101.227.73.20', 7709),
    ('101.227.77.254', 7709),
    ('14.215.128.18', 7709),
    ('59.173.18.140', 7709),
    ('60.28.23.80', 7709),
    ('218.60.29.136', 7709),
    ('122.192.35.44', 7709), 
    ('122.192.35.44', 7709)
)

All_Index_Tb = pd.read_csv(os.path.join('..','common','csi_index.csv'))

def csi_index_getter(code:str, beg:Optional[str], end:Optional[str]):
    ''' 中证指数系列 '''
    beg = beg.replace('-','').replace('/','')
    end = end.replace('-','').replace('/','')
    aa = ak.stock_zh_index_hist_csindex(code,beg,end)
    aa = aa.iloc[:,[0,1,3,6,7,8,9,10,11,12,13]]
    aa.columns = ['date','code','name_zh','open','high','low','close','inc','pct','volume','amount']
    aa.insert(7,'pre-close', aa['close'].shift(1))
    aa.insert(8,'amp', (aa['high']-aa['low'])/aa['pre-close']*100.0)
    aa.insert(2, 'type', 'csi-index')
    aa.set_index('date',inplace=True)
    aa.index = aa.index.map(lambda x:x.strftime('%Y-%m-%d'))
    return aa.iloc[1:]

def sw_index_getter(code:str, beg:Optional[str]=None, end:Optional[str]=None):
    sw_tab = ak.index_hist_sw(code,"day")
    sw_tab.columns = ['code', 'date','close', 'open', 'high', 'low', 'volume', 'amount']
    sw_tab = sw_tab[['date', 'code','open', 'high', 'low', 'close', 'volume', 'amount']]
    zh_name = All_Index_Tb.loc[All_Index_Tb['code'].apply(lambda x:x.strip()==code), 'name_zh'].values[0]
    sw_tab.insert(2,'type','sw-index')
    sw_tab.insert(3,'name_zh',zh_name)
    sw_tab.insert(8,'pre-close', sw_tab.close.shift(1))
    sw_tab.insert(9,'amp',(sw_tab['high']-sw_tab['low'])/sw_tab['pre-close']*100.0)
    sw_tab.insert(10,'inc', sw_tab['close']-sw_tab['pre-close'])
    sw_tab.insert(10,'pct', sw_tab['close']/sw_tab['pre-close']*100.0-100.0)
    sw_tab.set_index('date',inplace=True)
    sw_tab.index = sw_tab.index.map(lambda x:x.strftime('%Y-%m-%d'))
    if beg is not None:
        sw_tab = sw_tab.loc[beg:]
    if end is not None:
        sw_tab = sw_tab.loc[:end]
    return sw_tab

def tdx_index_getter(code:str,
                     beg:Optional[str],
                     end:Optional[str],
                     old_d:bool=False):
    ''' tdx 指数系列 '''
    if old_d:
        beg = get_delta_trade_day(beg,-1,date_fmt='%Y-%m-%d')
    tlst = get_trade_day_between(beg,end,date_fmt='%Y-%m-%d')
    ret_tab = list()
    sc_type = 1
    fday, eday = tlst[0], tlst[-1]
    cnt = (len(tlst)-1)//800+1
    if code.startswith('399'): sc_type = 0
    for i in range(cnt):
        with Tdx_Getter.connect(*Tdx_Connect_List[0]):
            dwk = Tdx_Getter.get_index_bars(9, sc_type, code, i*800, 800)
            if not dwk: break
            dwk = Tdx_Getter.to_df(dwk)
            dwk = dwk.iloc[:,[0,1,2,3,4,5,11,12,13]]
            ret_tab.append(dwk)
    ret_tab.reverse()
    ret_tab = pd.concat(ret_tab, axis=0)
    ret_tab.rename(columns={'vol':'volume'},inplace=True)
    ret_tab['datetime'] = ret_tab['datetime'].apply(lambda x:x.split()[0])
    ret_tab = ret_tab[(ret_tab['datetime']>=fday) & (ret_tab['datetime']<=eday)]
    zh_name = All_Index_Tb.loc[All_Index_Tb['code'].apply(lambda x:x.strip()==code), 'name_zh'].values[0]
    ret_tab.insert(0,'code', code)
    ret_tab.insert(1,'type','tdx-index')
    ret_tab.insert(2,'name_zh', zh_name)
    ret_tab.insert(7,'pre-close', ret_tab['close'].shift(1))
    ret_tab.insert(8,'amp',(ret_tab['high']-ret_tab['low'])/ret_tab['pre-close']*100.0)
    ret_tab.insert(9,'inc', ret_tab['close']-ret_tab['pre-close'])
    ret_tab.insert(9,'pct', ret_tab['close']/ret_tab['pre-close']*100.0-100.0)
    ret_tab.set_index('datetime',inplace=True)
    ret_tab.index.name = 'date'
    # print(All_Index_Tb['code']==code)
    return ret_tab.iloc[1:] if old_d else ret_tab


def other_index_getter(code:str,                               
                       beg:Optional[str]=None,
                       end:Optional[str]=None):
    beg = beg.replace('/','').replace('-','')
    end = end.replace('/','').replace('-','')
    aa = ef.stock.get_quote_history(code, beg, end)
    aa.columns = ['name_zh', 'code', 'date', 'open', 'close','high', 'low', 'volume', 'amount', 'amp', 'pct', 'inc', 'turnrate']
    aa.set_index('date', inplace=True)
    return aa

def future_index_getter(code:str,
                        beg:Optional[str]=None,
                        end:Optional[str]=None):
    aa = ak.futures_zh_daily_sina(symbol=code)
    aa.set_index('date',inplace=True)
    if beg: aa = aa.loc[beg:]
    if end: aa = aa.loc[:end]
    return aa


def basic_index_getter(code:str, beg:Optional[str], end:Optional[str]):
    beg = beg.replace('-','').replace('/','')
    end = end.replace('-','').replace('/','')
    aa = ak.index_zh_a_hist(code, start_date=beg, end_date=end)
    aa.columns = ['date', 'open', 'close','high', 'low', 'volume', 'amount', 'amp', 'pct', 'inc', 'turnrate']
    try:
        tdx_pd = tdx_index_getter(code, beg, end)
        aa['up_count'] = tdx_pd['up_count'].values
        aa['down_count'] = tdx_pd['down_count'].values
    except ValueError as e:
        pass
    zh_name = All_Index_Tb.loc[All_Index_Tb['code'].apply(lambda x:x.strip()==code), 'name_zh'].values[0]
    aa.insert(1,'code', code)
    aa.insert(2,'type','basic-index')
    aa.insert(3,'name_zh', zh_name)
    aa.set_index('date',inplace=True)
    return aa.iloc[1:]


def table_index_getter(beg:Optional[str], end:Optional[str], index_table=All_Index_Tb):
    ret = list()
    for _,c,tp,start in index_table.values:
        c, tp = c.strip(), tp.strip()
        if not isinstance(beg,str): beg = start
        if not isinstance(end,str): end = get_trade_day(date_fmt='%Y-%m-%d')
        if tp=='base-index':
            ret.append(basic_index_getter(c,beg,end))
        elif tp=='csi-index':
            ret.append(csi_index_getter(c,beg,end))
        # elif tp=='sw-index':
        #     ret.append(sw_index_getter(c,beg,end))
        elif tp=='tdx-index':
            ret.append(tdx_index_getter(c,beg,end,True))
    return pd.concat(ret,axis=0)


def get_bond_index(code:str, beg:str='2018-12-31', end:str='2024-01-30'):
    aa = csi_index_getter(code, beg, end)
    # aa.index = pd.to_datetime(aa.index)
    draw = drawdown_details(aa['close'])
    draw.sort_values('max drawdown', inplace=True, ascending=True)
    draw.index = pd.Index(range(1, len(draw)+1))
    return round((np.power(aa.close[-1]/aa.close[0],243/len(aa))-1)*100.0,2), round(-draw.iloc[0,5],2)


def draw_echart_test1():
    tt = basic_index_getter('399317', beg='2012-10-30', end='2024-03-22')
    tt2 = csi_index_getter('932055', beg='2012-10-30', end='2024-03-22')
    gg = csi_index_getter('931080', beg='2012-10-30',end='2024-03-22')
    zz = pd.DataFrame({'全A':tt.close/gg.close,'偏股':tt2.close/gg.close})
    zz /= zz.head(400).min()
    tend = make_line_echarts(pd.DataFrame({'全A':tt.close,'偏股':tt2.close,'30年债':gg.close}),'2012-10-30','2024-03-22',plt_title_opts={'title':'股债比较','subtitle':'国证A指、偏股基金与30年国债'}, plt_volume=False, other_tbs=[{'line':zz.round(3)}, {'line':gg.close.pct_change(60).round(4)*100}])
    tend.render('ggg.html')



if __name__=='__main__':
    # print(get_bond_index('931203'))
    # print(All_Index_Tb['type_name']=='base-index')
    # import sqlite3
    # cx = sqlite3.connect(r'../data_save/funds_index.db')
    # tt = table_index_getter(None, None, All_Index_Tb[All_Index_Tb['type_name']=='csi-index'])
    # tt.to_sql('funds', cx, if_exists='append')
    # tt.to_csv('../data_save/funds_csi.csv',float_format='%.3f')
    # print(future_index_getter('rb0','2024-01-01'))
    # print(other_index_getter('USDCNH','20220101','20240520'))
    pass