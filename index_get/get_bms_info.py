import time
import numpy as np
import pandas as pd
import akshare as ak
import gm.api as gmapi
import os.path as osp

ALLSTK_PTH = r'C:\\Users\\84066\\Documents\\trade_tools\\allstock'

gmapi.set_token('8e0cbb59dbaa4d03479e8a311152926f8f7a7885')

def gm_all_stocks_getter(date:str):
    ''' 通过gm-api获取数据 '''
    syms = gmapi.get_symbols(sec_type1=1010, sec_type2=101001, skip_st=False, df=True, trade_date=date)
    syms = syms[['symbol','sec_name','pre_close']]
    sym_lst = syms.symbol.values
    hist_lst = list()
    cnt = 0
    print(' @',end='')
    for slst in np.array_split(sym_lst,sym_lst.size//100):
        ss = ','.join(slst)
        history_data = gmapi.history(ss, frequency='1d',start_time=date,end_time=date,fields='symbol,open,high,low,close,volume,amount',skip_suspended=False, df=True)
        print(cnt+1,end=',')
        turnret = gmapi.stk_get_daily_basic_pt(ss,'turnrate',date,df=True)
        mktvalue = gmapi.stk_get_daily_mktvalue_pt(ss,'tot_mv,a_mv_ex_ltd',date,df=True)
        pevalue = gmapi.stk_get_daily_valuation_pt(ss,'pe_ttm,pb_mrq',date,df=True)
        # mktvalue = gmapi.get_fundamentals(table='trading_derivative_indicator', symbols=ss,start_date=date,end_date=date,fields='TURNRATE,TOTMKTCAP,NEGOTIABLEMV,PB,PETTM', df=True)
        print('',end=';')
        del mktvalue['trade_date']; del pevalue['trade_date']
        # mktvalue['date'] = mktvalue['pub_date'].apply(lambda x:x.strftime('%Y-%m-%d'))
        # mktvalue = mktvalue[['symbol','date','TURNRATE','TOTMKTCAP','NEGOTIABLEMV','PB','PETTM']]
        history_data = history_data.merge(turnret,how='inner',on='symbol')
        history_data = history_data.merge(mktvalue,how='inner',on='symbol')
        history_data = history_data.merge(pevalue,how='inner',on='symbol')
        hist_lst.append(history_data)
        cnt+=1
    hist_tab = pd.concat(hist_lst,axis=0)
    syms = syms.merge(hist_tab,how='inner',on='symbol')
    syms.insert(7,'pct',(syms['close']/syms['pre_close']-1.0)*100.0)
    syms.insert(8,'amp',(syms['high'] - syms['low'])/syms['pre_close']*100.0)
    return syms

def stocks_tocsv(beg:str,end:str):
    now_day = beg
    cnt = 0
    while now_day<=end:
        if cnt ==2: time.sleep(200); cnt=0
        print('\n>>', now_day)
        stk_tb = gm_all_stocks_getter(now_day)
        stk_tb.to_csv(osp.join(ALLSTK_PTH,'stk_{}.csv'.format(now_day.replace('-',''))), float_format='%.2f')
        now_day = gmapi.get_next_trading_date('SHSE',now_day)
        cnt += 1


# print(gm_all_stocks_getter('2024-02-23'))
stocks_tocsv('2022-01-03','2023-12-30')
