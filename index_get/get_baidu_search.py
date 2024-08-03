import time
import pandas as pd
from pathlib import Path
from datetime import datetime

from core import IndicatorGetter
from config import Update_Cond, DateTime_FMT, _logger
from common.walk_conds import to_numeric
from common.smooth_tool import LLT_MA
from common.trade_date import (
    get_delta_trade_day,
    get_trade_day,
    get_next_update_time,
    get_trade_day_between)
from common.baidu_utils import (
    baidu_search_index,
    choose_cookie,
    Search_Name_Path)

def get_bd_search_table(fpth=Search_Name_Path):
    ''' 生成待检索词列表 '''
    p1 = pd.read_csv(fpth)
    name_lst = p1['name'].to_list()
    return name_lst

def set_bd_search_table(fpth, near_day):
    p1 = pd.read_csv(fpth)
    p1['neardate'] = near_day
    p1.to_csv(fpth,index=False)


def bd_search_tonow(words:list, itempath:str,
                    beg_date:str='2018-01-01'):
    def gn(wstr):
        if isinstance(wstr,str): return wstr
        else: return wstr[0]
    tbs = list()
    name_pd = pd.read_csv(itempath)
    words = [w for w in words if gn(w) not in name_pd['name'].values]
    if not words: return None
    cookie = choose_cookie()
    if cookie is None: return None
    now_date = get_trade_day(date_fmt='%Y-%m-%d')
    if isinstance(words[0], str):
        words = [(w, beg_date) for w in words]
    for nm,sd in words:
        QS1 = pd.date_range(start=sd,end=now_date,freq='QS-JAN')
        QS2 = pd.date_range(start=sd,end=now_date,freq='QE-MAR')
        QS2 = QS2.append(pd.DatetimeIndex([now_date]))
        for d1,d2 in zip(QS1,QS2):
            time.sleep(0.2)
            dt1, dt2 = d1.strftime('%Y-%m-%d'), d2.strftime('%Y-%m-%d')
            print(nm,dt1,dt2,'...')
            tbs.append(baidu_search_index(nm,dt1,dt2,cookie))
    search_pd = pd.concat(tbs,axis=0)
    search_pd.index = search_pd.index.map(lambda x:x.strftime('%Y-%m-%d'))
    search_pd.columns = ['keyword','type','count']
    search_pd['count'] = search_pd['count'].astype('float')
    search_pd['llt_diff'] = 0.0
    recent_date = search_pd.index.max()
    for nm, sd in words:
        date_ex = get_trade_day_between(sd,recent_date,cut_hour=1,date_fmt='%Y-%m-%d')
        dsgn = search_pd.index.map(lambda x:x in date_ex)
        s1 = search_pd.loc[(dsgn) & (search_pd['keyword']==nm),'count']
        search_pd.loc[(dsgn) & (search_pd['keyword']==nm),'llt_diff'] = s1 - LLT_MA(s1,1/20.0)
    append_name_pd = pd.DataFrame(words,columns=['name','date'])
    append_name_pd['neardate'] = recent_date
    name_pd = pd.concat([name_pd,append_name_pd])
    name_pd.to_csv(itempath,index=False)
    return search_pd

def bd_search_nearday(fpth, beg, end=None):
    ''' 获取最近一段时间内关键词的检索量 '''
    cookie = choose_cookie()
    if cookie is None: return None
    words = get_bd_search_table(fpth)
    beg1 = (pd.to_datetime(beg)+pd.offsets.Day(1)).strftime('%Y-%m-%d')
    if end is None: end = datetime.today().strftime('%Y-%m-%d')
    bnear_tb = list()
    btime = (len(words)+4)//5
    for i in range(btime):
        ed_i = None if i==btime-1 else i*5+5
        bnear_tb.append(baidu_search_index(words[slice(i*5,ed_i)], beg1, end, cookie))
    search_pd = pd.concat(bnear_tb, axis=0)
    search_pd.sort_index(inplace=True)
    recent_date = search_pd.index.max().strftime('%Y-%m-%d')
    set_bd_search_table(fpth, recent_date)
    return search_pd

class bsearch_indicator(IndicatorGetter):
    def __init__(self, cator_name='baidu_search', fn=bd_search_nearday):
        super().__init__(cator_name, fn)
    
    def update_data(self):
        conf = self.cator_conf
        nupdate_time = conf['next_update_time']
        now_max_date = conf['max_date_idx']
        beg1_date = str(int(now_max_date[:4])-1)+now_max_date[4:]
        self.uppth = Path(self.project_dir, conf['fpath'])
        if self.now_datetime <= nupdate_time:
            _logger.info(f"{self.cator_name} no data to Update")
            return Update_Cond.Updated
        tday_lst = get_trade_day_between(beg1_date,left=False,date_fmt='%Y-%m-%d')
        pp = bd_search_nearday(conf['itempath'], now_max_date)
        pp.index = pp.index.map(lambda x:x.strftime('%Y-%m-%d'))
        pp['llt_diff'] = 0
        pp.rename(columns={'index':'count'},inplace=True)
        p_in = pp[pp.index.map(lambda x:x in tday_lst)]
        p_out = pp[pp.index.map(lambda x:x not in tday_lst)]
        if p_in.empty:
            pp.to_csv(self.uppth, mode='a', header=False,float_format='%.3f')
            conf['max_date_idx'] = pp.index.max()
            conf['next_update_time'] = get_next_update_time(
                conf['max_date_idx'], conf['morn_or_night'], date_fmt=DateTime_FMT)
            self.set_cator_conf(True, **conf)
            _logger.info(f"{len(pp)} rows data Update to {self.cator_name}")
        else:
            p1 = pd.read_csv(self.uppth, index_col=0)
            p1 = p1.loc[beg1_date:]
            p1 = p1[p1.index.map(lambda x:x in tday_lst)]
            p1 = pd.concat([p1, p_in], axis=0)
            p1['count'] = p1['count'].astype('float')
            item_name = list(set(p1['keyword']))
            for w in item_name:
                p1.loc[p1.keyword==w,'llt_diff'] = p1.loc[p1.keyword==w,'count'] - LLT_MA(p1.loc[p1.keyword==w,'count'],1/20.0)
            p_in = p1.loc[p_in.index.min():]
            p_ret = pd.concat([p_in, p_out],axis=0)
            p_ret.sort_index(inplace=True)
            p_ret.to_csv(self.uppth,mode='a',header=False,float_format='%.3f')
            conf['max_date_idx'] = p_ret.index.max()
            conf['next_update_time'] = get_next_update_time(
                conf['max_date_idx'], conf['morn_or_night'], date_fmt=DateTime_FMT)
            self.set_cator_conf(True, **conf)
            _logger.info(f"{len(p_ret)} rows data Update to {self.cator_name}")

    def append_data(self, words):
        conf = self.cator_conf
        self.uppth = Path(self.project_dir, conf['fpath'])
        app_data = bd_search_tonow(words, conf['itempath'])
        if app_data is None or app_data.empty:
            _logger.info(f"{self.cator_name} no data to Append")
            return None
        p1 = pd.read_csv(self.uppth,index_col=0)
        p1 = pd.concat([p1,app_data],axis=0)
        p1.sort_index(inplace=True)
        p1.to_csv(self.uppth, mode='w',float_format='%.3f')
        _logger.info(f"{len(app_data)} rows data append to {self.cator_name}")
        
    def set_warn_info(self, beg=None):
        conf = self.cator_conf
        if beg is None:
            near_trade_date = get_delta_trade_day(conf['max_date_idx'], 0, date_fmt='%Y-%m-%d')
            if near_trade_date is None:
                near_trade_date = get_delta_trade_day(conf['max_date_idx'], -1, date_fmt='%Y-%m-%d')
        data = pd.read_csv(Path(self.project_dir,conf['fpath']), index_col=0)
        data = to_numeric(data.loc[beg:] if beg else data.loc[near_trade_date])
        cond = pd.read_csv(Path(self.project_dir,conf['itempath']))
        cond = cond.query("count_th>0 or llt_th>0")
        warn_info = list()
        for gs in cond.values:
            ws = data.query("keyword==@gs[0] and (count>=@gs[-2] or llt_diff>=@gs[-1])")
            if not ws.empty:
                del ws['type']
                if beg:
                    ws.loc[:,'thres'] = 2*(ws.loc[:,'count']>gs[-2]).astype('int') + (ws.loc[:,'llt_diff']>gs[-1]).astype('int')
                    warn_info.append(ws)
                else:
                    wdict = ws.iloc[0].to_dict()
                    ct, ltd = wdict['count'], wdict['llt_diff']
                    if ct>gs[-2]: wdict['count'] = (ct, round(ct-gs[-2],3))
                    if ltd>gs[-1]: wdict['llt_diff'] = (ltd, round(ltd-gs[-1],3))
                    warn_info.append(wdict)
        if warn_info:
            if beg:
                warn_info = pd.concat(warn_info)
                warn_info = warn_info[warn_info['thres']>0]
                warn_info.index.name = 'date'
                warn_info.to_csv(Path(self.project_dir, conf['warning_info_path']))
            else:
                warn_info.insert(0, near_trade_date)
                self.set_cator_conf(True, warning_info=warn_info if warn_info else False)

if __name__=='__main__':
    # t1 = bd_search_nearday(Search_Name_Path, '2024-04-11')
    # print(t1)
    # t2 = bd_search_tonow(['人民币汇率'],Search_Name_Path)
    # print(t2)
    q = bsearch_indicator()
    q.update_data()
    # q.append_data(['人民币汇率'])
    # q.set_warn_info('2022-01-01')
    # q.set_warn_info()
    pass