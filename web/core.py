import json
import os, sys
import pandas as pd
import streamlit as st
from datetime import datetime
from pyecharts.charts import Tab
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from index_get.config import INDICATOR_CONFIG_PATH
from common.baidu_utils import Search_Name_Path
from draw_chart.draw_bsearch import draw_future_echart

def get_keywords():
    keys = list()
    with open(Search_Name_Path,'r') as f:
        for i, l in enumerate(f):
            if i==0 or ',' not in l: continue
            keys.append(l.split(',')[0])
    return keys


Arange_Info = dict(
    qvix_day = ('50ETF','300ETF','500ETF','1000ETF','CYB','KCB'),
    baidu_search = get_keywords()
)

Bsearch_Page_Name = {0:'国内',1:'香港',2:'海外',3:'大宗'}
Draw_Echarts_Path = os.path.join(os.path.dirname(__file__),'..','draw_chart')


def get_all_warnings():
    warning_infos = dict()
    with open(INDICATOR_CONFIG_PATH, 'r', encoding='UTF-8') as f:
        try:
            jread = json.load(f)
        except json.JSONDecodeError as e:
            jread = dict()
    for k,v in jread.items():
        warns = v['warning_info']
        if warns:
            arange = Arange_Info[k]
            ar_warns = [warns[0]]
            nms = [next(iter(n.values())) for n in warns[1:]]
            for nm in arange:
                if nm in nms:
                    ar_warns.append(warns[nms.index(nm)+1])
            warning_infos[v["zh"]] = ar_warns
    return warning_infos

Warning_Infos = get_all_warnings()


@st.cache_data
def st_metrics(n,l):
    """ 提示信息汇总的展示

    Args:
        n (str): 提示信息类别
        l (list): 提示信息内容, 格式为 [date, i1, i2,...]
    """
    date = l[0]
    wk = datetime.strptime(date,'%Y-%m-%d').weekday()
    date_week = "{}, 周{}".format(date,'一二三四五六日'[wk])
    st.markdown(f'## {n}')
    st.caption(date_week)
    for d in l[1:]:
        cols = st.columns(len(d))
        dter = iter(d.items())
        for c in cols:
            nm, val = next(dter)
            if isinstance(val, float):
                val = round(val,4)
            if isinstance(val,list):
                c.metric(nm, val[0], val[1], delta_color='inverse')
            else:
                c.metric(nm, val)


def main_page(infos=Warning_Infos):
    st.title("信息提示")
    # infos = get_all_warnings()
    for cap, info in infos.items():
        st_metrics(cap, info)


def echart_all_page(allpage:bool=False, infos=Warning_Infos):
    p = pd.read_csv(Search_Name_Path,index_col=0)
    tend_dic = defaultdict(list)
    if allpage:
        p = p.loc[p['type']>=0]
        p['date'] = p['neardate'].map(lambda x:str(int(x[:4])-1)+x[4:])
        for nm, row in p.transpose().to_dict().items():
            tend_dic[row['type']].append((nm, row['code'], row['date'], row['neardate']))
    else:
        for i in infos['搜索指数'][1:]:
            nm = i['keyword']
            date, code, tp = p.loc[p.index==nm,['neardate','code','type']].values[0]
            beg_date = str(int(date[:4])-1)+date[4:]
            if tp>=0: 
                tend_dic[tp].append((nm, code, beg_date, date))
    tends = dict()
    for k,v in tend_dic.items():
        tab = Tab()
        for nm,cd,beg,end in v:
            tend = draw_future_echart(cd,nm,False,beg,end,k)
            tab.add(tend,nm+'-'+cd)
        tab.render(os.path.join(Draw_Echarts_Path,Bsearch_Page_Name[k]+'.html'))
        # tends[Bsearch_Page_Name[k]] = tab
    # return tends


def echart_page(infos, set_id=0):
    tab = Tab()
    p = pd.read_csv(Search_Name_Path,index_col=0)
    for i in infos['搜索指数'][1:]:
        nm = i['keyword']
        date, code, tp = p.loc[p.index==nm,['neardate','code','type']].values[0]
        if tp==set_id: 
            beg_date = str(int(date[:4])-1)+date[4:]
            tend = draw_future_echart(code, nm, False, beg_date,date, tp)
            return tend
            # tab.add(tend, f'{nm}-{code}')
    # return tab


if __name__ == '__main__':
    # pgs = [st.Page(lambda :main_page(infos), title='汇总')]
    # for k,v in echart_page(infos):
    #     fun = lambda :st_pyecharts(v)
    #     fun.__name__ = k+'page'
    #     pgs.append(st.Page(fun, title=k))
    print(Warning_Infos)
    # echart_all_page(True)