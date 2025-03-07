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
# from draw_chart.draw_bsearch import draw_future_echart
from draw_chart.draw_index_cycle import plot_candlestick_with_lines

def get_keywords():
    keys = list()
    with open(Search_Name_Path,'r') as f:
        for i, l in enumerate(f):
            if i==0 or ',' not in l: continue
            keys.append(l.split(',')[0])
    return keys


Arange_Info = dict(
    page1 = dict(
        qvix_day = ('50ETF','300ETF','500ETF','1000ETF','CYB','KCB'),
        baidu_search = get_keywords(),
    ),
    page2 = dict(
        global_index = ('SPX','NDX','DJIA','NDX100')
    )
)

Bsearch_Page_Name = {0:'国内',1:'香港',2:'海外',3:'大宗'}
Draw_Echarts_Path = os.path.join(os.path.dirname(__file__),'..','draw_chart')


def get_all_warnings(arange_info:dict) -> dict:
    warning_infos = dict()
    with open(INDICATOR_CONFIG_PATH, 'r', encoding='UTF-8') as f:
        try:
            jread = json.load(f)
        except json.JSONDecodeError as e:
            jread = dict()
    for k,v in jread.items():
        warns = v['warning_info']
        if warns and arange_info.get(k):
            arange = arange_info[k]
            ar_warns = [warns[0]]
            nms = [next(iter(n.values())) for n in warns[1:]]
            for nm in arange:
                if nm in nms:
                    ar_warns.append(warns[nms.index(nm)+1])
            warning_infos[v["zh"]] = ar_warns
    return warning_infos

Warning_Infos = get_all_warnings(Arange_Info['page1'])


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

def second_page(name:str = 'page2'):
    warns = next(iter(get_all_warnings(Arange_Info[name]).values()))
    st.title('回撤观测')
    for warn in warns[1:]:
        st.header(f"{warn['code']}")
        st.markdown(f"- Cross@:blue[{warn['cross']}]")
        st.markdown(f"High: {warn['high_value']}, :gray[*{warn['high_date']}*]")
        st.markdown(f"Cut:  {warn['cross_ma']}, :gray[*{warn['cross_date']}*]")
        st.markdown(f"Low:  {warn['low_value']}")
        st.markdown(f"Back: {warn['pct1']}%, {warn['pct2']}%")
        st.markdown(f"Mins: :red[**{warn['ratio']}**], {warn['minvalue']}")


if __name__ == '__main__':
    second_page()