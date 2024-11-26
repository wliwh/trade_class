import json
import os, sys
from datetime import datetime
import time
import streamlit as st
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from index_get.config import INDICATOR_CONFIG_PATH
from common.baidu_utils import Search_Name_Path


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


if __name__ == '__main__':
    # print(get_all_warnings())

    st.title("信息提示")
    infos = get_all_warnings()
    for cap, info in infos.items():
        st_metrics(cap, info)
    pass