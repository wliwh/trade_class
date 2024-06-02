import json
import sys
from datetime import datetime
import streamlit as st
sys.path.append('..')
from index_get.config import INDICATOR_CONFIG_PATH


def get_all_warnings():
    warning_infos = dict()
    with open(INDICATOR_CONFIG_PATH, 'r', encoding='UTF-8') as f:
        try:
            jread = json.load(f)
        except json.JSONDecodeError as e:
            jread = dict()
    for k,v in jread.items():
        if v['warning_info']:
            warning_infos[v["zh"]] = v['warning_info']
    return warning_infos


def st_metrics(n,l):
    date = l[0]
    wk = datetime.strptime(date,'%Y-%m-%d').weekday()
    date_week = "{}, 周{}".format(date,'一二三四五六日'[wk])
    st.subheader(n)
    st.caption(date_week)
    for d in l[1:]:
        cols = st.columns(len(d))
        dter = iter(d.items())
        for c in cols:
            nm, val = next(dter)
            if isinstance(val, float):
                val = round(val,4)
            if isinstance(val,list):
                c.metric(nm, val[0], val[1],delta_color='inverse')
            else:
                c.metric(nm, val)


if __name__ == '__main__':
    st.title("信息提示")
    infos = get_all_warnings()
    for cap, info in infos.items():
        st_metrics(cap, info)