
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from draw_chart.draw_index_cycle import find_all_breaks, plot_candlestick_with_lines
from index_get.get_index_value import global_index_indicator

def to_chinese_date(d1, d2):
    """
    将日期格式转换为中文格式。
    """
    date_obj1 = datetime.strptime(d1, "%Y-%m-%d")
    date_obj2 = datetime.strptime(d2, "%Y-%m-%d")
    # 格式化为中文
    zh_date1 = date_obj1.strftime("%Y年%#m月%#d日")
    zh_date2 = date_obj2.strftime("%Y年%#m月%#d日") if date_obj2.year!=date_obj1.year else date_obj2.strftime("%#m月%#d日")
    return zh_date1+' -- '+zh_date2

def trans_week_info(t:list):
    w1 = "周{}".format('一二三四五六日'[t[0]-1])
    return f"{w1}, {t[1]}/{t[2]}"

def make_drop_items(pn:pd.DataFrame, zh_name:str='', code_name:str='NDX', beg_day:str='2020-01-01'):
    """
    获取并绘制图，并标注相关价格水平线。

    参数:
        choose_n (int): 选择要分析的警告信息索引，默认为1。

    功能描述:
        1. 从全局配置中获取数据路径和警告信息。
        2. 加载CSV数据文件，筛选出与警告信息中代码相同的记录。
        3. 根据警告信息中的高点日期确定起始日期，以确保图表包含足够的数据。
        4. 准备要标注的价格水平线，包括高点、交叉值、低点和任意其他指定值。
        5. 使用 `plot_candlestick_with_lines` 函数绘制K线图，并添加标水平线。
    """
    # glb = global_index_indicator()
    # conf = glb.get_cator_conf()
    # fpath = conf['fpath']
    # p1 = pd.read_csv(fpath)
    # pn = p1[(p1.code==code_name) & (p1.date>=beg_day)]
    # pn.reset_index(drop=True, inplace=True)
    st.title(f"{zh_name}({code_name})&ensp;对称性分析")
    for _, warn in find_all_breaks(pn).iterrows():
        warn_info = warn.to_dict()
        h_d, e_d = warn_info['high_date'], warn_info['end_date']
        h_dt, e_dt = datetime.strptime(h_d, '%Y-%m-%d'), datetime.strptime(e_d, '%Y-%m-%d')
        if (e_dt - h_dt).days<50:
            beg_day = (e_dt-timedelta(days=55)).strftime('%Y-%m-%d')
        else:
            beg_day = (h_dt-timedelta(days=10)).strftime('%Y-%m-%d')
        # streamlit
        st.subheader(f"{to_chinese_date(h_d, e_d)}")
        st.markdown(f"- 突破类型: :blue[**{warn_info['cross']}**]")
        st.markdown(f"- 高点: {warn_info['high_value']},&ensp;*{warn_info['high_date']}, {trans_week_info(warn_info['high_weeks'])}*")
        st.markdown(f"- 突破点: {warn_info['cross_ma']},&ensp;:gray[*{warn_info['cross_date']}*]")
        st.markdown(f"- 低点:  {warn_info['low_value']},&ensp;*{warn_info['low_date']}, {trans_week_info(warn_info['low_weeks'])}*")
        st.markdown(f"- 回撤: {warn_info['pct1']}%,&ensp;{warn_info['minvalue']}")
        st.markdown(f"- 倍率: :red[**{warn_info['ratio']}**]")
        pcrop = pn[(pn['date']>beg_day) & (pn['date']<=e_d)]
        line_annotate = [(h_d, w) for w in [warn_info['high_value'], warn_info['cross_ma'], warn_info['low_value']]+warn_info['tovalue']]
        fg = plot_candlestick_with_lines(pcrop, line_annotate, warn_info['cross'], warn_info['ratio_int'])
        st.plotly_chart(fg, use_container_width=False)


def multi_cycle_pages(beg:str='2020-01-01'):
    def _work_fun(pp,nm,b):
        px = pp[pp.code==nm]
        zhn = px.iloc[0]['name_zh']
        px.reset_index(drop=True, inplace=True)
        def _plot_fun():
            return make_drop_items(px, zhn, nm, b)
        _plot_fun.__name__ = nm
        return _plot_fun
    code_names = ('SPX','NDX','NDX100','DJIA')
    glb = global_index_indicator()
    conf = glb.get_cator_conf()
    fpath = conf['fpath']
    p1 = pd.read_csv(fpath)
    p1 = p1[p1.date>=beg]
    pg1 = _work_fun(p1, code_names[0], beg)
    pg2 = _work_fun(p1, code_names[1], beg)
    pg = st.navigation([
        st.Page(pg1, title=code_names[0]),
        st.Page(pg2, title=code_names[1]),
    ])
    pg.run()    


if __name__ == "__main__":
    multi_cycle_pages()