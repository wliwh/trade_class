
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

def make_drop_items(choose_n:int=1, code_name:str='NDX', beg_day:str='2020-01-01'):
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
    glb = global_index_indicator()
    conf = glb.get_cator_conf()
    fpath = conf['fpath']
    p1 = pd.read_csv(fpath)
    pn = p1[(p1.code==code_name) & (p1.date>=beg_day)]
    pn.reset_index(drop=True, inplace=True)
    zh_name = pn.iloc[0]['name_zh']
    warn_info = find_all_breaks(pn).iloc[choose_n].to_dict()
    h_d, e_d = warn_info['high_date'], warn_info['end_date']
    h_dt, e_dt = datetime.strptime(h_d, '%Y-%m-%d'), datetime.strptime(e_d, '%Y-%m-%d')
    if (e_dt - h_dt).days<50:
        beg_day = (e_dt-timedelta(days=55)).strftime('%Y-%m-%d')
    else:
        beg_day = (h_dt-timedelta(days=10)).strftime('%Y-%m-%d')
    # streamlit
    st.title(f"{zh_name}({code_name})&ensp;对称性分析")
    st.subheader(f"{to_chinese_date(h_d, e_d)}")
    st.markdown(f"- 突破类型: :blue[**{warn_info['cross']}**]")
    st.markdown(f"- 高点: {warn_info['high_value']},&ensp;:gray[*{warn_info['high_date']}*]")
    st.markdown(f"- 突破点: {warn_info['cross_ma']},&ensp;:gray[*{warn_info['cross_date']}*]")
    st.markdown(f"- 低点:  {warn_info['low_value']},&ensp;:gray[*{warn_info['low_date']}*]")
    st.markdown(f"- 回撤: {warn_info['pct1']}%,&ensp;{warn_info['minvalue']}")
    st.markdown(f"- 倍率: :red[**{warn_info['ratio']}**]")
    pn = pn[(pn['date']>beg_day) & (pn['date']<=e_d)]
    line_annotate = ((h_d, w) for w in [warn_info['high_value'], warn_info['cross_ma'], warn_info['low_value']]+warn_info['tovalue'])
    fg = plot_candlestick_with_lines(pn, line_annotate, warn_info['cross'])
    st.plotly_chart(fg, use_container_width=False)


if __name__ == "__main__":
    make_drop_items()