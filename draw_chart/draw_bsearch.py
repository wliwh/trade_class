import sys, os
import pandas as pd
from pyecharts.charts import Tab
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from common.chart_core import make_candle_echarts
from index_get.get_index_value import future_index_getter, basic_index_getter, other_index_getter

os.chdir(os.path.dirname(__file__))

def draw_future_echart(code:str,name:str,tt:str, beg:str, end:str, type:int=0):
    if type in (1,2) or code.upper()=='USDCNH':
        a1 = other_index_getter(code, beg, end)
    elif len(code)<4:
        a1 = future_index_getter(code, beg, end)
    else:
        a1 = basic_index_getter(code, beg, end, usetdx=False)
    p1 = pd.read_csv('../data_save/bsearch_calc.csv',index_col=0)
    p1 = p1.query("keyword=='{}'".format(name))
    p2 = pd.read_csv('../data_save/bd_handle.csv')
    p2 = p2.loc[p2['keyword']==name, ['date','thres']]
    a1['word_count'] = p1['count']
    a1['diff'] = p1['llt_diff']
    tend = make_candle_echarts(a1, beg, end,'open high low close volume'.split(), plt_shape={'plt_height':1250},
                               plt_title_opts={'is_show':tt} if tt==False else {'title':tt},
                               plt_add_ma=(20,60,240), plt_add_points=p2.values, 
                               other_tbs=[{'bar':a1['word_count']},{'bar':a1['diff']}])
    return tend

def draw_echarts(beg:str, end:str):
    tbs = (
        ('螺纹钢','RB0','黑色-螺纹', 1000),
        ('热卷','hc0','黑色-热卷', 270),
        ('铁矿石','i0','黑色-铁矿石', 1150),
        ('铜价','cu0','有色-铜', 3000),
        ('原油','sc0','化工-石油', 1500),
        ('生猪','lh0','农产品-生猪', 550),
        ('豆粕价格','m0','农产品-豆粕', 272),
        ('纸浆','sp0','轻工-纸浆', 400),
        ('人民币汇率','USDCNH','汇率-离岸',0)
    )

    tb2 = (
        ('a股','000001','股指-a股',0),
        ('上证指数','000001','股指-上证',0),
        ('上证50','000016','股指-大盘',0),
        ('创业板指','399006','股指-创业板',0),
        ('科创50','000688','股指-科创50',0),
    )

    for tnm, tzh in zip((tbs, tb2), ('大宗', '国内')):
        tab = Tab()
        for nm, code, tit, _ in tnm:
            tend = draw_future_echart(code,nm,f'{tit}: {nm}',beg,end)
            tab.add(tend, tit.split('-')[0])
        tab.render(os.path.join(os.path.dirname(__file__), f'{tzh}.html'))

if __name__=='__main__':
    draw_echarts('2025-01-01','2026-01-05')