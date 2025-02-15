from index_get.get_index_value import basic_index_getter
import pandas as pd
from common.chart_core import make_candle_echarts
from pyecharts.charts import Tab

Beg, End = '2019-01-04', '2024-05-31'

# szzs = basic_index_getter('000001','2019-01-04','2024-05-31')
# zz1000 = basic_index_getter('000852', '2019-01-04','2024-04-19')

def corr_mat_plt():
    corr_mat = pd.concat([
        szzs['close'].rolling(30).corr(sz50.close),
        szzs['close'].rolling(30).corr(hs300.close),
        szzs['close'].rolling(30).corr(zz1000.close)
    ],axis=1)
    corr_mat.columns = ('sz50','hs300','zz1000')

    corr_mat.loc['2022-08-26':'2023-01-05'].plot()
    plt.show()


def draw_qvix_ratio(cword='50ETF'):
    if cword=='50ETF':
        ohlc = basic_index_getter('000016', Beg, End)
    else:
        ohlc = basic_index_getter('000300',Beg, End)
    p1 = pd.read_csv('../data_save/qvix_day.csv',index_col=0)
    p1 = p1.query("code=='{}'".format(cword))
    ohlc['qval'] = p1['close']
    ohlc['qrat'] = p1['cp_div']
    tend = make_candle_echarts(ohlc, Beg, End,'open high low close volume'.split(), plt_title_opts={'title':cword}, plt_add_ma=(20,60,240),other_tbs=[{'line':ohlc['qval']},{'line':ohlc['qrat']}])
    return tend

def draw_qvix_echarts():
    clst = ('50ETF', '300ETF')
    tab = Tab()
    for c in clst:
        tend = draw_qvix_ratio(c)
        tab.add(tend,c)
    tab.render('rbs.html')

if __name__ == '__main__':
    draw_qvix_echarts()