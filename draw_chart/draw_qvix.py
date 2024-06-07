from index_get.get_funds_value import basic_index_getter
import pandas as pd
from common.chart_core import make_candle_echarts

Beg, End = '2019-01-04', '2024-05-31'

# szzs = basic_index_getter('000001','2019-01-04','2024-05-31')
sz50 = basic_index_getter('000016', Beg, End)
# hs300 = basic_index_getter('000300','2019-01-04','2024-04-19')
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
    p1 = pd.read_csv('../data_save/qvix_day.csv',index_col=0)
    p1 = p1.query("code=='{}'".format(cword))
    sz50['qval'] = p1['close']
    sz50['qrat'] = p1['cp_div']
    tend = make_candle_echarts(sz50, Beg, End,'open high low close volume'.split(), plt_title_opts={'title':cword}, plt_add_ma=(20,60,240),other_tbs=[{'line':sz50['qval']},{'line':sz50['qrat']}])
    tend.render('qworks.html')

draw_qvix_ratio()