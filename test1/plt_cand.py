from typing import Union, List, Tuple, Optional, Iterable
import pandas as pd
from pathlib import Path
import sys
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Kline, Bar, Grid, Line
sys.path.append(str(Path(__file__).parents[1]))


from common.chart_core import parse_other_tb_name, get_grid_hts
from index_get.get_index_value import basic_index_getter

Indus_Pth = str(Path(__file__).parents[1] / 'industries_score.csv')


def get_industries_rank():
    p = pd.read_csv(Indus_Pth,index_col=0)
    p1 = p.loc[:,~p.columns.isin(('aver','sum'))]
    jsg_names = [x for x in p1.columns if x.startswith(('银行','钢铁','煤炭','有色'))]
    p1 = pd.concat([p1.loc[:,p1.columns.isin(jsg_names)], p1.loc[:,~p1.columns.isin(jsg_names)]], axis=1)
    p2 = p1.rank(axis=1, method='min', ascending=False)
    p2 = pd.concat([p2, p[['aver','sum']]], axis=1)
    p2.fillna(50, inplace=True)
    return p2

def get_industries_dic(rank_df:pd.DataFrame):
    ranks, rdic = list(), dict()
    wws = ((rank_df.iloc[:,:4])<4).any(axis=1)
    rank_df = rank_df[wws & (rank_df['sum']>65)]
    for i, row in rank_df.iterrows():
        rdic = dict()
        rdic['date'] = i
        rdic['aver'] = float(round(row['aver'],3))
        rdic['sum'] = float(round(row['sum'],3))
        rr = {i[:2]: int(v) for i,v in row[:4].to_dict().items() if v<3.4}
        pls = [[],[],[]]
        rdic['count'] = len(rr)
        rdic['score'] = sum([5-v for v in rr.values()])
        for k, v in rr.items():
            pls[v-1].append(k)
        rdic['names'] = pls
        ranks.append(rdic)
    return ranks


def make_candle_echarts(ohlc:pd.DataFrame,
                 beg:str, end:str,
                 ohlc_names:Union[List,Tuple,None] = None,
                 plt_shape:dict = dict(),
                 plt_title_opts:Optional[dict] = None,
                 plt_volume:bool = True,
                 plt_add_ma:Union[List,Tuple,None] = None,
                 plt_add_lines:Union[List,Tuple,None] = None,
                 plt_add_points:Union[List,Tuple,None] = None,
                 plt_add_drawdown:Union[List,Tuple,None] = None,
                 other_tbs:Union[List,Tuple,None] = None):
    ''' 输出使用echarts绘制的kline图形
        @param ohlc,            日线表格, 题头至少需要有ohlcv五项
        @param beg,end          起始日期-结束日期
        @param ohlc_names       用于转换ohlc表格的题头
        @param plt_shape        kline图形的整体大小
        @param plt_title_opts   kline标题设置
        @param plt_add_ma       在kline中添加均线的参数
        @param plt_add_lines    在kline中添加别的线
        @param plt_add_drawdown 
        @param other_tbs        添加其余图形
    '''
    # TODO 多个窗口比例尺推测，多tab输出
    ohlc_tb = ohlc.copy()
    std_col_names = ['Open','Close','Low', 'High', 'Volume']
    if ohlc_names is not None:
        trans_d = dict(o='Open',c='Close',h='High',l='Low',v='Volume')
        name_trans = {x:trans_d[x.lower()[0]] for x in ohlc_names if x.lower()[0] in trans_d.keys()}
        ohlc_tb.rename(columns=name_trans,inplace=True)
    # print(ohlc_tb.columns)
    _plt_range_len = plt_shape.get('df_range_len',100)
    _plt_width = plt_shape.get('plt_width',1300)
    _plt_height = plt_shape.get('plt_height',800)
    _plt_titleopts = {'subtitle': f"{beg} ~ {end}"} if plt_title_opts is None else plt_title_opts
    if plt_add_lines:
        assert all(x in ohlc_tb.columns for x in plt_add_lines)
    if other_tbs:
        assert all(parse_other_tb_name(j)>0 for k in other_tbs for j in k.keys())
    _plt_wind_n = len(other_tbs) if other_tbs is not None else 0
    _plt_grids = get_grid_hts(_plt_wind_n,plt_volume)
    _data_zoom_index = 1 if _plt_wind_n else 0

    data = ohlc_tb.loc[beg:end,std_col_names[:4]]
    if plt_volume:
        _data_zoom_index += 1
        volume_ser = ohlc_tb.loc[beg:end,std_col_names[4]]
    _range_len = int((_plt_range_len*100)/len(data))
    _range_len = 90 if _range_len>100 else _range_len

    _datazoom_opt = [
        opts.DataZoomOpts(is_show=False, type_="inside", 
                          xaxis_index=([0, 0] if _data_zoom_index else None), 
                          range_start=100-_range_len,range_end=100),
        opts.DataZoomOpts(is_show=True, pos_bottom="2%",
                          xaxis_index=([0, 1] if _data_zoom_index else None),
                          range_start=100-_range_len, range_end=100)
    ]
    for _n in range(_plt_wind_n):
        _datazoom_opt.append(opts.DataZoomOpts(is_show=False, range_start=100-_range_len,
                    xaxis_index=[0,2+_n],range_end=100))

    if isinstance(plt_add_points, Iterable) and len(plt_add_points):
        _cand_colors = {'银行':'#009683','煤炭':'#4a4a4a','钢铁':'#4682b4','有色':'#d4af37'}
        _candle_points = [
            opts.MarkPointItem(
                coord=[x, 9300+70*(3-j)], # float(data.loc[x, 'High'])
                symbol='circle',
                symbol_size=12,
                itemstyle_opts=opts.ItemStyleOpts(color=_cand_colors[sorted(q)[-1]]))
            for x,v in plt_add_points.items() if x in data.index for j,q in enumerate(v) if q]
        _markpoint_data = opts.MarkPointOpts(data=_candle_points)
    else:
        _markpoint_data = None
    kline = (
        Kline()
        .add_xaxis([d for d in data.index])
        .add_yaxis("kline",
                   data.values.tolist(),
                   markpoint_opts=_markpoint_data)
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            legend_opts=opts.LegendOpts(is_show=True),
            datazoom_opts=_datazoom_opt,
            title_opts=opts.TitleOpts(**_plt_titleopts),
        )
    )

    _kline_line = Line().add_xaxis(xaxis_data=data.index.tolist())
    if plt_add_ma:
        for d in plt_add_ma:
            ma_d = ohlc_tb['Close'].rolling(d).mean()
            _kline_line.add_yaxis(
                series_name="MA"+str(d),
                y_axis=ma_d.loc[beg:end],
                is_smooth=True,
                is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(width=1.5,opacity=0.8),
                label_opts=opts.LabelOpts(is_show=False),
            )
    if plt_add_lines:
        for cnm in plt_add_lines:
            _kline_line.add_yaxis(
                series_name=cnm,
                y_axis=ohlc_tb.loc[beg:end,cnm],
                is_smooth=False,
                is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(width=3,opacity=0.8),
                label_opts=opts.LabelOpts(is_show=False),
            )
    kline_line = (
        _kline_line.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=1,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=1,
                split_number=3,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=True),
            ),
        )
    )
    # Overlap Kline + Line
    overlap_kline_line = kline.overlap(kline_line)

    if plt_volume:
        bar = (
            Bar()
            .add_xaxis(xaxis_data=data.index.tolist())  # X轴数据
            .add_yaxis(
                series_name="Volume",
                y_axis=volume_ser.tolist(),  # Y轴数据
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode(
                    """
                    function(params) {
                        var colorList;
                        if (barData[params.dataIndex][1] > barData[params.dataIndex][0]) {
                            colorList = '#ef232a';
                        } else {
                            colorList = '#14b143';
                        }
                        return colorList;
                    }
                    """
                    )
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    is_scale=True,
                    grid_index=1,
                    boundary_gap=False,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    split_number=20,
                    min_="dataMin",
                    max_="dataMax",
                ),
                yaxis_opts=opts.AxisOpts(
                    grid_index=1,
                    is_scale=True,
                    split_number=2,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                    axisline_opts=opts.AxisLineOpts(is_show=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

    other_lines = list()
    if other_tbs:
        for _i, gg in enumerate(other_tbs):
            n, tb = next(iter(gg.items()))
            tb_type_num = parse_other_tb_name(n)
            if isinstance(tb, (list, tuple)):
                leg_pos, tb_clm_names, tbb = _plt_grids['others_legend'], tb, ohlc.loc[beg:end]
            elif isinstance(tb, (str,int,float)):
                leg_pos, tb_clm_names, tbb = [tb], [tb], ohlc.loc[beg:end]
            elif isinstance(tb, pd.Series):
                leg_pos = _plt_grids['others_legend']
                tb_clm_names = [f'val{_i+1}'] if tb.name is None else [tb.name]
                tbb = pd.DataFrame(tb[beg:end])
            else:
                leg_pos, tb_clm_names, tbb = _plt_grids['others_legend'], tb.columns, tb.loc[beg:end]
            if len(leg_pos)==1: leg_pos = None
            if tb_type_num==1:
                othL = Line()
                for cnm in tb_clm_names:
                    othL.add_xaxis(xaxis_data=tbb.index.tolist())\
                        .add_yaxis(
                            series_name=cnm,
                            y_axis= tbb.to_clipboard() if cnm=='_v1' else tbb[cnm].tolist(),
                            is_smooth=False,
                            # yaxis_index=1,
                            symbol_size=2,
                            linestyle_opts=opts.LineStyleOpts(opacity=1),
                            label_opts=opts.LabelOpts(is_show=False),
                        )
                othK = (
                    othL
                    .set_global_opts(
                        tooltip_opts=opts.TooltipOpts(is_show=True,),
                            #trigger="axis", axis_pointer_type="cross"),
                        xaxis_opts=opts.AxisOpts(
                            type_="category",
                            axislabel_opts=opts.LabelOpts(is_show=False)),
                        yaxis_opts=opts.AxisOpts(
                            type_="value",
                            grid_index=2+_i,
                            split_number=3,
                            position='right',
                            axistick_opts=opts.AxisTickOpts(is_show=True),
                            splitline_opts=opts.SplitLineOpts(is_show=True),
                        ),
                        legend_opts=opts.LegendOpts(
                            is_show=(leg_pos is not None),
                            pos_top=leg_pos[_i] if leg_pos else None
                        ),
                    )
                )
            elif tb_type_num==2:
                othB = Bar()
                for cnm in tb_clm_names:
                    othB.add_xaxis(xaxis_data=tbb.index.tolist())\
                        .add_yaxis(
                            series_name=cnm,
                            y_axis=tbb[cnm].tolist(),
                            xaxis_index=1,
                            yaxis_index=1,
                            label_opts=opts.LabelOpts(is_show=False),
                        )
                othK = (
                    othB
                    .set_global_opts(
                        xaxis_opts=opts.AxisOpts(
                            type_="category",
                            grid_index=2+_i,
                            axislabel_opts=opts.LabelOpts(is_show=False),
                        ),
                        yaxis_opts=opts.AxisOpts(
                            type_='value',
                            # name='count',
                            grid_index=2+_i,
                            split_number=3,
                            axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                            axistick_opts=opts.AxisTickOpts(is_show=False),
                            splitline_opts=opts.SplitLineOpts(is_show=False),
                            axislabel_opts=opts.LabelOpts(is_show=True),
                        ),
                        legend_opts=opts.LegendOpts(
                            is_show=(leg_pos is not None),
                            pos_top=leg_pos[_i] if leg_pos else None
                        ),
                    )
                )
            other_lines.append(othK)

    grid_chart = Grid(
        init_opts=opts.InitOpts(
            width="{}px".format(_plt_width),
            height="{}px".format(_plt_height),
            animation_opts=opts.AnimationOpts(animation=False),
        )
    )
    grid_chart.add_js_funcs("var barData = {}".format(data.values.tolist()))
    grid_chart.add(
        overlap_kline_line,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height=_plt_grids['kline_cls'][0]),
    )
    if plt_volume:
        grid_chart.add(
            bar,
            grid_opts=opts.GridOpts(
                pos_left="10%", pos_right="8%", pos_top=_plt_grids['kline_cls'][1], height=_plt_grids['kline_cls'][2]
            ),
        )
    for _n in range(_plt_wind_n):
        grid_chart.add(
            other_lines[_n],
            grid_opts=opts.GridOpts(
                pos_left="10%",pos_right="8%", pos_top=_plt_grids['others_cls'][_n],height=_plt_grids['others_high'][_n]
            )
        )
    # grid_chart.render("professional_kline_brush.html")
    return grid_chart

def draw_future_echart(tt:str, beg:str, end:str):
    a1 = basic_index_getter('399303', beg, end, usetdx=False)
    a1_tm = list(a1.index)
    p1 = get_industries_rank()
    p2 = get_industries_dic(p1)
    p2_tm = [p['date'] for p in p2]
    p2_names = {k['date']: k['names'] for k in p2}
    a1['score'] = pd.Series([p['score'] for p in p2], index=p2_tm)
    a1.fillna({'score':0}, inplace=True)
    a1['aver1'] = p1['aver']
    a1['aver2'] = p1['sum']
    tend = make_candle_echarts(a1, beg, end,'open high low close volume'.split(), plt_shape={'plt_height':1250},
                               plt_title_opts={'is_show':tt} if tt==False else {'title':tt},
                               plt_volume=False,
                               plt_add_ma=(20,60,240),
                               plt_add_points=p2_names,
                               other_tbs=[{'bar': 'score'}, {'line': ('aver1','aver2')}])
    return tend


if __name__ == "__main__":
    tend = draw_future_echart(False, '2022-01-01','2025-03-28')
    tend.render('gz.html')
    pass