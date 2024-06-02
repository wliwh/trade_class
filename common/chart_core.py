import pandas as pd
import numpy as np
from typing import List, Tuple, Sequence, Iterable, Union, Optional
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.globals import CurrentConfig
from pyecharts.charts import Kline, Bar, Grid, Line


def parse_other_tb_name(k:str):
    if k.lower() in ('l', 'line', 'lines'):
        return 1
    elif k.lower() in ('b', 'bar', 'bars'):
        return 2
    else:
        return 0

def get_grid_hts(snc:Union[int,None] = None,
                 add_vol:bool = True):
    ''' kline 的整体框架设计 '''
    snc = 0 if snc is None else snc
    nc = snc if add_vol else -snc-1
    kline_vol_cls = {
        -3: (52,),
        -2: (60,),
        -1: (80,),
        0: (60,73,18),
        1: (52,63,15),
        2: (45,55,13),
        4: (34,43,11),
    }
    others_cls = {
        -3: (65, 81),
        -2: (73,),
        1: (79,),
        2: (68,82),
        4: (53,64,74,85),
    }
    others_high = {
        -3: (15, 13),
        -2: (20,),
        1: (14,),
        2: (12, 10),
        4: (10, 10, 10, 10)
    }
    others_legend_pos = {
        -3: (66, 82),
        -2: (74,),
        1: (80,),
        2: (69,83),
        4: (53,64,74,85)
    }
    all_cls = dict(
        kline_cls = kline_vol_cls[nc],
        others_cls = others_cls[nc] if snc>0 else 0,
        others_high = others_high[nc] if snc>0 else 0,
        others_legend = others_legend_pos[nc] if snc>0 else 0,
    )
    all_cls = {k:(0 if v==0 else tuple(str(x)+'%' for x in v)) for k,v in all_cls.items()}
    return all_cls

def make_line_echarts(lpic:Union[pd.Series,pd.DataFrame],
                      beg:str, end:str,
                      plt_shape:dict = dict(),
                      plt_title_opts:Optional[dict] = None,
                      plt_volume:Union[pd.Series,bool,None] = False,
                      plt_add_ma:Union[List,Tuple,None] = None,
                      plt_add_lines:Union[pd.Series, pd.DataFrame, None] = None,
                      plt_add_drawdown:Union[List,Tuple,None] = None,
                      other_tbs:Union[List,Tuple,None] = None):
    ''' 输出使用echarts绘制的line图形
        @param lpic,            数据列
        @param beg,end          起始日期-结束日期
        @param plt_shape        line图形的整体大小
        @param plt_title_opts   line标题设置
        @param plt_volume       line中的成交量, 可选
        @param plt_add_ma       在line中添加均线的参数
        @param plt_add_lines    在line中添加别的线
        @param plt_add_drawdown 在line中添加纵向区域或直线
        @param other_tbs        在line之外的窗口添加图形
        需注意 `lpic`, `plt_volume`, `plt_add_lines`, `other_tbs`
        时间范围包含beg--end
    '''
    # lpic_src = lpic.copy()
    _plt_range_len = plt_shape.get('df_range_len',100)
    _plt_width = plt_shape.get('plt_width',1200)
    _plt_height = plt_shape.get('plt_height',700)
    _plt_titleopts = {'subtitle': f"{beg} ~ {end}"} if plt_title_opts is None else plt_title_opts
    if other_tbs:
        assert all(parse_other_tb_name(j)>0 for k in other_tbs for j in k.keys())
    _plt_wind_n = len(other_tbs) if other_tbs is not None else 0
    _plt_grids = get_grid_hts(_plt_wind_n, isinstance(plt_volume, pd.Series))
    _data_zoom_index = 1 if _plt_wind_n else 0

    if not isinstance(lpic.index[0], str):
        lpic.index = lpic.index.map(lambda x:x.strftime('%Y-%m-%d'))
    data = lpic[beg:end].round(3)
    # print(data.tail(5))
    if isinstance(plt_volume,pd.Series):
        _data_zoom_index += 1
        volume_ser = plt_volume[beg:end]
    _range_len = int((_plt_range_len*100)/len(data))
    _range_len = 90 if _range_len>100 else _range_len
    _datazoom_opt = [
        opts.DataZoomOpts(is_show=False, type_="inside", 
                            xaxis_index=[0, 0] if _data_zoom_index else None,
                            range_start=100-_range_len,
                            range_end=100),
        opts.DataZoomOpts(is_show=True, pos_bottom="2%",
                            xaxis_index=[0, 1] if _data_zoom_index else None,
                            range_start=100-_range_len,
                            range_end=100)
    ]
    for _n in range(_plt_wind_n):
        _datazoom_opt.append(
            opts.DataZoomOpts(is_show=False, range_start=100-_range_len,
                    xaxis_index=[0,2+_n],range_end=100))

    _lines = Line().add_xaxis([d for d in data.index])
    if isinstance(lpic, pd.Series):
        _line_name = 'value' if lpic.name is None else lpic.name
        _lines.add_yaxis(_line_name, data.values.tolist(), label_opts=opts.LabelOpts(is_show=False))
    else:
        _line_name = lpic.columns.to_list()
        data = data / data.iloc[0,:]
        for n in _line_name:
            _lines.add_yaxis(n, data[n].values.tolist(),label_opts=opts.LabelOpts(is_show=False))
    lines = (
        _lines
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True),
            yaxis_opts=opts.AxisOpts(
                type_='value',
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            datazoom_opts=_datazoom_opt,
            title_opts=opts.TitleOpts(**_plt_titleopts),
        )
    )

    _added_lines = Line().add_xaxis(xaxis_data=data.index.tolist())
    if plt_add_ma and isinstance(lpic, pd.Series):
        for d in plt_add_ma:
            ma_d = lpic.rolling(d).mean()
            _added_lines.add_yaxis(
                series_name="MA"+str(d),
                y_axis=ma_d.loc[beg:end],
                is_smooth=True,
                is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(width=1.5,opacity=0.8),
                label_opts=opts.LabelOpts(is_show=False),
            )
    if plt_add_lines:
        _add_lines = pd.DataFrame(plt_add_lines) if isinstance(plt_add_lines, pd.Series) else plt_add_lines
        for cnm in _add_lines.columns:
            _added_lines.add_yaxis(
                series_name='added' if isinstance(plt_add_lines, pd.Series) else cnm,
                y_axis=_add_lines.loc[beg:end,cnm],
                is_smooth=False,
                is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(width=3,opacity=0.8),
                label_opts=opts.LabelOpts(is_show=False),
            )
    added_lines = (
        _added_lines.set_global_opts(
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
    overlap_added_lines = lines.overlap(added_lines)

    if isinstance(plt_volume, pd.Series):
        bar = (
            Bar()
            .add_xaxis(xaxis_data=data.index.tolist())  # X轴数据
            .add_yaxis(
                series_name="Volume",
                y_axis=volume_ser.tolist(),  # Y轴数据
                xaxis_index=1,
                yaxis_index=1,
                itemstyle_opts = opts.ItemStyleOpts(opacity=0.7),
                label_opts=opts.LabelOpts(is_show=False)
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
            if isinstance(tb, pd.Series):
                tb_clm_names = [f'val{_i+1}'] if tb.name is None else [tb.name]
                tbb = pd.DataFrame(tb[beg:end])
            else:
                tb_clm_names, tbb = (tb.columns, tb.loc[beg:end])
            if tb_type_num==1:
                othL = Line().add_xaxis(xaxis_data=tbb.index.tolist())
                for cnm in tb_clm_names:
                    othL.add_yaxis(
                            series_name=cnm,
                            y_axis= tbb[cnm].tolist(),
                            is_smooth=False,
                            # yaxis_index=1,
                            symbol_size=2,
                            linestyle_opts=opts.LineStyleOpts(opacity=1),
                            label_opts=opts.LabelOpts(is_show=False),
                        )
                othK = (
                    othL
                    .set_global_opts(
                        tooltip_opts=opts.TooltipOpts(is_show=False,),
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
                        legend_opts=opts.LegendOpts(is_show=False, pos_top='middle'),
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
                        legend_opts=opts.LegendOpts(pos_top='top'),
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

    grid_chart.add(
        overlap_added_lines,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height=_plt_grids['kline_cls'][0]),
    )
    if isinstance(plt_volume, pd.Series):
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

def make_candle_echarts(ohlc:pd.DataFrame,
                 beg:str, end:str,
                 ohlc_names:Union[List,Tuple,None] = None,
                 plt_shape:dict = dict(),
                 plt_title_opts:Optional[dict] = None,
                 plt_volume:bool = True,
                 plt_add_ma:Union[List,Tuple,None] = None,
                 plt_add_lines:Union[List,Tuple,None] = None,
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
    _plt_width = plt_shape.get('plt_width',1200)
    _plt_height = plt_shape.get('plt_height',700)
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

    kline = (
        Kline()
        .add_xaxis([d for d in data.index])
        .add_yaxis("kline", data.values.tolist())
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


def chart_test():
    rpth = r'C:/Users/84066/ZZZZZZZ.csv'
    if 0:
        xx = pd.read_csv(rpth,index_col=0)
        # xx.columns = ['o','h','l','c','inc','pct','vol','amt']
        xx.sort_index(inplace=True)
    else:
        xx = np.random.lognormal(5.44,0.4426,400)
        xx = np.array([np.random.normal(x,x*0.05,10) for x in xx])
        xx = pd.DataFrame(dict(o=xx[:,0],c=xx[:,-1],l=xx.min(axis=1),h=xx.max(axis=1),v=xx[:,4]*10,qq=xx[:,5]*10))
        xx.index = pd.date_range(start='2022-01-10',periods=len(xx),freq='B')
    # tend = make_candle_echarts(xx, '2022-05-31', '2023-05-31',ohlc_names=('o','h','l','c','v'),plt_add_ma=(10,20,60), plt_volume=False,other_tbs=[{'l':['v']}])
    tend = make_line_echarts(xx.c, '2022-05-31', '2023-05-31', plt_add_ma=(10,20,60), other_tbs=[{'line':xx.v}])
    tend.render('ggg.html')
    return xx

if __name__=='__main__':
    chart_test()