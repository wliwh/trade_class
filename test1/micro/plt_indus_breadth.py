from typing import Union, List, Tuple, Optional, Iterable, Dict
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import datetime
import requests
import akshare as ak
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Kline, Bar, Grid, Line

@dataclass
class Config:
    Indus_Pth: str = str(Path(__file__).parent / 'industries_score.csv')
    JSG_Color_Map = {
        '银行':'#009683',
        '煤炭':'#2a2a2a',
        '钢铁':'#4682b4',
        '有色':'#d4af37',
        'others': '#c0c0c0'
    }
    Base_Code = '399303'
    Min_Aver_Score = 10
    Average_Score = 'aver'
    Core_Indus = [k for k in JSG_Color_Map.keys() if k != 'others']
    Rank_Threshold = 2
    Note_Offset = 75

Config = Config()

def basic_index_getter(code:str, beg:Optional[str], end:Optional[str], name_zh:Optional[str]=None):
    beg = beg.replace('-','').replace('/','')
    end = end.replace('-','').replace('/','')
    try:
        aa = ak.index_zh_a_hist(code, start_date=beg, end_date=end)
        aa.columns = ['date', 'open', 'close','high', 'low', 'volume', 'amount', 'amp', 'pct', 'inc', 'turnrate']
    except requests.exceptions.ConnectionError as e:
        prefix = 'sh' if code.startswith(('6', '5')) or code in ('000001', '000300') else 'sz'
        # Check if code already has prefix (though rare for 6-digit input)
        if not code.startswith(('sh', 'sz')):
            symbol = prefix + code
        else:
            symbol = code
        aa = ak.stock_zh_index_daily(symbol)
        aa['date'] = aa['date'].map(str)
        beg = '-'.join((beg[:4], beg[4:6], beg[6:]))
        end = '-'.join((end[:4], end[4:6], end[6:]))
        aa = aa[(aa['date']>=beg) & (aa['date']<=end)]
    aa.insert(1,'code', code)
    aa.insert(2,'name_zh', name_zh if name_zh else '')
    aa.set_index('date',inplace=True)
    return aa

def parse_other_tb_name(k:str):
    if k.lower() in ('l', 'line', 'lines'):
        return 1
    elif k.lower() in ('b', 'bar', 'bars'):
        return 2
    else:
        return 0
    

def get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return 2400, 1300

def get_industries_rank(indus_csv_pth = None):
    if indus_csv_pth is None: indus_csv_pth = Config.Indus_Pth
    p = pd.read_csv(indus_csv_pth,index_col=0)
    p1 = p.loc[:,~p.columns.isin(('aver','sum'))]
    jsg_names = [x for x in p1.columns if x.startswith(tuple(Config.JSG_Color_Map.keys()))]
    p1 = pd.concat([p1.loc[:,p1.columns.isin(jsg_names)], p1.loc[:,~p1.columns.isin(jsg_names)]], axis=1)
    p2 = p1.rank(axis=1, method='min', ascending=False)
    p2 = pd.concat([p2, p[['aver','sum']]], axis=1)
    p2.fillna(50, inplace=True)
    return p2

def get_industries_dic(rank_df:pd.DataFrame):
    ranks, rdic = list(), dict()
    Core_Indus = [x for x in rank_df.columns if x.startswith(tuple(Config.JSG_Color_Map.keys()))]
    wws = ((rank_df.loc[:,Core_Indus])<Config.Rank_Threshold).any(axis=1)
    rank_df = rank_df[wws & (rank_df['aver']>Config.Min_Aver_Score)]
    for i, row in rank_df.iterrows():
        rdic = dict()
        rdic['date'] = i
        rdic['aver'] = float(round(row['aver'],3))
        rdic['sum'] = float(round(row['sum'],3))
        rr = {i[:2]: int(v) for i,v in row[Core_Indus].to_dict().items() if v < Config.Rank_Threshold}
        pls = [[],[],[]]
        rdic['count'] = len(rr)
        rdic['score'] = sum([5-v for v in rr.values()])
        for k, v in rr.items():
            pls[v-1].append(k)
        rdic['names'] = pls
        ranks.append(rdic)
    return ranks


def calc_zigzag(df: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
    """
    Calculate ZigZag points.
    """
    if df.empty:
        return pd.Series(dtype=float)
    
    dates = df.index
    highs = df['High'].values
    lows = df['Low'].values
    
    pivots = pd.Series(index=dates, dtype=float)
    
    # State: 1 for up, -1 for down, 0 for initial
    trend = 0
    last_pivot_idx = 0
    last_pivot_val = highs[0] # or lows[0], init
    
    # Initialize first point
    # Simple init: assume first point is a pivot (refined later if needed)
    pivots.iloc[0] = (highs[0] + lows[0]) / 2 
    last_pivot_val = pivots.iloc[0]
    
    for i in range(1, len(df)):
        h = highs[i]
        l = lows[i]
        
        if trend == 0:
            if h > last_pivot_val * (1 + threshold):
                trend = 1
                last_pivot_idx = i
                last_pivot_val = h
                pivots.iloc[0] = lows[0] # Adjust start if we go up immediately
            elif l < last_pivot_val * (1 - threshold):
                trend = -1
                last_pivot_idx = i
                last_pivot_val = l
                pivots.iloc[0] = highs[0] # Adjust start if we go down immediately
        
        elif trend == 1: # Up trend
            if h > last_pivot_val:
                # Higher high, update pivot
                last_pivot_idx = i
                last_pivot_val = h
            elif l < last_pivot_val * (1 - threshold):
                # Reversal
                pivots.iloc[last_pivot_idx] = last_pivot_val
                trend = -1
                last_pivot_idx = i
                last_pivot_val = l
        
        elif trend == -1: # Down trend
            if l < last_pivot_val:
                # Lower low, update pivot
                last_pivot_idx = i
                last_pivot_val = l
            elif h > last_pivot_val * (1 + threshold):
                # Reversal
                pivots.iloc[last_pivot_idx] = last_pivot_val
                trend = 1
                last_pivot_idx = i
                last_pivot_val = h
                
    # Set the final pivot
    pivots.iloc[last_pivot_idx] = last_pivot_val
    # Connect to the very last point
    pivots.iloc[-1] = highs[-1] if trend == 1 else lows[-1]
    return pivots


def _create_kline_chart(data, markpoint_data, title_opts, datazoom_opt, ma_lines=None, other_lines_data=None, shadow_highs=None, show_legend=True, zigzag_data=None):
    # Base Kline
    kline = (
        Kline()
        .add_xaxis(data.index.tolist())
        .add_yaxis("kline", data.values.tolist(), markpoint_opts=markpoint_data)
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True, axislabel_opts=opts.LabelOpts(is_show=False), grid_index=0),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                grid_index=0,
                split_number=3,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=True),
                splitarea_opts=opts.SplitAreaOpts(is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)),
            ),
            legend_opts=opts.LegendOpts(is_show=show_legend),
            datazoom_opts=datazoom_opt,
            title_opts=opts.TitleOpts(**title_opts),
        )
    )

    # Overlap Lines (MA, etc.)
    line = Line().add_xaxis(data.index.tolist())
    
    # MA Lines
    if ma_lines:
        for name, series in ma_lines:
            line.add_yaxis(
                series_name=name, y_axis=series, is_smooth=True, is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(width=1.5, opacity=0.8),
                label_opts=opts.LabelOpts(is_show=False)
            )
            
    # Other overlaid lines
    if other_lines_data:
        for name, series in other_lines_data:
            line.add_yaxis(
                series_name=name, y_axis=series, is_smooth=False, is_symbol_show=False,
                linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.8),
                label_opts=opts.LabelOpts(is_show=False)
            )

    # Dummy Legend for Industries
    for indus_name, color in Config.JSG_Color_Map.items():
        if indus_name == 'others': continue
        line.add_yaxis(
            series_name=indus_name, y_axis=[None]*len(data), is_smooth=False, is_symbol_show=True,
            symbol='circle', symbol_size=10, linestyle_opts=opts.LineStyleOpts(width=0),
            itemstyle_opts=opts.ItemStyleOpts(color=color), label_opts=opts.LabelOpts(is_show=False)
        )
        
    # Shadow Limit for Dynamic Y-Axis
    if shadow_highs:
        line.add_yaxis(
            series_name="", y_axis=shadow_highs, is_smooth=True, is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=0, opacity=0), label_opts=opts.LabelOpts(is_show=False)

        )

    # ZigZag Line
    if zigzag_data is not None:
        line.add_yaxis(
            series_name="ZigZag",
            y_axis=zigzag_data.where(pd.notnull(zigzag_data), None).tolist(), # Convert NaNs to None for JSON
            is_smooth=False,
            is_symbol_show=False,
            is_connect_nones=True,
            linestyle_opts=opts.LineStyleOpts(width=1, color="purple", type_="dashed"),
            label_opts=opts.LabelOpts(is_show=False),
            z=10
        )


    return kline.overlap(line)

def _create_volume_chart(data, volume_ser, xaxis_idx=1, yaxis_idx=1):
    return (
        Bar()
        .add_xaxis(data.index.tolist())
        .add_yaxis(
            series_name="Volume", y_axis=volume_ser.tolist(), xaxis_index=xaxis_idx, yaxis_index=yaxis_idx,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode("""function(params) {
                    var colorList;
                    if (barData[params.dataIndex][1] > barData[params.dataIndex][0]) { colorList = '#ef232a'; }
                    else { colorList = '#14b143'; }
                    return colorList;
                }""")
            ),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category", is_scale=True, grid_index=xaxis_idx, boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False), axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False), axislabel_opts=opts.LabelOpts(is_show=False),
                min_="dataMin", max_="dataMax",
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=yaxis_idx, is_scale=True, split_number=2,
                axislabel_opts=opts.LabelOpts(is_show=False), axisline_opts=opts.AxisLineOpts(is_show=False),
                axistick_opts=opts.AxisTickOpts(is_show=False), splitline_opts=opts.SplitLineOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

def _create_additional_charts(other_tbs, ohlc_data, grids, grid_start_idx=2):
    charts = []
    for _i, gg in enumerate(other_tbs):
        n, tb = next(iter(gg.items()))
        tb_type_num = parse_other_tb_name(n)
        
        # Prepare Data
        if isinstance(tb, (list, tuple)):
            leg_pos, tb_clm_names, tbb = grids['others_legend'], tb, ohlc_data
        elif isinstance(tb, (str,int,float)):
            leg_pos, tb_clm_names, tbb = grids['others_legend'], [tb], ohlc_data
        elif isinstance(tb, pd.Series):
            leg_pos = grids['others_legend']
            tb_clm_names = [f'val{_i+1}'] if tb.name is None else [tb.name]
            tbb = pd.DataFrame(tb)
        else:
            leg_pos, tb_clm_names, tbb = grids['others_legend'], tb.columns, tb
            
        grid_idx = grid_start_idx + _i
        
        show_legend = (len(leg_pos) > 1) and ('score' not in tb_clm_names)

        common_axis_opts = opts.AxisOpts(
            grid_index=grid_idx, split_number=3, axistick_opts=opts.AxisTickOpts(is_show=False),
            axislabel_opts=opts.LabelOpts(is_show=True), splitline_opts=opts.SplitLineOpts(is_show=False)
        )
        
        charts.append(None) # Placeholder
        
        if tb_type_num == 1: # Line
            line = Line()
            for cnm in tb_clm_names:
                line.add_xaxis(tbb.index.tolist()).add_yaxis(
                    series_name=cnm, y_axis=tbb.to_clipboard() if cnm=='_v1' else tbb[cnm].tolist(),
                    is_smooth=False, symbol_size=2, linestyle_opts=opts.LineStyleOpts(opacity=1),
                    label_opts=opts.LabelOpts(is_show=False)
                )
            line.set_global_opts(
                xaxis_opts=opts.AxisOpts(type_="category", grid_index=grid_idx, axislabel_opts=opts.LabelOpts(is_show=False)),
                yaxis_opts=opts.AxisOpts(type_="value", grid_index=grid_idx, position='right', split_number=3),
                legend_opts=opts.LegendOpts(is_show=show_legend, pos_top=leg_pos[_i] if len(leg_pos)>1 else None)
            )
            charts[-1] = line
            
        elif tb_type_num == 2: # Bar
            bar = Bar()
            for cnm in tb_clm_names:
                bar.add_xaxis(tbb.index.tolist()).add_yaxis(
                    series_name=cnm, y_axis=tbb[cnm].tolist(), xaxis_index=grid_idx, yaxis_index=grid_idx,
                    label_opts=opts.LabelOpts(is_show=False)
                )
            bar.set_global_opts(
                xaxis_opts=opts.AxisOpts(type_="category", grid_index=grid_idx, axislabel_opts=opts.LabelOpts(is_show=False)),
                yaxis_opts=common_axis_opts,
                legend_opts=opts.LegendOpts(is_show=show_legend, pos_top=leg_pos[_i] if len(leg_pos)>1 else None)
            )
            charts[-1] = bar
            
    return charts

def make_candle_echarts(ohlc:pd.DataFrame,
                 beg:str, end:str,
                 ohlc_names:Union[List,Tuple,None] = None,
                 plt_shape:dict = dict(),
                 plt_title_opts:Optional[dict] = None,
                 plt_volume:bool = True,
                 plt_add_ma:Union[List,Tuple,None] = None,
                 plt_add_lines:Union[List,Tuple,None] = None,
                 plt_add_points:Union[List,Tuple,None] = None,
                 other_tbs:Union[List,Tuple,None] = None,
                 plt_zigzag:Union[bool, float] = False):
    
    # 1. Prepare Data
    ohlc_tb = ohlc.copy()
    std_col_names = ['Open','Close','Low', 'High', 'Volume']
    if ohlc_names:
        trans_d = dict(o='Open',c='Close',h='High',l='Low',v='Volume')
        name_trans = {x:trans_d[x.lower()[0]] for x in ohlc_names if x.lower()[0] in trans_d.keys()}
        ohlc_tb.rename(columns=name_trans,inplace=True)
        
    data = ohlc_tb.loc[beg:end,std_col_names[:4]]
    volume_ser = ohlc_tb.loc[beg:end,std_col_names[4]] if plt_volume else None
    
    # 2. Layout & Grids
    vol_grids = {'kline_cls': ('45%', '55%', '13%'), 'others_cls': ('68%', '82%'), 'others_high': ('12%', '10%'), 'others_legend': ('69%', '83%')}
    nvol_grids = {'kline_cls': ('52%',), 'others_cls': ('65%', '81%'), 'others_high': ('15%', '13%'), 'others_legend': ('66%', '82%')}
    _plt_grids = vol_grids if plt_volume else nvol_grids
    
    if 'plt_width' in plt_shape and 'plt_height' in plt_shape:
        _plt_width, _plt_height = plt_shape['plt_width'], plt_shape['plt_height']
    else:
        scr_w, scr_h = get_screen_size()
        _plt_width = plt_shape.get('plt_width', int(scr_w * 0.95))
        _plt_height = plt_shape.get('plt_height', int(scr_h * 0.85))

    # 3. DataZoom
    _plt_range_len = plt_shape.get('df_range_len', 100)
    _range_len = 90 if int((_plt_range_len*100)/len(data)) > 100 else int((_plt_range_len*100)/len(data))
    
    _plt_wind_n = len(other_tbs) if other_tbs else 0
    _data_zoom_index = 1 if _plt_wind_n else 0
    if plt_volume: _data_zoom_index += 1
    
    _datazoom_opt = [
        opts.DataZoomOpts(is_show=False, type_="inside", xaxis_index=([0, 0] if _data_zoom_index else None), range_start=100-_range_len,range_end=100),
        opts.DataZoomOpts(is_show=True, pos_bottom="2%", xaxis_index=([0, 1] if _data_zoom_index else None), range_start=100-_range_len, range_end=100)
    ]
    for _n in range(_plt_wind_n):
        _datazoom_opt.append(opts.DataZoomOpts(is_show=False, xaxis_index=[0,2+_n], range_start=100-_range_len,range_end=100))

    # 4. MarkPoints & Shadow Limits
    _markpoint_data, shadow_highs = None, []
    _show_note_n = Config.Rank_Threshold - 1
    if isinstance(plt_add_points, Iterable) and len(plt_add_points):
        _candle_points = [
            opts.MarkPointItem(
                coord=[x, int(float(data.loc[x, 'High'])+Config.Note_Offset*(_show_note_n-j))],
                symbol='circle' if v[j] else 'rect', symbol_size=12,
                itemstyle_opts=opts.ItemStyleOpts(color=Config.JSG_Color_Map[sorted(v[j])[-1]] if v[j] else Config.JSG_Color_Map['others'])
            ) for x,v in plt_add_points.items() if x in data.index for j in range(_show_note_n)]
        _markpoint_data = opts.MarkPointOpts(data=_candle_points)
        for d in data.index:
            h = float(data.loc[d, 'High'])
            if d in plt_add_points: h += (5 + Config.Note_Offset*_show_note_n)
            shadow_highs.append(h)

    # 5. Create Components
    _plt_titleopts = {'subtitle': f"{beg} ~ {end}"} if plt_title_opts is None else plt_title_opts
    
    ma_lines = [("MA"+str(d), ohlc_tb['Close'].rolling(d).mean().loc[beg:end].tolist()) for d in plt_add_ma] if plt_add_ma else None
    other_lines_data = [(cnm, ohlc_tb.loc[beg:end,cnm].tolist()) for cnm in plt_add_lines] if plt_add_lines else None
    
    _zigzag_ser = None

    if plt_zigzag:
        th = 0.05 if isinstance(plt_zigzag, bool) else plt_zigzag
        _zigzag_ser = calc_zigzag(data, threshold=th)

    kline_chart = _create_kline_chart(data, _markpoint_data, _plt_titleopts, _datazoom_opt, ma_lines, other_lines_data, shadow_highs, zigzag_data=_zigzag_ser)

    
    vol_chart = _create_volume_chart(data, volume_ser) if plt_volume else None
    
    other_charts = _create_additional_charts(other_tbs, ohlc_tb.loc[beg:end], _plt_grids) if other_tbs else []

    # 6. Assembly Grid
    grid_chart = Grid(init_opts=opts.InitOpts(width="{}px".format(_plt_width), height="{}px".format(_plt_height), animation_opts=opts.AnimationOpts(animation=False)))
    grid_chart.add_js_funcs("var barData = {}".format(data.values.tolist()))
    
    grid_chart.add(kline_chart, grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height=_plt_grids['kline_cls'][0]))
    
    if vol_chart:
        grid_chart.add(vol_chart, grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top=_plt_grids['kline_cls'][1], height=_plt_grids['kline_cls'][2]))
        
    for _n, chart in enumerate(other_charts):
        if chart:
            grid_chart.add(chart, grid_opts=opts.GridOpts(pos_left="10%",pos_right="8%", pos_top=_plt_grids['others_cls'][_n],height=_plt_grids['others_high'][_n]))

    return grid_chart


def draw_future_echart(index_code = Config.Base_Code, tt = False, beg = None, end=None):
    p1 = get_industries_rank()
    p1.sort_index(ascending=True, inplace=True)
    
    if not p1.empty:
        csv_beg = p1.index[0]
        csv_end = p1.index[-1]
        
        if beg is None: beg = csv_beg
        if end is None: end = csv_end
    else:
        if beg is None: beg = '2024-01-01'
        if end is None: end = datetime.datetime.now().strftime('%Y-%m-%d')

    a1 = basic_index_getter(index_code, beg, end)
    
    # Strict alignment: intersect indices to ensure matching dates
    common_idx = a1.index.intersection(p1.index)
    a1 = a1.loc[common_idx]
    p1 = p1.loc[common_idx]
    
    p2 = get_industries_dic(p1)
    p2_tm = [p['date'] for p in p2]
    p2_names = {k['date']: k['names'] for k in p2}
    
    # p2 based score assignment
    a1['score'] = pd.Series([p['score'] for p in p2], index=p2_tm)
    a1.fillna({'score':0}, inplace=True)
    
    # Align additional columns
    a1['aver'] = p1['aver']
    a1['sum'] = p1['sum']
    
    tend = make_candle_echarts(a1, beg, end,
                        'open close high low volume'.split(),
                        plt_title_opts={'is_show':tt} if tt==False else {'title':tt}, 
                        plt_volume=False,
                        plt_add_ma=(20,),
                        plt_add_points=p2_names,
                        other_tbs=[{'bar': 'score'}, {'line': Config.Average_Score}],
                        plt_zigzag=0.05)
    return tend


if __name__ == "__main__":
    tend = draw_future_echart()
    tend.render(Path(__file__).parent / 'industries_breadth.html')
    pass