import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Sequence, Iterable, Union, Optional

def parse_other_tb_name(k: str):
    if k.lower() in ('l', 'line', 'lines'):
        return 1
    elif k.lower() in ('b', 'bar', 'bars'):
        return 2
    else:
        return 0

def get_grid_hts(snc: Union[int, None] = None, add_vol: bool = True):
    snc = 0 if snc is None else snc
    nc = snc if add_vol else -snc-1
    row_heights = {
        -3: [0.6, 0.2, 0.2],
        -2: [0.7, 0.3],
        -1: [1.0],
        0: [0.6, 0.4] if add_vol else [1.0],
        1: [0.52, 0.18, 0.3],
        2: [0.45, 0.15, 0.2, 0.2],
        4: [0.38, 0.15, 0.12, 0.12, 0.12, 0.11]
    }
    return row_heights.get(nc, [1.0])

def create_plotly_figure(rows: int, row_heights: list, plt_shape: dict):
    fig = make_subplots(
        rows=rows, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.0,
        row_heights=row_heights,
        specs=[[{"secondary_y": True}]] * rows
    )
    
    fig.update_layout(
        width=plt_shape.get('plt_width', 2400),
        height=plt_shape.get('plt_height', 1300),
        # margin=dict(l=50, r=50, t=80, b=50),
        margin=dict(t=30, b=30, pad=0),
        autosize=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    return fig

def add_volume(fig, volume_ser, row):
    colors = ['#4B8BBE' if close >= open_ else '#B03060' 
             for open_, close in zip(volume_ser['Open'], volume_ser['Close'])]
    
    fig.add_trace(
        go.Bar(
            x=volume_ser.index,
            y=volume_ser['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=row, col=1
    )
    return fig

def add_lines(fig, data, row, **kwargs):
    if isinstance(data, pd.Series):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.values,
                name=data.name or 'value',
                # line=dict(width=1.5),
                **kwargs
            ),
            row=row, col=1
        )
    else:
        for col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[col],
                    name=col,
                    # line=dict(width=1.5),
                    **kwargs
                ),
                row=row, col=1
            )
    return fig

def add_ma(fig, data, periods, row):
    for period in periods:
        ma = data.rolling(period).mean()
        fig.add_trace(
            go.Scatter(
                x=ma.index,
                y=ma.values,
                name=f'MA{period}',
                line=dict(width=2, dash='dot')
            ),
            row=row, col=1
        )
    return fig

def make_line_plotly(lpic: Union[pd.Series, pd.DataFrame],
                     beg: str, end: str,
                     plt_shape: dict = dict(),
                     plt_title_opts: Optional[dict] = None,
                     plt_volume: Union[pd.Series, bool, None] = False,
                     plt_add_ma: Union[List, Tuple, None] = None,
                     plt_add_lines: Union[pd.Series, pd.DataFrame, None] = None,
                     other_tbs: Union[List, Tuple, None] = None):
    ''' 输出使用plotly绘制的line图形
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
    if not isinstance(lpic.index[0], str):
        lpic.index = lpic.index.map(lambda x:x.strftime('%Y-%m-%d'))
    data = lpic[beg:end].round(3)
    rows = 1
    row_heights = [1.0]
    
    if isinstance(plt_volume, pd.Series):
        rows += 1 
    if other_tbs:
        rows += len(other_tbs)
        row_heights = get_grid_hts(len(other_tbs), isinstance(plt_volume, pd.Series))
    
    fig = create_plotly_figure(rows, row_heights, plt_shape)
    current_row = 1
    # 去除日期中的空隙
    beg_, end_ = data.index.min(), data.index.max()
    all_days = set(x.strftime(r'%Y-%m-%d') for x in pd.date_range(start=beg_,end=end_,freq='D'))
    rm_days = sorted(list(all_days - set(data.index)))
    fig.update_xaxes(
        # rangeslider_visible=False,
        # rangeselector_visible=False,
        tickformat=r'%m-%d',
        rangebreaks=[dict(values=rm_days)],  # hide holidays (Christmas and New Year's, etc)
        type='date'
    )
    # Main price plot
    fig = add_lines(fig, data, current_row)
    
    # Add MA lines
    if plt_add_ma and isinstance(lpic, pd.Series):
        fig = add_ma(fig, lpic[beg:end], plt_add_ma, current_row)
    
    # Add additional lines
    if plt_add_lines is not None:
        fig = add_lines(fig, plt_add_lines[beg:end], current_row, line=dict(width=3, dash='dash'))
    
    # Add volume
    if isinstance(plt_volume, pd.Series):
        current_row += 1
        add_volume(fig, plt_volume[beg:end], current_row)
    
    # Add other tabs
    if other_tbs:
        for _i, tb in enumerate(other_tbs):
            current_row += 1
            chart_type, tb = next(iter(tb.items()))
            if isinstance(tb, pd.Series):
                tb_data = pd.DataFrame(tb[beg:end])
                if tb.name is None: tb_data.columns = [f'val{_i+1}']
            elif isinstance(tb, pd.DataFrame):
                tb_data = tb.loc[beg:end]
            chart_type = parse_other_tb_name(chart_type)
            
            if chart_type == 1:
                fig = add_lines(fig, tb_data[beg:end], current_row)
            elif chart_type == 2:
                fig.add_trace(
                    go.Bar(
                        x=tb_data[beg:end].index,
                        y=tb_data[beg:end].values.flatten(),
                        name=tb_data.name if isinstance(tb_data, pd.Series) else 'Bar'
                    ),
                    row=current_row, col=1
                )
    
    fig.update_layout(
        title=plt_title_opts.get('title', '') if plt_title_opts else f"{beg} ~ {end}",
        xaxis_rangeslider_visible=False,
        template='seaborn'
    )
    fig.show()

def make_candle_plotly(ohlc: pd.DataFrame,
                       beg: str, end: str,
                       ohlc_names: Union[List, Tuple, None] = None,
                       plt_shape: dict = dict(),
                       plt_title_opts: Optional[dict] = None,
                       plt_volume: bool = True,
                       plt_add_ma: Union[List, Tuple, None] = None,
                       plt_add_lines: Union[List, Tuple, None] = None,
                       plt_add_points: Union[List, Tuple, None] = None,
                       other_tbs: Union[List, Tuple, None] = None):
    ''' 输出使用plotly绘制的kline图形
        @param ohlc,            日线表格, 题头至少需要有ohlcv五项
        @param beg,end          起始日期-结束日期
        @param ohlc_names       用于转换ohlc表格的题头
        @param plt_shape        kline图形的整体大小
        @param plt_title_opts   kline标题设置
        @param plt_add_ma       在kline中添加均线的参数
        @param plt_add_lines    在kline中添加别的线
        @param other_tbs        添加其余图形
    '''
    # TODO 多个窗口比例尺推测，多tab输出
    ohlc_tb = ohlc.copy()
    std_col_names = ['Open','Close','Low', 'High', 'Volume']
    if ohlc_names is not None:
        trans_d = dict(o='Open',c='Close',h='High',l='Low',v='Volume')
        name_trans = {x:trans_d[x.lower()[0]] for x in ohlc_names if x.lower()[0] in trans_d.keys()}
        ohlc_tb.rename(columns=name_trans,inplace=True)
    if plt_add_lines:
        assert all(x in ohlc_tb.columns for x in plt_add_lines)
    if other_tbs:
        assert all(parse_other_tb_name(j)>0 for k in other_tbs for j in k.keys())
    rows = 1
    row_heights = [1.0]
    
    if plt_volume:
        rows += 1
    if other_tbs:
        rows += len(other_tbs)
        row_heights = get_grid_hts(len(other_tbs), plt_volume)
    
    fig = create_plotly_figure(rows, row_heights, plt_shape)
    current_row = 1
    
    data = ohlc_tb.loc[beg:end,std_col_names]
    # 去除日期中的空隙
    beg_, end_ = data.index.min(), data.index.max()
    all_days = set(x for x in pd.date_range(start=beg_,end=end_,freq='D'))
    rm_days = sorted(list(all_days - set(data.index)))
    fig.update_xaxes(
        # rangeslider_visible=False,
        # rangeselector_visible=False,
        tickformat=r'%m-%d',
        rangebreaks=[dict(values=rm_days)],  # hide holidays (Christmas and New Year's, etc)
        type='date'
    )
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=''
        ),
        row=current_row, col=1
    )
    
    # Add MA lines
    if plt_add_ma:
        fig = add_ma(fig, data['Close'], plt_add_ma, current_row)
    
    # Add additional lines
    if plt_add_lines:
        fig = add_lines(fig, plt_add_lines[beg:end], current_row, line=dict(width=2, dash='dash'))
        # for line in plt_add_lines:
        #     fig.add_trace(
        #         go.Scatter(
        #             x=data.index,
        #             y=data[line],
        #             name=line,
        #             line=dict(width=2)
        #         ),
        #         row=current_row, col=1
        #     )
    
    # Add points
    if plt_add_points:
        markers = []
        for x, y in plt_add_points:
            if x in data.index:
                markers.append(
                    go.Scatter(
                        x=[x],
                        y=[data.loc[x, 'Low']],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down' if y == 1 else 'square' if y == 2 else 'circle',
                            size=12,
                            color='#FF0000' if y == 1 else '#00FF00' if y == 2 else '#0000FF'
                        ),
                        name=f'Marker {y}'
                    )
                )
        for marker in markers:
            fig.add_trace(marker, row=current_row, col=1)
    
    # Add volume
    if plt_volume:
        current_row += 1
        add_volume(fig, data, current_row)
    
    # Add other tabs
    if other_tbs:
        for tb in other_tbs:
            current_row += 1
            chart_type, tb_names = next(iter(tb.items()))
            if isinstance(tb_names, (list, tuple)):
                tb_data = ohlc.loc[beg:end,list(tb_names)]
            elif isinstance(tb_names, (str, int, float)):
                tb_data = ohlc.loc[beg:end, [tb_names]]
            chart_type = parse_other_tb_name(chart_type)
            
            if chart_type == 1:
                fig = add_lines(fig, tb_data, current_row)
            elif chart_type == 2:
                fig.add_trace(
                    go.Bar(
                        x=tb_data.index,
                        y=tb_data.values.flatten(),
                        name=tb_data.name if isinstance(tb_data, pd.Series) else 'Bar'
                    ),
                    row=current_row, col=1
                )
    
    fig.update_layout(
        title=plt_title_opts.get('title', '') if plt_title_opts else f"{beg} ~ {end}",
        showlegend=False,
        template='plotly_white'
    )
    
    fig.show()


def chart_test():
    xx = np.random.lognormal(5.44,0.4426,400)
    xx = np.array([np.random.normal(x,x*0.05,10) for x in xx])
    xx = pd.DataFrame(dict(o=xx[:,0],c=xx[:,-1],l=xx.min(axis=1),h=xx.max(axis=1),v=xx[:,4]*10,qq=xx[:,5]*10))
    xx.index = pd.date_range(start='2022-01-10',periods=len(xx),freq='B')
    # make_candle_plotly(xx, '2022-05-31', '2023-05-31',
    #                    ohlc_names=('o','h','l','c','v'), plt_add_ma=(10,20,60),
    #                    plt_volume=True, other_tbs=[{'l':['v']}])
    make_line_plotly(xx.c, '2022-05-31', '2023-05-31',
                     plt_add_ma=(10,20,60),
                     plt_add_lines=xx.l,
                     other_tbs=[{'line':xx[['v','qq']]},{'bar':xx.v}])

if __name__=='__main__':
    chart_test()
    pass