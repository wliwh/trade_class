"""
绘制递归图
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from index_get.get_index_value import other_index_getter, global_index_indicator
from common.smooth_tool import drawdown_series

Search_Index = {
    '道琼斯':'道琼斯',
    '标普500':'标普500',
    '纳斯达克':'NDX',
    '纳指100':'NDX100'
}

def _find_ma_breakthrough_points(df: pd.DataFrame, ma_period: int, min_days_below: int = 5, 
                                window_days: int = None) -> pd.DataFrame:
    """
    查找指数跌破均线的关键点位
    
    参数:
        df: 包含 'close' 和 'low' 列的DataFrame，index为日期
        ma_period: 均线周期（如60，240）
        min_days_below: 连续低于均线的最小天数
        window_days: 前后寻找最低点的窗口天数
    
    返回:
        包含突破点信息的DataFrame
    """
    if window_days is None:
        window_days = ma_period // 4  # 默认窗口为均线周期的1/4
    code_name_ = df['code'][0]
    zh_name_ =df['name_zh'][0]
        
    # 计算移动平均线
    df['ma'] = df['close'].rolling(window=ma_period).mean()
    
    # 计算是否低于均线
    df['below_ma'] = df['close'] < df['ma']
    
    # 找出连续低于均线的区间
    below_periods = []
    current_period = []
    
    for date, row in df.iterrows():
        if row['below_ma']:
            current_period.append(date)
        else:
            if len(current_period) >= min_days_below:
                below_periods.append(current_period)
            current_period = []
    
    if len(current_period) >= min_days_below:
        below_periods.append(current_period)
    
    # print(below_periods)
        
    # 对每个区间找出最低点
    breakthrough_points = []
    for period in below_periods:
        start_idx = max(0, df.index.get_loc(period[0]) - window_days)
        end_idx = min(len(df), df.index.get_loc(period[-1]) + window_days + 1)
        high_start_idx = max(0, start_idx - window_days)
        
        window_data = df.iloc[start_idx:end_idx]
        min_low_idx = window_data['low'].idxmin()
        # print(high_start_idx, type(high_start_idx), period[0], type(period[0]))
        max_high_idx = df.iloc[high_start_idx:df.index.get_loc(period[0])]['high'].idxmax()
        
        # 重复元素不要添加进去
        if period[0] <= min_low_idx <= period[-1]:
            highV =  float(df.loc[max_high_idx, 'high'])
            lowV = float(df.loc[min_low_idx, 'low'])
            crossV = float(df.loc[period[0], 'ma'])
            ratio_HL = (crossV-lowV)/(highV-crossV)
            tov_bl = 3 if ratio_HL > 1.45 else 2
            breakthrough_points.append({
                'name_zh': zh_name_,
                'code': code_name_,
                'cross': ma_period,
                'start_date': period[0],
                'start_price': crossV,
                'end_date': period[-1],
                'highest_date': max_high_idx,
                'highest_price': highV,
                'lowest_date': min_low_idx,
                'lowest_price': lowV,
                'high_date':  max_high_idx,
                'high_value': highV,
                'cross_date': period[0],
                'cross_ma': crossV,
                'low_date': min_low_idx,
                'low_value': lowV,
                'pct1': round(100 * (1 - lowV / highV), 2),
                'pct2': 0.0,
                # 'pct2': round(100 * (1 - close_hl[1] / close_hl[0]), 2),
                'minValue': round(highV - lowV, 2),
                'ratio_int': tov_bl,
                'ratio': round(ratio_HL,2),
                'tovalue': [round(c,2) for c in (tov_bl*1.1*crossV-(tov_bl*1.1-1)*highV,\
                                tov_bl*crossV-(tov_bl-1)*highV, tov_bl*0.9*crossV-(tov_bl*0.9-1)*highV)]
            })
        
    return pd.DataFrame(breakthrough_points)

def _analyze_index_ma_points(index_df: pd.DataFrame, start_date: str = None) -> dict:
    """
    分析指数的均线突破点位
    
    参数:
        index_df: 包含指数数据的DataFrame，需要包含'close'和'low'列
        start_date: 开始分析的日期，如果为None则使用全部数据
    """
    if start_date:
        index_df = index_df[index_df.index >= pd.to_datetime(start_date)]
        
    # 确保数据足够长
    if len(index_df) < 240:
        raise ValueError("数据长度不足240个交易日，无法进行240日均线分析")

    # 240日均线分析
    ma240_points = _find_ma_breakthrough_points(
        index_df, 
        ma_period=240,
        min_days_below=5,
        window_days=120  # 前后半年
    )
    
    # 60日均线分析
    ma60_points = _find_ma_breakthrough_points(
        index_df, 
        ma_period=60, 
        min_days_below=5,
        window_days=30  # 前后1.5个月
    )

    # 过滤掉ma60中与ma240重叠的突破点
    ma240_dates = set(ma240_points['lowest_date'])
    ma60_filtered = ma60_points[~ma60_points['lowest_date'].isin(ma240_dates)]
    
    return {
        'ma240_points': ma240_points,
        'ma60_points': ma60_filtered
    }

def _format_breakthrough_points(points_df: pd.DataFrame) -> str:
    """格式化突破点信息"""
    if points_df.empty:
        return "未发现符合条件的突破点"
        
    result = []
    for _, row in points_df.iterrows():
        info = (f"名称： {row['name_zh']} {row['code']}\n"
                f"突破区间: {row['start_date'].strftime('%Y-%m-%d')} 到 "
                f"{row['end_date'].strftime('%Y-%m-%d')}\n"
                f"前高处日期: {row['highest_date'].strftime('%Y-%m-%d')}\n"
                f"最高价: {row['highest_price']:.2f}\n"
                f"起点处均线: {row['start_price']:.2f}\n"
                f"最低点日期: {row['lowest_date'].strftime('%Y-%m-%d')}\n"
                f"最低价: {row['lowest_price']:.2f}\n"
                f"差异: {(row['start_price']-row['lowest_price'])/(row['highest_price']-row['start_price']):.2f}\n")
                # f"当时均线值: {row['ma_value']:.2f}\n"
                # f"偏离均线: {((row['lowest_price'] / row['ma_value'] - 1) * 100):.2f}%\n")
        result.append(info)
        
    return "\n".join(result)

def _show_cross_ma(choose_code:str='IXIC', begin_date:str='20160101'):
    # 获取指数数据
    if Search_Index.get(choose_code):
        choose_code = Search_Index.get(choose_code)
    else:
        choose_code = 'IXIC'
    index_getter = other_index_getter(choose_code,begin_date,'20200101')
    index_getter.index = pd.to_datetime(index_getter.index)
    
    # 分析最近一年的数据
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=8)).strftime('%Y-%m-%d')
    results = _analyze_index_ma_points(index_getter, start_date)
    
    print(f"=== 指数{choose_code}均线分析结果 ===")
    print("\n--- 240日均线突破点 ---")
    print(_format_breakthrough_points(results['ma240_points']))
    print("\n--- 60日均线突破点 ---")
    print(_format_breakthrough_points(results['ma60_points'])) 


# 短周期分析与绘制
def detect_cycle_lows(df, price_col='price', window_size=60, cycle_range=(35,54)):
    """
    分析日频金融数据，识别周期性低点
    参数：
    df - 数据文件路径（CSV格式，含date和price列）
    window_size - 局部低点检测窗口（交易日天数，默认600天≈2年）
    cycle_range - 目标周期范围（月数，默认33-49个月）
    """
    # 数据加载与预处理
    df['date'] = pd.to_datetime(df.index)
    # df.index = range(len(df))
    # df.index.name = 'idx'
    # df.sort_values('date', inplace=True)
    if isinstance(price_col, str):
        col1 = col2 = price_col
    else:
        col1, col2 = price_col

    # 寻找局部低点
    min_idx = argrelextrema(df[col1].values, np.less, order=window_size)[0]
    low_points = df.iloc[min_idx][['date', col1]].reset_index(drop=True)
    low_points['diff'] = (low_points['date']-df['date'].min()).dt.days

    # 找出第一个周期起点
    right_date = low_points.loc[low_points['diff']<=int(cycle_range[1]*30.4),col1].idxmin()
    first_cycle = low_points.iloc[right_date]
    low_points['diff'] = low_points['diff'] - first_cycle['diff']

    # 找出所有周期低点
    cycle_lows = [{'date':first_cycle['date'],'price':first_cycle[col1],'diff':0,'type':'L'}]
    while True:
        wks = low_points[(low_points['diff']>=int(cycle_range[0]*30.4)) & (low_points['diff']<=int(cycle_range[1]*30.4))]
        if wks.empty: break
        wks['diff'] = wks['diff'] / 30.4 / 42 -1
        wks['rank'] = 5*(wks[col1]/wks[col1].min()-1) + wks['diff'].apply(lambda x: x*2.5 if x>0 else -x)
        wks['rank'] = wks['rank'].apply(lambda x: 0 if x<0 else x)
        now_cycle = low_points.iloc[wks['rank'].idxmin()]
        cycle_lows.append({'date':now_cycle['date'],'price':now_cycle[col1],'diff':now_cycle['diff']/30.4,'type':'L'})
        low_points['diff'] = low_points['diff'] - now_cycle['diff']

    # 找出周期内的高点
    idx = df.loc[df['date'] <= cycle_lows[0]['date'],col2].idxmax()
    cycle_highs = [{'date':df.loc[idx,'date'],'price':df.loc[idx,col2],'diff':(df.loc[idx,'date']-cycle_lows[0]['date']).days/30.4,'type':'H'}]
    for t in range(len(cycle_lows)-1):
        left_date = cycle_lows[t]['date']
        right_date = cycle_lows[t+1]['date']
        idx = df.loc[(df['date']>=left_date) & (df['date']<=right_date),col2].idxmax()
        cycle_highs.append({'date':df.loc[idx,'date'],'price':df.loc[idx,col2],'diff':(df.loc[idx,'date']-left_date).days/30.4,'type':'H'})
    idx = df.loc[df['date'] >= cycle_highs[-1]['date'],col2].idxmax()
    cycle_highs.append({'date':df.loc[idx,'date'],'price':df.loc[idx,col2],'diff':(df.loc[idx,'date']-right_date).days/30.4,'type':'H'})

    # 依次寻找每个周期低点
    # end_date = df['date'].max()
    # now_date = first_cycle['date']
    # cycle_lows = [now_date]
    # while now_date < end_date:
    #     left_date = now_date+pd.Timedelta(days=int(cycle_range[0]*30.4))
    #     right_date = now_date+pd.Timedelta(days=int(cycle_range[1]*30.4))
    #     if left_date > end_date:
    #         break
    #     now_idx = df.loc[(df['date']>=left_date) & (df['date']<=right_date),col1].idxmin()
    #     now_date = df.iloc[now_idx]['date']
    #     cycle_lows.append(now_date)
    print(cycle_lows)
    cycles = pd.DataFrame(cycle_lows + cycle_highs)
    cycles.sort_values('date', inplace=True)
    cycles.reset_index(drop=True, inplace=True)
    return cycles

def plot_index_cycles(df, cycles, df_name='price'):
    ''' 对指数的周期进行绘制, 搭配 detect_cycle_lows 函数  '''
    # 采用上下布局
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10))
    # 添加对数同比
    df['log'] = np.log(df[df_name])/(np.log(df[df_name]).shift(250))
    df['log'].fillna(1, inplace=True)
    cycles_low = cycles[cycles['type']=='L']
    cycles_high = cycles[cycles['type']=='H']
    ax1.plot(df['date'], df[df_name], label='Price', alpha=0.3)
    # plt.plot(df['date'], df['smoothed'], label='平滑曲线', color='orange')
    ax1.scatter(cycles_low['date'], cycles_low['price'], 
                color='red', zorder=5, label='Cycle Low')
    ax1.scatter(cycles_high['date'], cycles_high['price'], 
                color='green', zorder=5, label='Cycle High')
    ax2.plot(df['date'], df['log'], label='Log Ratio', alpha=0.7)
    for _, row in cycles.iterrows():
        ax2.scatter(row['date'], df.loc[row['date'] == df['date'], 'log'], 
                color='red' if row['type']=='L' else 'green', zorder=5)

    # for i, row in cycles_low.iterrows():
    #     plt.annotate(f'Cyc{(i+1)//2}: {row["date"].strftime("%Y-%m")}',
    #                  (row['date'], row['price']),
    #                  textcoords="offset points",
    #                  xytext=(0,-14),
    #                  ha='left')
    
    plt.title('Cycle Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def _test_index_cycles1():
    df = other_index_getter(Search_Index['标普500'],'1990-01-01','2008-01-01')
    # df['date'] = pd.to_datetime(df.index)
    cycles = detect_cycle_lows(df, price_col='close')
    print(cycles)
    plot_index_cycles(df, cycles, df_name='close')


def find_break_ma_range(df:pd.DataFrame, ma_period:int, window_days: int = None, min_days:int=5):
    ''' 分析价格击破均线前后的关键信息，包括前期高点、均线下方的低点、对称性等 '''
    code_name, zh_name = df.loc[0, ['code', 'name_zh']]
    if window_days==None:
        window_days = ma_period // 2  # 默认窗口为均线周期的1/2

    def set_dict(cros_idx, idx, ma_d, cnm, zh_name):
        if cros_idx<=0: return False
        # print(cros_idx, idx)
        min_low_idx = df.iloc[cros_idx:idx]['low'].idxmin()
        high_idx = df.iloc[max(cros_idx-window_days, 0):cros_idx+1]['high'].idxmax()
        highV =  float(df.loc[high_idx, 'high'])
        lowV = float(df.loc[min_low_idx, 'low'])
        crossV = float(df.loc[cros_idx, f'ma{ma_d}'])
        ratio_HL = (crossV-lowV)/(highV-crossV)
        tov_bl = 2 if ratio_HL > 1.45 else 1
        end_idx = min(idx+5, len(df)-1)
        return dict(
                name_zh = zh_name,
                code = cnm,
                down_day = idx-cros_idx,
                cross=ma_d,
                high_date=df.loc[high_idx, 'date'],
                high_weeks = [df.loc[high_idx, 'day_of_week'],
                              int(df.loc[high_idx, 'day_of_trade_week']),
                              int(df.loc[high_idx, 'total_trading_days'])],
                high_value=highV,
                cross_date=df.loc[cros_idx, 'date'],
                cross_ma=round(crossV,2),
                low_date=df.loc[min_low_idx,'date'],
                low_weeks = [df.loc[min_low_idx, 'day_of_week'],
                             int(df.loc[min_low_idx, 'day_of_trade_week']),
                             int(df.loc[min_low_idx, 'total_trading_days'])],
                low_value=lowV,
                end_date = df.loc[end_idx,'date'],
                pct1 = round(100*(1-lowV/highV),2),
                minvalue = round(highV-crossV,2),
                ratio = round(ratio_HL,2),
                ratio_int = tov_bl,
                ratio_sim = ratio_HL/tov_bl if ratio_HL>tov_bl else tov_bl/ratio_HL,
                tovalue = [round(c,2) for c in ((tov_bl+1)*1.1*crossV-(tov_bl*1.1+0.1)*highV,\
                                (tov_bl+1)*crossV-(tov_bl)*highV, (tov_bl+1)*0.9*crossV-(tov_bl*0.9-0.1)*highV)])
    
    # 初始化结果列表
    result = []
    old_idx = [0]
    cross_idx = None
    in_sequence = False
    
    # 遍历每一行
    for index, row in df.iterrows():
        if row[f'ld{ma_period}'] >= min_days:
            if not in_sequence:
                # 开始一个新的序列
                cross_idx = index-min_days+1
                in_sequence = True
        else:
            if in_sequence:
                # 结束当前序列
                dic = set_dict(cross_idx, index, ma_period, code_name, zh_name)
                if dic:
                    result.append(dic)
                    old_idx.extend([cross_idx, index])
                in_sequence = False
    
    # print(pd.concat([df.loc[old_idx[::2],'date'], df.loc[old_idx[1::2],'date']], axis=1, ignore_index=True))
    # 处理最后一个序列
    if in_sequence:
        end_idx = df.index[-1]
        result.append(set_dict(cross_idx, end_idx, ma_period, code_name, zh_name))
    
    df = pd.DataFrame(result)
    df.sort_values(by=['high_date','low_value'],inplace=True)
    df.drop_duplicates(subset=['high_date'],keep='first',inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _test_find_break1():
    p1 = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data_save/global_index.csv'))
    p1 = p1[(p1.code=='NDX') & (p1.date>'2020-01-01')]
    p1.reset_index(drop=True, inplace=True)
    # print(p1.tail())
    print(find_break_ma_range(p1, 60, 30, 5))


def find_all_breaks(pn:pd.DataFrame) -> dict:
    """
    分析价格序列，找出跌破250日、120日和60日均线的关键点位
    
    参数:
        pn: 包含 'close' 和 'low' 列的DataFrame，index为日期
        
    返回:
        表格信息
    """
    ma250_break = find_break_ma_range(pn, 250, 120, 5)
    ma250_break = ma250_break[ma250_break['pct1']>15]
    ma120_break = find_break_ma_range(pn, 120, 85, 5)
    ma60_break = find_break_ma_range(pn, 60, 25, 5)

    # ma250_low_dates = set(ma250_break['low_date'])
    # ma120_low_dates = set(ma120_break['low_date'])
    # ma120_filtered = ma120_break[~ma120_break['low_date'].isin(ma250_low_dates)]
    # ma60_filtered = ma60_break[~ma60_break['low_date'].isin(ma250_low_dates)]
    # ma60_filtered = ma60_filtered[~ma60_filtered['low_date'].isin(ma120_low_dates)]

    p1 = pd.concat([ma250_break, ma120_break, ma60_break])
    p1.sort_values(by=['high_date','ratio_sim'],inplace=True)
    group_c = {group: i + 1 for i, group in enumerate(p1['high_date'].unique())}
    p1['group_cnt'] = p1.groupby('high_date').cumcount() + 1
    p1 = p1[(p1['group_cnt']==1) & (p1['ratio']<3) & (p1['ratio']>0.5)]
    # p1.sort_values(by=['low_date','ratio_sim'],inplace=True)
    return p1

def _test_all_breaks():
    """
    测试find_all_breaks函数
    """
    glb = global_index_indicator()
    conf = glb.get_cator_conf()
    fpath = conf['fpath']
    p1 = pd.read_csv(fpath)
    pn = p1[(p1.code=='NDX') & (p1.date>'2020-01-01')]
    pn.reset_index(drop=True, inplace=True)
    print(find_all_breaks(pn))

def plot_candlestick_with_lines(df: pd.DataFrame, line_tuple: tuple, cross_ma: int, rt_int: int=1):
    """
    使用plotly绘制图，并标注水平线
    
    参数:
        df: DataFrame，包含OHLC数据
        line_tuple: 字典，格式为 {日期: 价格水平}
    """

    # 去除日期中的空隙
    beg_, end_ = df['date'].min(), df['date'].max()
    all_days = set(x.strftime('%Y-%m-%d') for x in pd.date_range(start=beg_,end=end_,freq='D'))
    rm_days = all_days - set(df['date'])
    
    # 创建K线图对象
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                                        open=df['open'],
                                        high=df['high'],
                                        low=df['low'],
                                        close=df['close'],
                                        name='',
                                        increasing_line_color='#FF4136',
                                        decreasing_line_color='#3D9970',
                                        increasing_fillcolor='#FF9F9A',
                                        decreasing_fillcolor='#9DCCB7'
                                        )])
    fig.update_xaxes(
        rangeslider_visible=False,
        # rangeselector_visible=False,
        tickformat='%m-%d',
        rangebreaks=[
            dict(values=list(rm_days))  # hide holidays (Christmas and New Year's, etc)
        ],
        type='date'
    )
    fig.update_yaxes(
        # autorange=True,
        tickformat='d',
        separatethousands=False,            # 启用千位分隔符
        nticks=5                            # 控制刻度数量（可选）
    )
    # print(df['ld250'].max(), beg_, end_, cross_ma)
    ma_colors = ('rgb(89,181,93)','rgb(81,136,244)','rgb(209,81,166)')
    line_colors = ('rgba(218,64,100,0.8)','rgba(120,123,133,0.8)','rgba(62,192,128,0.8)', 'rgba(74,185,210,0.8)', 'rgba(255,128,0,0.6)')
    # 添加矩形框
    for j in range(2):
        _color = line_colors[j+2] if rt_int>1 and j==1 else line_colors[j+1]
        fig.add_shape(
            type="rect",
            x0=line_tuple[j][0], x1=df.iloc[-1]['date'],
            y0=line_tuple[j][1], y1=line_tuple[j+1][1],
            xref="x", yref="y",
            fillcolor=_color.replace('0.8','0.15'),
            line=dict(color="rgba(0,0,0,0)")  # 透明边框
        )
    # 添加MA
    if cross_ma < 180:
        fig.add_trace(go.Scatter(x=df['date'],y=df[f'ma60'],line={'color':ma_colors[0]},name=f'ma60'))
        if cross_ma>90:
            fig.add_trace(go.Scatter(x=df['date'],y=df[f'ma{cross_ma}'],line={'color':ma_colors[1]},name=f'ma{cross_ma}'))
        if df['ld250'].max()>0:
            fig.add_trace(go.Scatter(x=df['date'],y=df[f'ma250'],line={'color':ma_colors[2]},name=f'ma250'))
    else:
        fig.add_trace(go.Scatter(x=df['date'],y=df[f'ma120'],line={'color':ma_colors[1]},name=f'ma120'))
        fig.add_trace(go.Scatter(x=df['date'],y=df[f'ma{cross_ma}'],line={'color':ma_colors[2]},name=f'ma{cross_ma}'))
    # 添加水平线
    for j,(date, price) in enumerate(line_tuple):
        if j<3:
            _color = line_colors[3] if rt_int>1 and j==2 else line_colors[j]
            _dashed = 'solid'
        else:
            _color = line_colors[4]
            _dashed = 'longdash' if j ==4 else 'dot'
        fig.add_shape(
            type='line',
            x0=date,
            y0=price,
            x1=df.iloc[-1]['date'],  # 延伸到图表最右端
            y1=price,
            line=dict(
                color=_color,
                width=4,
                dash=_dashed
            )
        )
        # 添加标注
        fig.add_annotation(
            x=date if j<3 else df.iloc[-3]['date'],
            y=price,
            text=f'{price:.2f}',
            showarrow=True if j<3 else False,
            arrowhead=2,
            ax=-40,
            ay=-40
        )
    
    # 更新布局
    fig.update_layout(
        # title='title',
        # yaxis_title='价格',
        # xaxis_title='日期',
        margin=dict(t=30, b=30, pad=0),
        autosize=True,
        showlegend=False,
        template='plotly_white'
    )
    # 显示图表
    # fig.show()
    return fig


def plot_cand_test(choose_n:int=1):
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
    pn = p1[(p1.code=='NDX') & (p1.date>'2020-01-01')]
    pn.reset_index(drop=True, inplace=True)
    warn_info = find_all_breaks(pn).iloc[choose_n].to_dict()
    h_d, e_d = pd.to_datetime(warn_info['high_date']), pd.to_datetime(warn_info['end_date'])
    if (e_d - h_d).days<50:
        beg_day = (e_d-pd.offsets.Day(55)).strftime('%Y-%m-%d')
    else:
        beg_day = (h_d-pd.offsets.Day(10)).strftime('%Y-%m-%d')
    pn = pn[(pn['date']>beg_day) & (pn['date']<=warn_info['end_date'])]
    days_dict = [warn_info['high_value'],warn_info['cross_ma'], warn_info['low_value']]+warn_info['tovalue']
    days_dict = [(warn_info['high_date'],d) for d in days_dict]
    plot_candlestick_with_lines(pn, days_dict, warn_info['cross'], warn_info['ratio_int'])
    # high_day = warn_info['high_date']
    # if (pd.to_datetime(conf['max_date_idx']) - pd.to_datetime(high_day)).days<50:
    #     beg_day = p1.index[-55]
    # else:
    #     beg_day = (pd.to_datetime(high_day)-pd.offsets.Day(10)).strftime('%Y-%m-%d')
    # p1 = p1[p1.index>beg_day]
    # days_dict = [warn_info['high_value'],warn_info['cross_ma'], warn_info['low_value']]+warn_info['tovalue']
    # days_dict = ((high_day,d) for d in days_dict)
    # plot_candlestick_with_lines(p1, days_dict, warn_info['cross'])


if __name__ == '__main__':
    # plot_cand_test(1)
    # _test_find_break1()
    print(plot_cand_test(0))
    pass