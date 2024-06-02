import pandas as pd
import numpy as np
import talib as ta
from quantstats.utils import _prepare_prices
from quantstats.stats import to_drawdown_series, remove_outliers
from functools import partial
from typing import Union, Optional, List, Callable


def SP_SMOTH(ser: pd.Series, lens:int=20) -> np.ndarray:
    """ 平滑工具. 二阶低通滤波器 """
    if not isinstance(ser, pd.Series):
        raise ValueError('ser must be pd.Series.')
    gst = ser.to_list()
    a1 = np.exp(-1.414*np.pi*2/lens)
    b1 = 2*a1*np.cos(1.414*180*2/lens)
    c3 = -a1*a1
    c1 = 1-b1-c3
    filt = gst[:2]+[0 for _ in range(2, len(gst))]
    for i in range(2, len(gst)):
        filt[i] = c1/2*(gst[i]+gst[i-1])+b1*filt[i-1]+c3*filt[i-2]
    return np.array(filt)


def RE_FLEX(gst:pd.Series, lens:int=20) -> np.ndarray:
    """ reflex curve """
    filt = SP_SMOTH(gst, lens)
    slope = [0 for _ in range(len(gst))]
    sums = slope.copy()
    Ms = slope.copy()
    reflex = slope.copy()
    for i in range(20, len(gst)):
        slope[i] = (filt[i-lens]-filt[i])/lens
        sums[i] = filt[i] + (lens+1)/2*slope[i] - sum(filt[i-lens:i])/lens
        Ms[i] = 0.04*sums[i]*sums[i]+0.96*Ms[i-1]
        if Ms[i] != 0:
            reflex[i] = sums[i]/np.sqrt(Ms[i])
    return np.array(reflex)


def TREND_FLEX(gst: pd.Series, lens:int=20) -> np.ndarray:
    """ trendflex curve """
    filt = SP_SMOTH(gst, lens)
    sums = [0 for _ in range(len(gst))]
    Ms = sums.copy()
    trendflex = sums.copy()
    for i in range(20, len(gst)):
        sums[i] = filt[i]-sum(filt[i-lens:i])/lens
        Ms[i] = 0.04*sums[i]*sums[i]+0.96*Ms[i-1]
        if Ms[i] != 0:
            trendflex[i] = sums[i]/np.sqrt(Ms[i])
    return np.array(trendflex)


def ema_tan(gst: pd.Series, short: int, longd: int):
    """ 计算 ema-tan
        短期EMA与长期EMA的比值. 用于衡量上涨或下降的力度 """
    if not isinstance(gst, pd.Series):
        raise ValueError('gst must be pd.Series.')
    return ta.EMA(gst, short)/ta.EMA(gst, longd)-1


def LLT_MA(price: pd.Series, alpha: float) -> pd.Series:
    """ 计算低延迟趋势线

    Args:
        price (pd.Series): 价格数据. index-date values
        alpha (float): 窗口期的倒数.比如想要窗口期为5,则为1/5

    Raises:
        ValueError: 必须为pd.Series

    Returns:
        pd.Series: index-date values
    """
    if not isinstance(price, pd.Series):
        raise ValueError('price必须为pd.Series')

    llt_ser: pd.Series = pd.Series(index=price.index,dtype='float64')
    llt_ser.iloc[:2] = price.iloc[:2]

    for i, e in enumerate(price.values):
        if i > 1:
            v = (alpha - alpha**2 * 0.25) * e + (alpha ** 2 * 0.5) * price.iloc[i - 1] - (alpha - 3 * (
                alpha**2) * 0.25) * price.iloc[i - 2] + 2 * (1 - alpha) * llt_ser.iloc[i - 1] - (1 - alpha)**2 * llt_ser.iloc[i - 2]
            llt_ser.iloc[i] = v

    return llt_ser

def HMA(price: pd.Series, window: int) -> pd.Series:
    """HMA均线

    Args:
        price (pd.Series): 价格数据. index-date values
        window (int): 计算窗口

    Raises:
        ValueError: 必须为pd.Series

    Returns:
        pd.Series: index-date values
    """
    if not isinstance(price, pd.Series):

        raise ValueError('price必须为pd.Series')

    return ta.WMA(2 * ta.WMA(price, int(window * 0.5)) - ta.WMA(price, window),int(np.sqrt(window)))


def FRAMA(se, periods, clip=True):
    ''' 计算FRAMA均线 '''
    T = int(periods/2)
    df = se.copy()

    # 1.用窗口 W1 内的最高价和最低价计算 N1 = (最高价 – 最低价) / T
    N1 = (df.rolling(T).max()-df.rolling(T).min())/T

    # 2.用窗口 W2 内的最高价和最低价计算 N2 = (最高价 – 最低价) / T
    n2_df = df.shift(T)
    N2 = (n2_df.rolling(T).max()-n2_df.rolling(T).min())/T

    # 3.用窗口 T 内的最高价和最低价计算 N3 = (最高价 – 最低价) / (2T)
    N3 = (df.rolling(periods).max() -
          df.rolling(periods).min())/periods

    # 4.计算分形维数 D = [log(N1+N2) – log(N3)] / log(2)
    D = (np.log10(N1+N2)-np.log10(N3))/np.log10(2)

    # 5.计算指数移动平均的参数 alpha = exp(-4.6(D-1))
    alpha = np.exp(-4.6*(D-1))

    # 设置上线
    if clip:
        alpha = np.clip(alpha, 0.01, 0.2)

    FRAMA = []
    idx = min(np.argwhere(~np.isnan(alpha)))-1
    for row, data in enumerate(alpha):
        if row == (idx):
            FRAMA.append(df.iloc[row])
        elif row > (idx):
            FRAMA.append(data*df.iloc[row] +
                         (1-data)*FRAMA[-1])
        else:
            FRAMA.append(np.nan)

    FRAMA_se = pd.Series(FRAMA, index=df.index)

    return FRAMA_se


def High_Low_Ndays(ser:pd.Series,
                   start_idx:int,
                   close:Union[int, float, bool]=False) -> pd.Series:
    ''' 某个序列中各元素一直大于或小于它前面元素的最大个数
        TODO: 加速
    '''
    if isinstance(close,float) and 0.8<=close<=1.0:
        clr = close
    elif close is False:
        clr = 1
    else:
        raise ValueError
    rkn = pd.Series(np.zeros_like(ser))
    for i,tmp in enumerate(ser.iloc[start_idx:]):
        sgn, tmpi = 0, i+start_idx
        for j in range(tmpi-1,-1,-1):
            if tmp>clr*ser.iloc[j] and sgn>=0:
                sgn = 1
            elif tmp<(2-clr)*ser.iloc[j] and sgn<=0:
                sgn = -1
            elif clr==1.0 and tmp==ser.iloc[j]:
                pass
            else:
                rkn.iloc[tmpi] = sgn*(tmpi-j) -sgn
                break
            if j==0:
                rkn.iloc[tmpi] = tmpi if sgn>=0 else -tmpi
    return rkn


def High_Low_Ndays2(ser:pd.Series,
                    start_idx:int,
                    close:Union[int, float, bool]=False) -> pd.Series:
    ''' 某个序列中各元素一直大于或小于它前面元素的最大个数 '''
    if isinstance(close,float) and 0.8<=close<=1.0:
        clr = close
    elif close is False:
        clr = 1
    else:
        raise ValueError
    rkn = pd.Series(np.zeros_like(ser,dtype='int'))
    for i,tmp in enumerate(ser.iloc[start_idx:]):
        sgn, tmpi, j, vrkn = 0, i+start_idx, i+start_idx-1, 0
        while j>-1:
            jval = ser.iloc[j]
            vrkn = rkn.iloc[j]
            if sgn>=0 and tmp>clr*jval:
                sgn = 1; j-=(vrkn if vrkn>=0 else 0)
            elif sgn<=0 and tmp<(2-clr)*jval:
                sgn = -1; j-=(-vrkn if vrkn<0 else 0)
            elif clr==1 and tmp==jval:
                pass
            else:
                rkn.iloc[tmpi] = sgn*(tmpi-j-1)
                break
            j-=1
        if j==-1: rkn.iloc[tmpi] = tmpi if sgn>=0 else -tmpi
    return rkn


def NHigh_Days(val:Union[dict, pd.Series],
               base:Optional[pd.DataFrame]=None,
               name:Optional[str]=None) -> pd.Series:
    if isinstance(val, dict): vs = pd.Series(val)
    else: vs = val.copy()
    nidx = vs.index
    if base is not None:
        start_idx = base.shape[0]
        varr = np.append(base.values[:,0], vs.values)
        iarr = np.append(base.values[:,1], np.zeros_like(vs,dtype='int'))
    else:
        start_idx = 1
        varr, iarr = vs.values, np.zeros(vs,dtype='int')
    # TODO: New_High
    


def MM_Dist_Pd(pds:pd.DataFrame,
               windows:int=120,
               fn:Union[Callable, None]=None):
    ''' 最大最小分布：滚动窗口 '''
    mm_dist_pd = pds.copy()
    mm_dist_pd.iloc[:windows]=np.nan
    if fn is None:
        def fn(x): return x
    for i,bt in enumerate(pds.rolling(windows)):
        if len(bt) < windows:
            continue
        lmin, lmax = fn(bt.min()), fn(bt.max())
        val = (fn(bt.iloc[-1])-lmin)/(lmax-lmin)
        mm_dist_pd.iloc[i] = val*100
    return mm_dist_pd

def MM_Dist_Ser(ser:pd.Series,
                winds:int=120,
                fn:Union[Callable, None]=None):
    ''' pd.Series 的最大最小分布 '''
    mm_dist_ser = ser.copy()
    mm_dist_ser.iloc[:winds]=np.nan
    if fn is None: fn = lambda x:x
    for i,bt in enumerate(ser.rolling(winds)):
        if len(bt) < winds:
            continue
        lmin, lmax = fn(bt.min()+0.01), fn(bt.max()+0.01)
        if lmax==lmin:
            val=0
        else:
            val = (fn(bt.iloc[-1]+0.01)-lmin)/(lmax-lmin)
        mm_dist_ser.iloc[i] = val*100
    return mm_dist_ser

def MM_Range_Pd(pds:pd.DataFrame,
                id_name=None,
                fn:Union[Callable, None]=None):
    ''' 最大最小分布：全范围
        有未来函数
    '''
    mm_dist_pd = pds.copy()
    if fn is None:
        def fn(x): return x
    lmin, lmax = fn(pds.min()), fn(pds.max())
    mm_dist_pd = (fn(pds)-lmin)/(lmax-lmin)*100
    return mm_dist_pd if id_name is None else mm_dist_pd[id_name]

def SMM_Dist_Pd(pds:pd.DataFrame,
                windows:int=120,
                fn:Union[Callable, None]=None):
    ''' 最大最小分布：滚动窗口 '''
    mm_dist_pd = pds.copy()
    mm_dist_pd.iloc[:windows]=np.nan
    if fn is None:
        def fn(x): return x
    for i,bt in enumerate(pds.rolling(windows)):
        if len(bt) < windows:
            continue
        lmin = fn(0.7*bt.mean())
        lmax =  fn(bt.max())
        val = (fn(bt.iloc[-1])-lmin)/(lmax-lmin)
        val[val<0] = 0
        mm_dist_pd.iloc[i] = val*100
    return mm_dist_pd

def SMM_Dist_Ser(ser:pd.Series,
                 windows:int=120,
                 fn:Union[Callable, None]=None):
    ''' 最大最小分布：滚动窗口 '''
    mm_dist_ser = ser.copy()
    mm_dist_ser.iloc[:windows]=np.nan
    if fn is None:
        def fn(x): return x
    for i,bt in enumerate(ser.rolling(windows)):
        if len(bt) < windows:
            continue
        lmin = fn(0.7*bt.mean())
        lmax =  fn(bt.max())
        if lmax==lmin:
            val=0
        else:
            val = (fn(bt.iloc[-1])-lmin)/(lmax-lmin)
        mm_dist_ser.iloc[i] = val*100
    return mm_dist_ser

Log_MM_Dist_Pd = partial(MM_Dist_Pd,fn=np.log)
Log_MM_Dist_Ser = partial(MM_Dist_Ser,fn=np.log)


def _to_drawup_series(returns):
    prices = _prepare_prices(returns)
    dd = prices / np.minimum.accumulate(prices) - 1.0
    return dd.replace([np.inf, -np.inf, -0], 0)


def drawdown_details(drawdown, is_up:bool=True):

    if is_up:
        clm_name = [
            "start","valley","end","days","vdays","max drawdown","99% max drawdown"
        ]
        drawdown = to_drawdown_series(drawdown)
    else:
        clm_name = [
            "start","top","end","days","tdays","max rebound","99% max rebound"
        ]
        drawdown = _to_drawup_series(drawdown)
    
    if isinstance(drawdown.index[0], str):
        drawdown.index = pd.to_datetime(drawdown.index)

    def _drawdown_details(drawdown):
        # mark no drawdown
        no_dd = drawdown == 0

        # extract dd start dates, first date of the drawdown
        if is_up:
            starts = ~no_dd & no_dd.shift(1)
        else:
            starts = (~no_dd).shift(-1) & no_dd
        starts = list(starts[starts.values].index)

        # extract end dates, last date of the drawdown
        ends = no_dd & (~no_dd).shift(1)
        ends = ends.shift(-1, fill_value=False)
        ends = list(ends[ends.values].index)
        # print(starts[:4],ends[:4])

        # no drawdown :)
        if not starts:
            return pd.DataFrame(
                index=[],
                columns=clm_name
            )

        # drawdown series begins in a drawdown
        if ends and starts[0] > ends[0]:
            starts.insert(0, drawdown.index[0])

        # series ends in a drawdown fill with last date
        if not ends or starts[-1] > ends[-1]:
            ends.append(drawdown.index[-1])

        # build dataframe from results
        data = []
        for i, _ in enumerate(starts):
            dd = drawdown[starts[i] : ends[i]]
            if is_up:
                clean_dd = -remove_outliers(-dd, 0.99)
            else:
                clean_dd = remove_outliers(-dd, 0.95)
            data.append(
                (
                    starts[i],
                    dd.idxmin() if is_up else dd.idxmax(),
                    ends[i],
                    (ends[i] - starts[i]).days + 1,
                    ((dd.idxmin() if is_up else dd.idxmax())-starts[i]).days+1,
                    dd.min() * 100 if is_up else dd.max()*100,
                    clean_dd.min() * 100 if is_up else -clean_dd.min()*100,
                )
            )

        df = pd.DataFrame(
            data=data,
            columns=clm_name
        )
        df["days"] = df["days"].astype(int)
        if is_up:
            df["max drawdown"] = df["max drawdown"].astype(float)
            df["99% max drawdown"] = df["99% max drawdown"].astype(float)
        else:
            df["max rebound"] = df["max rebound"].astype(float)
            df["99% max rebound"] = df["99% max rebound"].astype(float)
        df["end"] = df["end"].apply(lambda x:x.strftime("%Y-%m-%d"))
        df["start"] = df["start"].apply(lambda x:x.strftime("%Y-%m-%d"))
        if is_up:
            df["valley"] = df["valley"].apply(lambda x:x.strftime("%Y-%m-%d"))
        else:
            df["top"] = df["top"].apply(lambda x:x.strftime("%Y-%m-%d"))

        return df

    if isinstance(drawdown, pd.DataFrame):
        _dfs = {}
        for col in drawdown.columns:
            _dfs[col] = _drawdown_details(drawdown[col])
        return pd.concat(_dfs, axis=1)

    return _drawdown_details(drawdown)