from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import pickle
from six import StringIO,BytesIO # py3的环境，使用BytesIO
import talib
import datetime
import unicodedata
import builtins

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict

pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 设置显示宽度，避免列被换行
pd.set_option('display.expand_frame_repr', False) # 3. 每行都不折行


today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1) 
last_trade_day = get_trade_days(end_date=yesterday, count=1)[0]

# --------------------参数设置--------------------------
min_liquidity = {"slow": 50e6, "fast": 100e6}  # 慢轨5000万, 快轨1亿
corr_threshold = 0.85  # 相关性阈值

# --------------------初始池子筛选--------------------------
def initial_etf_filter():
    # 获取所有证券信息，筛选 ETF
    df = get_all_securities(["etf"])
    # 定义深市核心品种的白名单（如创业板ETF、创业50ETF、深证100等）
    core_whitelist = ["159915.XSHE", "159949.XSHE", "159901.XSHE"]

    # 定义黑名单，排除特定 ETF
    blacklist = ["510900.XSHG"]

    # 只保留沪市的 ETF（代码以 '5' 开头）或白名单中的深市核心品种
    df = df[df.index.str.startswith("5") | df.index.isin(core_whitelist)]

    # 剔除黑名单中的 ETF
    df = df[~df.index.isin(blacklist)]

    # 剔除沪市中债券、境外、黄金、货币基金类的 ETF
    df = df[~df.index.str.startswith(("511", "513", "518", "519", "520", "551"))]

    # 剔除上市时间不足 60 天的 ETF，且 end_date 不等于 "2200-01-01" 的 ETF
    df = df[((yesterday - df["start_date"]).dt.days >= 60) & (df["end_date"] > yesterday)]
    
    return df.index.tolist()

# --------------------慢轨--------------------------
def slow_track_filter(etf_list):

    # 筛选过去 20 日成交额中位数大于 5000 万的 ETF
    money_median_20d = history(20, "1d", "money", etf_list, df=True).median()
    filtered_etfs = [etf for etf in etf_list if money_median_20d[etf] > min_liquidity["slow"]]

    # 获取过去 120 日的每日收益率
    price_data = history(120, "1d", "close", filtered_etfs, df=True)
    price_data = price_data.fillna(method="ffill").dropna(axis=1, how="any")
    returns = price_data.pct_change().dropna()
    filtered_etfs = returns.columns.tolist()

    # 计算Spearman相关系数矩阵
    corr_matrix = returns.corr(method="spearman")

    # 转换为距离矩阵
    distance_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(distance_matrix.values, 0)

    # 层次聚类
    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")

    # 聚类划分（基于距离阈值）
    threshold = np.sqrt(2 * (1 - corr_threshold))
    clusters = fcluster(Z, threshold, criterion="distance")

    # 创建聚类字典
    cluster_dict = defaultdict(list)
    for etf, cluster_id in zip(filtered_etfs, clusters):
        cluster_dict[cluster_id].append(etf)

    # 获取成交额数据用于选优
    money_data = history(30, "1d", "money", filtered_etfs, df=True).fillna(0)

    # 每个簇选择流动性最好的ETF
    selected_etfs = [max(etfs, key=lambda etf: money_data[etf].mean()) for etfs in cluster_dict.values() if etfs]

#     # 记录聚类信息
#     print(f"层次聚类结果: {len(cluster_dict)}个簇")
#     for cluster_id, etf_list in cluster_dict.items():
#         if len(etf_list) > 1:
#             print(f"  簇{cluster_id}: {len(etf_list)}只ETF")
#             for etf in etf_list[:5]:
#                 print(f"    - {get_security_info(etf).display_name}")

    return selected_etfs


# --------------------快轨--------------------------
def fast_track_filter(etf_list, etf_pool):

    # 获取数据
    money_data = history(5, "1d", "money", etf_list, df=True).mean()
    price_data = history(60, "1d", "close", etf_list, df=True).fillna(method="ffill").dropna(axis=1, how="any")
    returns_data = price_data.pct_change().dropna()
    etf_list = price_data.columns.tolist()

    # 排除已在慢轨池中的ETF，并筛选流动性和涨幅条件
    close_prices = history(20, "1d", "close", etf_list, df=True)
    filtered_etfs = [
        etf
        for etf in etf_list
        if etf not in etf_pool and money_data[etf] > min_liquidity["fast"] and (close_prices[etf].iloc[-1] / close_prices[etf].iloc[0] - 1) > 0.05
    ]
    filtered_etfs = sorted(filtered_etfs, key=lambda etf: money_data[etf], reverse=True)

    core_pool = etf_pool.copy()

    for etf_new in filtered_etfs:
        returns_new = returns_data[[etf_new]]
        returns_old = returns_data[core_pool]
        # 计算相关性
        corr_with_core = returns_old.corrwith(returns_new[etf_new])

        # 情况 A：与 Core Pool 中某只 ETF_Old 高度相关
        if corr_with_core.max() > corr_threshold:

            etf_old = corr_with_core.idxmax()  # 找到相关性最高的 ETF_Old

            # 仅当 ETF_New 的流动性显著优于 ETF_Old 时，替换
            if money_data[etf_new] > money_data[etf_old] * 2:
                core_pool.remove(etf_old)
                core_pool.append(etf_new)

        if corr_with_core.max() < 0.7:
            # 情况 B：与 Core Pool 中任何 ETF_Old 都不相关
            core_pool.append(etf_new)

    return core_pool

# --------------------计算动量--------------------------
# 动量得分选股
def get_best_etf(etf_list, days, max_annualized_returns, min_r2):

    data = pd.DataFrame(index=etf_list, columns=["code","name", "annualized_returns", "r2", "score"])
    for etf in etf_list:
        # 获取数据
        df = get_price(etf, end_date=today, frequency="daily", fields=["close", "high"], count=days, panel=False)
        prices = df["close"].values

        # 设置参数
        y = np.log(prices)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))

        # 计算年化收益率
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        data.loc[etf, "annualized_returns"] = round(math.exp(slope * 250) - 1, 2)

        # 计算R²
        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        data.loc[etf, "r2"] = round(1 - ss_res / ss_tot if ss_tot else 0, 2)

        # 计算得分
        data.loc[etf, "score"] = round(data.loc[etf, "annualized_returns"] * data.loc[etf, "r2"], 2)

        # 获取代码与名称
        data.loc[etf, "code"] = etf
        data.loc[etf, "name"] = get_security_info(etf).display_name

        # 过滤近3日跌幅超过5%的ETF
        if min(prices[-1] / prices[-2], prices[-2] / prices[-3], prices[-3] / prices[-4]) < 0.95:
            data.loc[etf, "score"] = 0

    # 过滤ETF，并按得分降序排列
    filtered = data[(data["score"] > 0) & (data["annualized_returns"] < max_annualized_returns) & (data["r2"] > min_r2)].sort_values(
        by="score", ascending=False
    )

    return filtered

#---------------------------美化打印--------------------------

# 计算字符宽度
def _width(s: str) -> int:
    """返回字符串在终端里真实占的列数"""
    return builtins.sum(2 if unicodedata.east_asian_width(ch) in 'FW' else 1 for ch in s)


# df打印
def pretty_print(df: pd.DataFrame, sep: str = '  ') -> None:
    """强制左对齐、屏幕宽度对齐、不换行打印 DataFrame"""
    # 1. 全部转字符串
    str_df = df.astype(str)
    cols = list(df.columns)

    # 2. 计算每列“屏幕最大宽度”
    widths = {}
    for c in cols:
        w = max(_width(str(v)) for v in str_df[c])   # 数据最大宽度
        widths[c] = max(w, _width(c))                # 再与列名比

    # 3. 表头
    header = sep.join(c.ljust(widths[c] - (_width(c) - len(c))) for c in cols)
    print(header)

    # 4. 数据行
    for _, row in str_df.iterrows():
        line = sep.join(
            row[c].ljust(widths[c] - (_width(row[c]) - len(row[c]))) for c in cols
        )
        print(line)

# --------------------主函数--------------------------
list_step1 = initial_etf_filter()
pool_step2 = slow_track_filter(list_step1)
final_pool = fast_track_filter(list_step1, pool_step2)
print(f"Final Pool Size: {len(final_pool)}")
# 简单验证一下名字，确保没有奇怪的东西混进来
for etf in final_pool:
    print(f"{etf} - {get_security_info(etf).display_name}")

data = get_best_etf(final_pool, days=25, max_annualized_returns=999, min_r2=0.7)
pretty_print(data)


 