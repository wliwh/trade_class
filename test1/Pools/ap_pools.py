from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import datetime
import unicodedata
import builtins
import math
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict
from pca_analysis import analyze_pool_pca

# -------------------- 配置区域 (Initial Configuration) --------------------
CONFIG = {
    # 基础过滤参数
    "min_liquidity": 50e6,       # 最小成交额 (5000万)
    "min_listing_days": 500,      # 最小上市天数
    
    # 类型过滤配置 (True=剔除, False=保留)
    "filter_bond_money": True,   # 剔除债券、货币 (511)
    "filter_qdii": False,         # 剔除跨境/QDII (513)
    "filter_gold": False,         # 剔除黄金 (518)
    "filter_others": False,       # 剔除其他/老基金 (519, 520, 551)
    
    # 聚类方法选择 ("hierarchical", "ap", "mst")
    "clustering_method": "ap",

    # 聚类参数 (Hierarchical Clustering)
    "cluster_corr_threshold": 0.85, # 聚类相关性阈值 (决定簇的数量/大小)
    "correlation_window": 120,      # 计算相关性的回看窗口 (天)
    
    # MST 聚类参数
    "mst_n_clusters": 10,        # MST聚类目标簇数量 (切断由强到弱的K-1条边)

    # AP聚类参数
    "ap_damping": 0.8,           # 阻尼系数 (0.5-1.0)，防止震荡，越大越稳定
    "ap_preference": None,       # 偏好值 (None = 使用中位数)，值越大簇越多
    
    # 动量评分参数
    "score_window": 25,          # 动量评分的回看窗口 (天)
    "min_r2": 0.7,               # 最小 R 方
    "max_annualized_returns": 9.99, # 最大年化收益 (过滤异常值)
    
    # PCA 分析参数
    "pca_rolling_window": 60,    # 滚动 PCA 窗口 (天)
    "pca_overlay_etf": "510300.XSHG", # 叠加显示的基准 ETF (如沪深300)，为 None 则不显示
}

# -------------------- 工具函数 --------------------------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

def _width(s: str) -> int:
    """返回字符串在终端里真实占的列数"""
    return builtins.sum(2 if unicodedata.east_asian_width(ch) in 'FW' else 1 for ch in s)

def pretty_print(df: pd.DataFrame, sep: str = '  ') -> None:
    """强制左对齐、屏幕宽度对齐、不换行打印 DataFrame"""
    str_df = df.astype(str)
    cols = list(df.columns)
    widths = {}
    for c in cols:
        w = max(_width(str(v)) for v in str_df[c])
        widths[c] = max(w, _width(c))
    
    header = sep.join(c.ljust(widths[c] - (_width(c) - len(c))) for c in cols)
    print(header)
    for _, row in str_df.iterrows():
        line = sep.join(
            row[c].ljust(widths[c] - (_width(row[c]) - len(row[c]))) for c in cols
        )
        print(line)

# -------------------- 1. 初始池筛选 --------------------------
def initial_etf_filter():
    """
    基础过滤：
    1. 获取所有ETF
    2. 剔除黑名单、非核心深市、特定前缀(债券/货币等)
    3. 剔除上市时间过短的
    """
    print("Step 1: Initial Filtering...")
    df = get_all_securities(["etf"])
    
    # 白名单/黑名单
    # 黑名单
    # core_whitelist = ["159915.XSHE", "159949.XSHE", "159901.XSHE"] <-- No longer needed
    blacklist = ["510900.XSHG"]
    
    # 筛选逻辑
    # 筛选逻辑：不限制交易所，仅剔除黑名单
    # df = df[df.index.str.startswith("5") | df.index.isin(core_whitelist)]  <-- Removed restriction
    df = df[~df.index.isin(blacklist)]
    
    # 动态构建剔除列表
    exclude_prefixes = []
    # if CONFIG["filter_bond_money"]: exclude_prefixes.append("511")
    if CONFIG["filter_qdii"]:       exclude_prefixes.append("513")
    if CONFIG["filter_gold"]:       exclude_prefixes.append("518")
    if CONFIG["filter_others"]:     exclude_prefixes.extend(["519", "520", "551"])
    
    if exclude_prefixes:
        print(f"  -> Excluding prefixes: {exclude_prefixes}")
        df = df[~df.index.str.startswith(tuple(exclude_prefixes))]
    
    # 关键字剔除 (主要针对深市及名字中包含特定标识的ETF)
    exclude_keywords = []
    # if CONFIG["filter_bond_money"]: 
    #     exclude_keywords.extend(["债", "货币", "理财"])
    if CONFIG["filter_gold"]:       
        exclude_keywords.extend(["黄金", "上海金"])
    if CONFIG["filter_qdii"]:       
        exclude_keywords.extend(["QDII", "标普", "纳指", "道琼斯", "恒生", "H股", "日经", "德国", "法国", "英国", "美国", "海外"])
    
    if exclude_keywords:
        print(f"  -> Excluding keys: {exclude_keywords}")
        # using regex to filter
        pattern = "|".join(exclude_keywords)
        df = df[~df['display_name'].str.contains(pattern, regex=True)]
    
    # 上市时间检查
    df = df[((yesterday - df["start_date"]).dt.days >= CONFIG["min_listing_days"]) & (df["end_date"] > yesterday)]

    # ----------------------------------------------------
    # 新增：价格过滤 (针对债券、货币)
    # 策略：如果最近1日均价 > 90，则认为是债券或货币基金
    # ----------------------------------------------------
    if CONFIG["filter_bond_money"]:
        print("  -> Checking for high-priced ETFs (Bond/Money > 90)...")
        # 使用 history 获取最近1天均价 (avg)
        # Transpose后的 DataFrame: index=etf_code, columns=[date] (though history with df=True usually returns index=date, columns=code)
        # Wait, history(..., df=True) returns index=date, columns=security. 
        # So .iloc[0] or .transpose() is needed.
        # User/Comment suggested: history(1, "1d", "avg", df.index, df=True).transpose()
        # This results in index=security, columns=[date]
        
        try:
            avg_prices_df = history(1, "1d", "avg", df.index.tolist(), df=True)
            
            if not avg_prices_df.empty:
                # avg_prices_df: rows=date (1 row), cols=securities
                last_prices = avg_prices_df.iloc[-1]
                
                # 找出价格 > 90 的代码
                high_price_etfs = last_prices[last_prices > 90].index.tolist()
                
                if high_price_etfs:
                    print(f"  -> Filtering out {len(high_price_etfs)} ETFs with price > 90 (likely Bond/Money):")
                    # print(f"     Examples: {high_price_etfs[:5]}")
                    
                    # 从 df 中剔除
                    df = df[~df.index.isin(high_price_etfs)]
        except Exception as e:
            print(f"  [Warning] Failed to filter by price: {e}")

    initial_list = df.index.tolist()
    print(f"  -> Found {len(initial_list)} candidate ETFs.")
    return initial_list

# -------------------- 2.1 AP 聚类筛选 (Affinity Propagation) --------------------------
def ap_clustering_filter(etf_list):
    """
    使用 Affinity Propagation (AP) 算法进行聚类筛选
    """
    print(f"Step 2: Affinity Propagation Clustering (Damping={CONFIG['ap_damping']})...")
    
    # 1. 预先过滤流动性 (减少计算量)
    money_median_20d = history(20, "1d", "money", etf_list, df=True).median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > CONFIG["min_liquidity"]]
    
    if not valid_etfs:
        print("  -> No ETFs passed liquidity filter.")
        return []

    # 2. 获取价格并计算收益率
    price_data = history(CONFIG["correlation_window"], "1d", "close", valid_etfs, df=True)
    price_data = price_data.fillna(method="ffill").dropna(axis=1, how="any")
    
    if price_data.empty:
        print("  -> Price data empty.")
        return []
        
    returns = price_data.pct_change().dropna()
    final_etf_list = returns.columns.tolist()
    
    # 3. 计算相关性矩阵 (Spearman)
    print(f"  -> Computing correlation matrix for {len(final_etf_list)} ETFs...")
    corr_matrix = returns.corr(method="spearman")
    
    # 4. 执行 AP 聚类
    # 注意：AP 接受相似度矩阵 (Similarity Matrix)，相关系数本身就是一种相似度 (-1 ~ 1)
    # affinity='precomputed' 表示我们直接传入矩阵
    ap = AffinityPropagation(
        damping=CONFIG["ap_damping"],
        preference=CONFIG["ap_preference"],
        affinity='precomputed',
        random_state=42
    )
    ap.fit(corr_matrix)
    
    labels = ap.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  -> AP converged into {n_clusters_} clusters.")
    
    # 5. 从每个簇中选择流动性最好的 ETF
    
    # 准备成交额数据
    money_data = history(30, "1d", "money", final_etf_list, df=True).mean()
    
    cluster_dict = defaultdict(list)
    for etf, label in zip(final_etf_list, labels):
        cluster_dict[label].append(etf)
        
    selected_etfs = []
    
    print("  -> Selecting best liquidity ETF from each cluster:")
    for label, etfs in cluster_dict.items():
        # 选出该组中 money_data 最大的
        best_etf = max(etfs, key=lambda x: money_data.get(x, 0))
        selected_etfs.append(best_etf)
        
        # 可选：打印每组详情
        display_name = get_security_info(best_etf).display_name
        
        print(f"    Cluster {label}: Selected {display_name} ({best_etf}) from {len(etfs)} candidates.")

    return selected_etfs

# -------------------- 2.2 层次聚类筛选 (Hierarchical Clustering) --------------------------
def hierarchical_clustering_filter(etf_list):
    """
    使用层次聚类筛选 (移植自 oix_pools.py slow_track_filter)
    """
    print(f"Step 2: Hierarchical Clustering (Corr Threshold={CONFIG['cluster_corr_threshold']})...")

    # 1. 预先过滤流动性
    money_median_20d = history(20, "1d", "money", etf_list, df=True).median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > CONFIG["min_liquidity"]]

    if not valid_etfs:
        print("  -> No ETFs passed liquidity filter.")
        return []

    # 2. 获取价格并计算收益率
    # 注意: oix 使用 120 天，这里 CONFIG["correlation_window"] 也是 120
    price_data = history(CONFIG["correlation_window"], "1d", "close", valid_etfs, df=True)
    price_data = price_data.fillna(method="ffill").dropna(axis=1, how="any")

    if price_data.empty:
        print("  -> Price data empty.")
        return []

    returns = price_data.pct_change().dropna()
    final_etf_list = returns.columns.tolist()

    # 3. 计算相关性矩阵 (Spearman)
    print(f"  -> Computing correlation matrix for {len(final_etf_list)} ETFs...")
    corr_matrix = returns.corr(method="spearman")

    # 4. 转换为距离矩阵 & 聚类
    # distance = sqrt(2 * (1 - correlation))
    distance_matrix = np.sqrt(2 * (1 - corr_matrix))
    # distance_matrix 对角线设为0 (虽然公式结果也是0，确保数值稳定)
    np.fill_diagonal(distance_matrix.values, 0)

    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")

    # 5. 聚类划分
    # threshold = sqrt(2 * (1 - corr_threshold))
    dist_threshold = np.sqrt(2 * (1 - CONFIG["cluster_corr_threshold"]))
    clusters = fcluster(Z, dist_threshold, criterion="distance")
    
    n_clusters_ = len(set(clusters))
    print(f"  -> Converged into {n_clusters_} clusters.")

    # 6. 从每个簇中选择流动性最好的 ETF
    money_data = history(30, "1d", "money", final_etf_list, df=True).median() # or mean, oix uses mean in fast/slow logic varies slightly, oix slow uses median for filter but mean for selection? oix line 88 uses fillna(0) then selection uses mean.
    # oix line 88: money_data = history(30, "1d", "money", filtered_etfs, df=True).fillna(0)
    # oix selection: max(etfs, key=lambda etf: money_data[etf].mean()) 
    # Wait, money_data[etf] is a series. .mean() is scalar. Correct.
    # I will stick to simple scalar mean for sorting.
    money_means = money_data.mean() if isinstance(money_data, pd.DataFrame) else money_data # JQ returns DF usually. 
    # Actually, let's just get the mean directly
    money_means = history(30, "1d", "money", final_etf_list, df=True).mean()

    cluster_dict = defaultdict(list)
    for etf, label in zip(final_etf_list, clusters):
        cluster_dict[label].append(etf)

    selected_etfs = []
    
    print("  -> Selecting best liquidity ETF from each cluster:")
    for label, etfs in cluster_dict.items():
        # 选出该组中 money_means 最大的
        best_etf = max(etfs, key=lambda x: money_means.get(x, 0))
        selected_etfs.append(best_etf)
        
        display_name = get_security_info(best_etf).display_name
        print(f"    Cluster {label}: Selected {display_name} ({best_etf}) from {len(etfs)} candidates.")

    return selected_etfs

# -------------------- 2.3 MST 聚类筛选 (Minimum Spanning Tree) --------------------------
def mst_clustering_filter(etf_list):
    """
    使用最小生成树 (MST) 算法进行聚类筛选
    原理: 构建完全图 (权重=距离)，生成MST，切断最长的K-1条边，形成K个连通分量
    """
    print(f"Step 2: MST Clustering (Target Clusters={CONFIG['mst_n_clusters']})...")

    # 1. 预先过滤流动性
    money_median_20d = history(20, "1d", "money", etf_list, df=True).median()
    valid_etfs = [etf for etf in etf_list if money_median_20d.get(etf, 0) > CONFIG["min_liquidity"]]

    if not valid_etfs:
        print("  -> No ETFs passed liquidity filter.")
        return []

    # 2. 获取价格并计算收益率
    price_data = history(CONFIG["correlation_window"], "1d", "close", valid_etfs, df=True)
    price_data = price_data.fillna(method="ffill").dropna(axis=1, how="any")

    if price_data.empty:
        print("  -> Price data empty.")
        return []

    returns = price_data.pct_change().dropna()
    final_etf_list = returns.columns.tolist() # 节点列表 of size N

    # 3. 构建距离矩阵 (Distance Matrix)
    # distance = sqrt(2 * (1 - correlation))
    print(f"  -> Computing correlation & distance matrix for {len(final_etf_list)} ETFs...")
    corr_matrix = returns.corr(method="spearman")
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(dist_matrix.values, 0) # 自身距离为0

    # 4. 构建完全图并生成 MST
    G = nx.Graph()
    # 添加节点
    G.add_nodes_from(final_etf_list)
    
    # 添加边 (完全图)
    # 优化: networkx 处理完全图比较慢，我们可以直接通过距离矩阵构建
    # 这里直接遍历矩阵的上三角部分添加边
    edges = []
    n = len(final_etf_list)
    for i in range(n):
        for j in range(i + 1, n):
            u = final_etf_list[i]
            v = final_etf_list[j]
            weight = dist_matrix.iloc[i, j]
            edges.append((u, v, weight))
    
    G.add_weighted_edges_from(edges)
    
    print("  -> Building Minimum Spanning Tree...")
    mst = nx.minimum_spanning_tree(G)
    
    # 5. 切断最长的 K-1 条边
    # 获取MST中所有边及其权重
    mst_edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    
    # 需要移除的边数 = 目标簇数 - 1
    # 如果当前连通分量已经是1 (MST本身)，移除1条边变2个分量...
    # 注意: MST是连通的，初始分量为1
    num_to_remove = CONFIG['mst_n_clusters'] - 1
    
    if num_to_remove > 0 and len(mst_edges) >= num_to_remove:
        # 移除权重最大的前 num_to_remove 条边 (即距离最远/相关性最低的连接)
        # 这样会断开大的群组
        edges_to_remove = mst_edges[:num_to_remove]
        mst.remove_edges_from([(u, v) for u, v, d in edges_to_remove])
        print(f"  -> Removed {num_to_remove} longest edges to form clusters.")
    
    # 6. 获取连通分量 (即聚类结果)
    components = list(nx.connected_components(mst))
    print(f"  -> Generated {len(components)} clusters.")
    
    # 7. 从每个簇中选择流动性最好的 ETF
    money_means = history(30, "1d", "money", final_etf_list, df=True).mean()
    
    selected_etfs = []
    print("  -> Selecting best liquidity ETF from each cluster:")
    
    for i, comp in enumerate(components):
        etfs_in_cluster = list(comp)
        # 选出该组中 money_means 最大的
        best_etf = max(etfs_in_cluster, key=lambda x: money_means.get(x, 0))
        selected_etfs.append(best_etf)
        
        display_name = get_security_info(best_etf).display_name
        print(f"    Cluster {i+1}: Selected {display_name} ({best_etf}) from {len(etfs_in_cluster)} candidates.")
        
    return selected_etfs

# -------------------- 2.5 PCA 分析 (Absorption Ratio) --------------------------
# Moved to pca_analysis.py

# -------------------- 3. 动量评分 --------------------------
def get_best_etf(etf_list):
    """
    计算动量得分并排序
    """
    print("Step 3: Momentum Scoring & Ranking...")
    data = pd.DataFrame(index=etf_list, columns=["code", "name", "annualized_returns", "r2", "score"])
    
    for etf in etf_list:
        # 获取数据
        df = get_price(etf, end_date=today, frequency="daily", fields=["close"], count=CONFIG["score_window"], panel=False)
        if df.empty:
            continue
            
        prices = df["close"].values
        
        # 线性回归 (log price)
        y = np.log(prices)
        x = np.arange(len(y))
        
        # 加权 (近期权重更高)
        weights = np.linspace(1, 2, len(y))
        
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        
        # 年化收益
        ann_ret = round(math.exp(slope * 250) - 1, 2)
        
        # R2
        y_pred = slope * x + intercept
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r2 = round(1 - ss_res / ss_tot if ss_tot else 0, 2)
        
        # 得分
        score = round(ann_ret * r2, 2)
        
        data.loc[etf, "code"] = etf
        try:
            data.loc[etf, "name"] = get_security_info(etf).display_name
        except:
            data.loc[etf, "name"] = "Unknown"
        data.loc[etf, "annualized_returns"] = ann_ret
        data.loc[etf, "r2"] = r2
        data.loc[etf, "score"] = score
        
        # 简单止损过滤：最近3天跌幅超过5%则归零 (参考原逻辑)
        if len(prices) >= 4:
             if min(prices[-1]/prices[-2], prices[-2]/prices[-3], prices[-3]/prices[-4]) < 0.95:
                 data.loc[etf, "score"] = 0

    # 排序与过滤
    filtered = data[
        (data["score"] > 0) & 
        (data["annualized_returns"] < CONFIG["max_annualized_returns"]) & 
        (data["r2"] > CONFIG["min_r2"])
    ].sort_values(by="score", ascending=False)
    
    return filtered

# -------------------- 主程序 --------------------------
if __name__ == "__main__":
    print("-" * 50)
    print(f"Starting {CONFIG['clustering_method'].upper()}-Based ETF Pool Generation")
    print("Configuration:", CONFIG)
    print("-" * 50)
    
    # 1. 初始筛选
    candidates = initial_etf_filter()
    
    # 2. 聚类选择
    if CONFIG["clustering_method"] == "ap":
        final_pool = ap_clustering_filter(candidates)
        method_name = "Affinity Propagation"
    elif CONFIG["clustering_method"] == "mst":
        final_pool = mst_clustering_filter(candidates)
        method_name = "Minimum Spanning Tree (MST)"
    else:
        final_pool = hierarchical_clustering_filter(candidates)
        method_name = "Hierarchical Clustering"
        
    print(f"{method_name} Complete. Pool Size: {len(final_pool)}")
    
    # 2.5 PCA 分析
    analyze_pool_pca(final_pool, CONFIG, overlay_code=CONFIG.get("pca_overlay_etf"))
    
    # 3. 动量评分展示
    result_df = get_best_etf(final_pool)
    
    print("\nFinal Selected ETF Pool (Top Momentum):")
    pretty_print(result_df)
