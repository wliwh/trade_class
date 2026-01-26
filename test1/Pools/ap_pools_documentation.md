# 寻找市场的脊梁——基于聚类算法的ETF轮动池构建研究

## 研究背景

论坛上已经由很多ETF轮动策略，但是如何确定轮动所在的ETF池子尚待研究。

现有的动量轮动策略，往往将重点放在“选哪只”上面，而忽略了“从哪里选”这个问题。大多数策略使用的ETF池子，要么是有后视镜嫌疑的4只ETF，要么是基于主观经验加入特定行业ETF。

这两种方式都存在明显的局限性，难免带有幸存者偏差，且难以适应市场风格的长期演变。因此，本研究试图探讨一个核心问题：能否通过数学方法，客观地筛选出一个既能覆盖市场主流风格，又彼此低相关的精英ETF池？

## 研究思路

本研究提出了一种基于聚类算法的ETF池构建框架。我们的核心观点是，一个优秀的轮动池应当具备两个特征：低冗余和高代表性。

为此，我们设计了包含三个步骤的筛选流程：

### 1. 基础过滤：提升数据质量

在进行复杂的数学计算之前，首先需要清洗数据。我们设定了一系列硬性指标，剔除那些不具备研究价值的标的。我们将成交额过低、上市时间过短以及非核心资产（如部分边缘的债券或含有特定风险的品种）排除在外。这一步是为了确保后续选出的标的都具有良好的流动性和可交易性。

### 2. 聚类分析：发现市场结构

这是本研究的核心环节。我们引入了无监督学习中的聚类算法，利用ETF之间的历史价格相关性矩阵，自动识别市场的板块结构。

我们分别测试了三种方法，它们各有侧重：

*   Affinity Propagation (AP聚类)：这是一种自适应的聚类方法。我们不需要预先告诉它市场分成了几个板块，它会通过节点间的消息传递，自动找出最具代表性的几个中心点（Exemplars）。这种方法特别适合发现市场中自然形成的、未被定义的风格。
*   Hierarchical Clustering (层次聚类)：通过计算资产间的距离，像搭积木一样由下而上逐步合并。我们可以通过设定一个相关性阈值（比如0.85），像切蛋糕一样把树状图切开，从而得到不同精细度的分类。
*   Minimum Spanning Tree (MST，最小生成树)：基于图论的方法。我们将所有ETF看作图中的节点，相关性看作连线。MST会保留连接所有节点所需的最强连线骨架，剪断所有冗余的弱联系。通过切断骨架上最长的几根线，我们就能得到自然分离的社群。

与传统的行业分类不同，聚类算法完全基于数据的真实波动。如果消费和医疗在某段时间走势高度一致，算法就会敏锐地将它们归为一类。

通过聚类，我们将高相关性的ETF归并为一个个簇。在每个簇中，我们只保留流动性最好的一只作为代表。这样做的目的，是强行切断资产间的高相关性，确保最终的池子里的每一只ETF，都代表了市场的一种独特声音。

### 3. 动量验证：检验池子效率

为了验证构建出的ETF池是否具有投资价值，我们引入了一个简单的动量评分机制。我们观察在同等动量策略下，基于聚类构建的池子是否能比简单的全市场池捕捉到更清晰的趋势。

## 研究结论展望

通过代码实现与初步回测，我们发现基于聚类构建的ETF池在降低波动率和提升夏普比率方面展现出了潜力。这表明，在动量轮动之前进行一步科学的“池子瘦身”，是提升策略稳健性的有效手段。本代码即为该研究思路的具体实现。

附录：研究代码实现

```python
from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import datetime
import unicodedata
import builtins
import math
import networkx as nx
from sklearn.cluster import AffinityPropagation
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict

# -------------------- 配置区域 (Initial Configuration) --------------------
CONFIG = {
    # 基础过滤参数
    "min_liquidity": 50e6,       # 最小成交额 (5000万)
    "min_listing_days": 500,     # 最小上市天数
    
    # 类型过滤配置 (True=剔除, False=保留)
    "filter_bond_money": True,   # 剔除债券、货币
    "filter_qdii": False,        # 剔除跨境/QDII
    "filter_gold": False,        # 剔除黄金
    "black_list": ["510900.XSHG"],  # 黑名单
    
    # 聚类方法选择 ("ap", "hierarchical", "mst")
    "clustering_method": "ap",

    # AP聚类参数
    "ap_damping": 0.8,           # 阻尼系数 (0.5-1.0)，防止震荡，越大越稳定
    "ap_preference": None,       # 偏好值 (None = 使用中位数)，值越大簇越多
    
    # 聚类参数 (Hierarchical Clustering)
    "cluster_corr_threshold": 0.85, # 聚类相关性阈值 (决定簇的数量/大小)
    "correlation_window": 120,      # 计算相关性的回看窗口 (天)
    
    # MST 聚类参数
    "mst_n_clusters": 10,        # MST聚类目标簇数量 (切断由强到弱的K-1条边)

    # 动量评分参数
    "score_window": 25,          # 动量评分的回看窗口 (天)
    "min_r2": 0.7,               # 最小 R 方
    "max_annualized_returns": 99.99, # 最大年化收益 (过滤异常值)
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
    2. 剔除上市时间过短的
    3. 剔除黑名单、非核心深市、特定前缀(债券/货币等)
    """
    print("Step 1: Initial Filtering...")
    df = get_all_securities(["etf"])
    
    # 上市时间检查
    df = df[((yesterday - df["start_date"]).dt.days >= CONFIG["min_listing_days"]) & (df["end_date"] > yesterday)]

    # 黑名单
    blacklist = CONFIG["black_list"]
    df = df[~df.index.isin(blacklist)]
    
    # 关键字剔除 (主要针对深市及名字中包含特定标识的ETF)
    exclude_keywords = []
    if CONFIG["filter_gold"]:       
        exclude_keywords.extend(["黄金", "上海金"])
    if CONFIG["filter_qdii"]:       
        exclude_keywords.extend(["QDII", "标普", "纳指", "纳斯达", "道琼斯", "恒生", "H股",\
            "沙特", "巴西", "日经", "德国", "法国", "英国", "美国", "海外"])
    
    if exclude_keywords:
        print(f"  -> Excluding keys: {exclude_keywords}")
        pattern = "|".join(exclude_keywords)
        df = df[~df['display_name'].str.contains(pattern, regex=True)]

    # 价格过滤 (针对债券、货币)
    # 策略：如果最近1日均价 > 90，则认为是债券或货币基金
    if CONFIG["filter_bond_money"]:
        print("  -> Checking for high-priced ETFs (Bond/Money > 90)...")
        try:
            # 获取最近1天均价
            avg_prices_df = history(1, "1d", "avg", df.index.tolist(), df=True)
            
            if not avg_prices_df.empty:
                last_prices = avg_prices_df.iloc[-1]
                # 找出价格 > 90 的代码
                high_price_etfs = last_prices[last_prices > 90].index.tolist()
                
                if high_price_etfs:
                    print(f"  -> Filtering out {len(high_price_etfs)} ETFs with price > 90 (likely Bond/Money):")
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
        
        display_name = get_security_info(best_etf).display_name
        print(f"    Cluster {label}: Selected {display_name} ({best_etf}) from {len(etfs)} candidates.")

    return selected_etfs

# -------------------- 2.2 层次聚类筛选 (Hierarchical Clustering) --------------------------
def hierarchical_clustering_filter(etf_list):
    """
    使用层次聚类筛选
    """
    print(f"Step 2: Hierarchical Clustering (Corr Threshold={CONFIG['cluster_corr_threshold']})...")

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
    final_etf_list = returns.columns.tolist()

    # 3. 计算相关性矩阵 (Spearman)
    print(f"  -> Computing correlation matrix for {len(final_etf_list)} ETFs...")
    corr_matrix = returns.corr(method="spearman")

    # 4. 转换为距离矩阵 & 聚类
    dist_threshold = np.sqrt(2 * (1 - CONFIG["cluster_corr_threshold"]))
    distance_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(distance_matrix.values, 0)

    condensed_dist = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_dist, method="average")

    # 5. 聚类划分
    clusters = fcluster(Z, dist_threshold, criterion="distance")
    
    n_clusters_ = len(set(clusters))
    print(f"  -> Converged into {n_clusters_} clusters.")

    # 6. 从每个簇中选择流动性最好的 ETF
    money_means = history(30, "1d", "money", final_etf_list, df=True).mean()

    cluster_dict = defaultdict(list)
    for etf, label in zip(final_etf_list, clusters):
        cluster_dict[label].append(etf)

    selected_etfs = []
    
    print("  -> Selecting best liquidity ETF from each cluster:")
    for label, etfs in cluster_dict.items():
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
    final_etf_list = returns.columns.tolist() 

    # 3. 构建距离矩阵 (Distance Matrix)
    print(f"  -> Computing correlation & distance matrix for {len(final_etf_list)} ETFs...")
    corr_matrix = returns.corr(method="spearman")
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(dist_matrix.values, 0)

    # 4. 构建完全图并生成 MST
    G = nx.Graph()
    G.add_nodes_from(final_etf_list)
    
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
    mst_edges = sorted(mst.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    
    num_to_remove = CONFIG['mst_n_clusters'] - 1
    
    if num_to_remove > 0 and len(mst_edges) >= num_to_remove:
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
        best_etf = max(etfs_in_cluster, key=lambda x: money_means.get(x, 0))
        selected_etfs.append(best_etf)
        
        display_name = get_security_info(best_etf).display_name
        print(f"    Cluster {i+1}: Selected {display_name} ({best_etf}) from {len(etfs_in_cluster)} candidates.")
        
    return selected_etfs

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
        
        # 简单止损过滤：最近3天跌幅超过5%则归零
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
    
    # 3. 动量评分展示
    result_df = get_best_etf(final_pool)
    
    print("\nFinal Selected ETF Pool (Top Momentum):")
    pretty_print(result_df)
```
