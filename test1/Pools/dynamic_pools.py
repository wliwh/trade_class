# 导入聚宽库和其他必要模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jqdata import *

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 常用ETF模板
ETF_TEMPLATES = {
    'default': [],  # 默认不指定任何ETF
    'gold_tech_bond': ['518880.XSHG', '159941.XSHE', '511220.XSHG'],  # 黄金+纳斯达克+城投债
    'conservative': ['518880.XSHG', '511220.XSHG', '511260.XSHG'],  # 保守组合：黄金+城投债+国债
    'aggressive': ['159941.XSHE', '513100.XSHG', '512480.XSHG'],  # 激进组合：纳斯达克+标普500+半导体
    'balanced': ['518880.XSHG', '159941.XSHE', '510300.XSHG', '511220.XSHG'],  # 平衡组合
    'income': ['511220.XSHG', '511260.XSHG', '515160.XSHG'],  # 收入组合：债券类ETF
    'growth': ['159941.XSHE', '513100.XSHG', '512480.XSHG', '515030.XSHG'],  # 成长组合
}

# 参数初始化函数 - 所有参数都在这里集中设置
def initialize_parameters():
    """
    初始化所有参数，方便集中管理和调整
    """
    params = {
        # 基本参数
        'n_etfs': 10,                    # 选择ETF数量
        'correlation_threshold': 0.8,    # 相关度阈值
        'top_n_candidates': 10,          # 从相关度最低的top N中选择规模最大的[3,4](@ref)
        
        # 时间参数
        'start_date': '2015-01-01',      # 回测开始日期
        'end_date': '2025-01-01',        # 回测结束日期
        'volume_days': 30,               # 成交量计算周期
        'min_data_days': 250,            # 最小数据天数
        'min_establish_days': 365,       # 成立时间要求
        
        # 筛选参数
        'volume_percentile': 50,         # 成交量排名百分比
        'min_correlation': -1.0,         # 最小相关度要求
        'max_correlation': 1.0,          # 最大相关度要求
        'max_price_threshold': 50,       # 最大价格阈值（元）
        
        # 自定义ETF参数
        'custom_etfs': [],               # 自定义必须包含的ETF列表
        'use_template': 'default',       # 使用模板
        'force_gold_first': True,        # 是否强制黄金ETF作为第一只
        
        # 可视化参数
        'plot_correlation': True,        # 是否绘制相关度矩阵
        'plot_returns': True,            # 是否绘制收益率曲线
        'plot_size': (12, 8),            # 图形大小
        'color_palette': 'coolwarm',     # 颜色主题：coolwarm效果好点
        
        # 高级参数
        'use_log_returns': True,         # 是否使用对数收益率
        'data_alignment_threshold': 0.8, # 数据对齐阈值(80%)
        'enable_progress_log': True,     # 是否启用进度日志
    }
    return params

# 主函数：构建低相关度ETF组合
def build_low_correlation_etf_portfolio(params):
    """
    改进版低相关度ETF组合构建，使用统一的参数字典
    新选择逻辑：先取相关度和最小的10只ETF，再从这10只中选择规模最大的一只[3,4](@ref)
    """
    # 从参数字典中提取参数
    n_etfs = params['n_etfs']
    correlation_threshold = params['correlation_threshold']
    start_date = params['start_date']
    end_date = params['end_date']
    volume_days = params['volume_days']
    min_data_days = params['min_data_days']
    volume_percentile = params['volume_percentile']
    min_establish_days = params['min_establish_days']
    enable_progress_log = params['enable_progress_log']
    top_n_candidates = params['top_n_candidates']
    max_price_threshold = params['max_price_threshold']
    custom_etfs = params['custom_etfs']
    use_template = params['use_template']
    force_gold_first = params['force_gold_first']
    
    if enable_progress_log:
        print("开始构建低相关度ETF组合...")
        print("=" * 60)
        print("当前参数设置:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        print("=" * 60)
    
    # 确定自定义必须包含的ETF
    forced_etfs = set()
    
    # 1. 添加模板中的ETF
    if use_template != 'default' and use_template in ETF_TEMPLATES:
        template_etfs = ETF_TEMPLATES[use_template]
        if template_etfs:
            forced_etfs.update(template_etfs)
            if enable_progress_log:
                print(f"使用模板: {use_template}，包含{len(template_etfs)}只ETF:")
                for etf in template_etfs:
                    print(f"  - {etf}")
    
    # 2. 添加自定义的ETF
    if custom_etfs:
        forced_etfs.update(custom_etfs)
        if enable_progress_log:
            print(f"添加自定义ETF {len(custom_etfs)}只:")
            for etf in custom_etfs:
                print(f"  - {etf}")
    
    # 强制包含的ETF总数
    if forced_etfs and enable_progress_log:
        print(f"总共强制包含{len(forced_etfs)}只ETF")
    
    # 获取市场所有ETF
    all_etf_df = get_all_securities(types=['etf'], date=end_date)
    market_etfs = all_etf_df.index.tolist()
    
    if enable_progress_log:
        print(f"市场ETF总数: {len(market_etfs)}")
    
    # 合并强制包含的ETF和市场ETF（去重）
    etf_pool = list(set(forced_etfs) | set(market_etfs))
    
    if enable_progress_log:
        print(f"初始ETF池总数（强制包含ETF+市场ETF）: {len(etf_pool)}")
    
    # 获取ETF名称映射
    name_mapping = {}
    for etf in etf_pool:
        if etf in all_etf_df.index:
            name_mapping[etf] = all_etf_df.loc[etf, 'display_name']
        else:
            name_mapping[etf] = etf  # 如果找不到名称，使用代码本身
    
    # 第一步：过滤成立时间不足的ETF（强制包含的ETF跳过此步骤）
    if enable_progress_log:
        print(f"\n第一步：过滤成立时间不足{min_establish_days}天的ETF...")
    
    established_etfs = []
    skipped_forced_etfs = []  # 记录被跳过的强制包含ETF
    
    for etf in etf_pool:
        # 如果是强制包含的ETF，直接加入
        if etf in forced_etfs:
            established_etfs.append(etf)
            continue
            
        # 获取ETF上市日期
        start_date_obj = pd.Timestamp(start_date)
        end_date_obj = pd.Timestamp(end_date)
        
        # 获取ETF的历史数据来判断成立时间
        try:
            # 尝试获取更早的数据来判断成立时间
            early_start = start_date_obj - pd.Timedelta(days=min_establish_days + 100)
            prices = get_price(etf, start_date=early_start, end_date=end_date_obj, 
                              fields=['close'], skip_paused=True)
            
            if len(prices) >= min_establish_days:
                established_etfs.append(etf)
        except:
            continue
    
    etf_pool = established_etfs
    if enable_progress_log:
        print(f"成立时间达标ETF数量: {len(etf_pool)}")
    
    # 第二步：过滤数据不完整的ETF（要求至少有min_data_days天的数据）
    if enable_progress_log:
        print(f"\n第二步：过滤数据不足{min_data_days}天的ETF...")
    
    valid_etfs = []
    for etf in etf_pool:
        # 如果是强制包含的ETF，直接加入
        if etf in forced_etfs:
            valid_etfs.append(etf)
            continue
            
        prices = get_price(etf, start_date=start_date, end_date=end_date, 
                          fields=['close'], skip_paused=True)
        if len(prices) >= min_data_days:
            valid_etfs.append(etf)
    
    etf_pool = valid_etfs
    if len(etf_pool) < n_etfs:
        n_etfs = len(etf_pool)
        if enable_progress_log:
            print(f"警告：可用ETF数量不足，将选择{n_etfs}只ETF")
    
    if enable_progress_log:
        print(f"数据完整ETF数量: {len(etf_pool)}")
    
    # 第三步：过滤成交量排名前volume_percentile%的ETF
    if enable_progress_log:
        print(f"\n第三步：过滤成交量排名前{volume_percentile}%的ETF...")
    
    volume_dict = {}
    for etf in etf_pool:
        # 获取最近volume_days天的成交量数据
        end_dt = pd.Timestamp(end_date)
        start_dt = end_dt - pd.Timedelta(days=volume_days)
        prices = get_price(etf, start_date=start_dt, end_date=end_dt, 
                          fields=['volume'], skip_paused=True)
        if len(prices) > 0:
            avg_volume = prices['volume'].mean()
            volume_dict[etf] = avg_volume
        else:
            volume_dict[etf] = 0
    
    # 计算成交量阈值（前volume_percentile%）
    if volume_dict:
        # 只对非强制包含的ETF计算阈值
        non_forced_volumes = [v for k, v in volume_dict.items() if k not in forced_etfs]
        if non_forced_volumes:  # 确保有非强制包含的ETF来计算阈值
            volume_threshold = np.percentile(non_forced_volumes, 100 - volume_percentile)
            high_volume_etfs = []
            for etf in volume_dict:
                if etf in forced_etfs or volume_dict[etf] >= volume_threshold:
                    high_volume_etfs.append(etf)
            etf_pool = high_volume_etfs
            if enable_progress_log:
                print(f"高成交量ETF数量: {len(etf_pool)}")
        else:
            if enable_progress_log:
                print("所有ETF都是强制包含的，跳过成交量过滤")
    else:
        raise ValueError("没有ETF的成交量数据可用")
    
    if len(etf_pool) < n_etfs:
        n_etfs = len(etf_pool)
        if enable_progress_log:
            print(f"警告：过滤后ETF数量不足，将选择{n_etfs}只ETF")
    
    # 第四步：过滤价格高于阈值的ETF
    if enable_progress_log:
        print(f"\n第四步：过滤价格高于{max_price_threshold}元的ETF...")
    
    # 获取当前价格数据
    price_filtered_etfs = []
    price_data_dict = {}  # 存储价格数据
    
    for etf in etf_pool:
        try:
            # 获取最新价格
            prices = get_price(etf, count=1, end_date=end_date, 
                             fields=['close'], skip_paused=True)
            if len(prices) > 0:
                current_price = prices['close'].iloc[-1]
                price_data_dict[etf] = current_price
                
                # 如果是强制包含的ETF，不受价格限制
                if etf in forced_etfs:
                    price_filtered_etfs.append(etf)
                    if enable_progress_log:
                        print(f"  强制包含ETF {etf} ({name_mapping[etf]}) 价格: {current_price:.2f}元 (不受价格限制)")
                # 检查价格是否超过阈值
                elif current_price <= max_price_threshold:
                    price_filtered_etfs.append(etf)
                else:
                    if enable_progress_log:
                        print(f"  过滤 {etf} ({name_mapping[etf]}): 价格{current_price:.2f}元 > {max_price_threshold}元")
        except Exception as e:
            # 如果获取价格失败，且是强制包含的ETF，仍然保留
            if etf in forced_etfs:
                price_filtered_etfs.append(etf)
                if enable_progress_log:
                    print(f"  强制包含ETF {etf} ({name_mapping[etf]}) 价格获取失败，但仍保留")
            continue
    
    etf_pool = price_filtered_etfs
    if enable_progress_log:
        print(f"价格过滤后ETF数量: {len(etf_pool)}")
    
    if len(etf_pool) < n_etfs:
        n_etfs = len(etf_pool)
        if enable_progress_log:
            print(f"警告：价格过滤后ETF数量不足，将选择{n_etfs}只ETF")
    
    # 优化：预先获取所有ETF的收益率数据，避免重复获取
    if enable_progress_log:
        print(f"\n预先获取{len(etf_pool)}只ETF的{min_data_days}日收益率数据...")
    
    returns_data = {}
    price_data = {}  # 存储价格数据用于后续绘图
    valid_etfs_final = []
    
    # 使用更高效的数据获取方式
    for etf in etf_pool:
        try:
            # 获取价格数据
            prices = get_price(etf, count=min_data_days, end_date=end_date, 
                             fields=['close'], skip_paused=True)
            if len(prices) >= min_data_days:
                # 计算日收益率
                if params['use_log_returns']:
                    returns = np.log(prices['close'] / prices['close'].shift(1)).dropna()
                else:
                    returns = prices['close'].pct_change().dropna()
                
                if len(returns) >= min_data_days * params['data_alignment_threshold']:
                    returns_data[etf] = returns
                    price_data[etf] = prices['close']
                    valid_etfs_final.append(etf)
                elif etf in forced_etfs:
                    # 强制包含的ETF即使数据不足也保留
                    returns_data[etf] = returns if len(returns) > 0 else pd.Series([0])
                    price_data[etf] = prices['close']
                    valid_etfs_final.append(etf)
                    if enable_progress_log:
                        print(f"  强制包含ETF {etf} ({name_mapping[etf]}) 数据不足，但仍保留")
        except Exception as e:
            # 强制包含的ETF即使获取数据失败也保留
            if etf in forced_etfs:
                # 创建空的收益率数据
                returns_data[etf] = pd.Series([0])
                price_data[etf] = pd.Series([1])
                valid_etfs_final.append(etf)
                if enable_progress_log:
                    print(f"  强制包含ETF {etf} ({name_mapping[etf]}) 数据获取失败，使用空数据")
            continue
    
    etf_pool = valid_etfs_final
    if enable_progress_log:
        print(f"最终有效ETF数量: {len(etf_pool)}")
    
    if len(etf_pool) < n_etfs:
        n_etfs = len(etf_pool)
        if enable_progress_log:
            print(f"最终将选择{n_etfs}只ETF")
    
    # 初始化已选ETF列表：首先加入强制包含的ETF
    selected_etfs = []
    
    # 如果强制包含黄金ETF作为第一只，且黄金ETF在ETF池中
    gold_etf_code = '518880.XSHG'
    if force_gold_first and gold_etf_code in etf_pool and gold_etf_code not in selected_etfs:
        selected_etfs.append(gold_etf_code)
        if enable_progress_log:
            print(f"\n第一轮强制选择黄金ETF: {gold_etf_code} ({name_mapping.get(gold_etf_code, gold_etf_code)})")
    
    # 加入其他强制包含的ETF（但排除黄金ETF，如果已经加入了）
    for etf in forced_etfs:
        if etf in etf_pool and etf not in selected_etfs:
            selected_etfs.append(etf)
            if enable_progress_log and etf != gold_etf_code:
                print(f"强制包含ETF: {etf} ({name_mapping.get(etf, etf)})")
    
    # 剩余待选ETF（排除已选的）
    remaining_etfs = [etf for etf in etf_pool if etf not in selected_etfs]
    
    # 如果还没有选择任何ETF（比如没有强制包含的ETF），则选择成交量最大的ETF
    if not selected_etfs and etf_pool:
        # 从etf_pool中选择成交量最大的
        valid_volume_dict = {k: v for k, v in volume_dict.items() if k in etf_pool}
        if valid_volume_dict:
            first_etf = max(valid_volume_dict, key=valid_volume_dict.get)
        else:
            first_etf = etf_pool[0]
        
        selected_etfs = [first_etf]
        remaining_etfs = [etf for etf in etf_pool if etf != first_etf]
        
        if enable_progress_log:
            print(f"\n第一轮选择（无强制包含ETF）: {first_etf} ({name_mapping.get(first_etf, first_etf)}) - 成交量最大")
    
    # 优化相关度计算函数（使用预计算的收益率数据）
    def calculate_correlation_fast(etf1, etf2):
        """
        快速计算两只ETF的相关度（使用预计算的收益率数据）
        """
        if etf1 not in returns_data or etf2 not in returns_data:
            return None
        
        returns1 = returns_data[etf1]
        returns2 = returns_data[etf2]
        
        # 对齐数据（确保日期一致）
        common_index = returns1.index.intersection(returns2.index)
        if len(common_index) < min_data_days * params['data_alignment_threshold']:
            return None
        
        ret1_aligned = returns1[common_index]
        ret2_aligned = returns2[common_index]
        
        # 计算相关系数
        correlation = np.corrcoef(ret1_aligned, ret2_aligned)[0, 1]
        
        # 应用相关度过滤
        if correlation < params['min_correlation'] or correlation > params['max_correlation']:
            return None
            
        return correlation
    
    # 计算还需要选择多少只ETF
    etfs_needed = n_etfs - len(selected_etfs)
    
    # 如果还需要选择更多ETF
    if etfs_needed > 0 and remaining_etfs:
        if enable_progress_log:
            print(f"\n已选强制包含ETF {len(selected_etfs)}只，还需要选择 {etfs_needed} 只ETF")
        
        # 迭代选择后续ETF
        for round_num in range(len(selected_etfs) + 1, n_etfs + 1):
            if len(remaining_etfs) == 0:
                if enable_progress_log:
                    print(f"停止：剩余ETF数量为0，已选{len(selected_etfs)}只ETF")
                break
            
            candidate_scores = {}  # 存储候选ETF的相关度和
            candidate_details = {}  # 存储与每个已选ETF的相关度
            
            if enable_progress_log:
                print(f"\n第{round_num}轮筛选：评估{len(remaining_etfs)}只候选ETF...")
            
            for candidate in remaining_etfs:
                # 计算candidate与每个已选ETF的相关度
                correlations = []
                skip_candidate = False
                
                for selected in selected_etfs:
                    corr = calculate_correlation_fast(candidate, selected)
                    if corr is None:
                        skip_candidate = True
                        break
                    correlations.append(corr)
                    
                    # 如果任何单相关度高于阈值，排除该候选
                    if corr > correlation_threshold:
                        skip_candidate = True
                        break
                
                if skip_candidate:
                    continue  # 跳过不符合条件的候选
                
                # 计算相关度之和
                total_correlation = sum(correlations)
                candidate_scores[candidate] = total_correlation
                candidate_details[candidate] = correlations
            
            # 如果没有候选ETF满足条件，终止
            if not candidate_scores:
                if enable_progress_log:
                    print(f"停止：第{round_num}轮无符合条件的ETF，已选{len(selected_etfs)}只ETF")
                break
            
            # 新选择逻辑：先取相关度和最小的top_n_candidates只ETF，再从中选择规模最大的一只[3,4](@ref)
            if enable_progress_log:
                print(f"从{len(candidate_scores)}只候选ETF中选取相关度和最小的{top_n_candidates}只...")
            
            # 使用堆排序获取相关度和最小的top_n_candidates只ETF[1,5](@ref)
            if len(candidate_scores) <= top_n_candidates:
                # 如果候选ETF数量不足top_n_candidates，取全部
                low_correlation_candidates = list(candidate_scores.keys())
            else:
                # 使用堆排序获取相关度和最小的top_n_candidates只ETF[1,5](@ref)
                low_correlation_candidates = heapq.nsmallest(top_n_candidates, candidate_scores, key=candidate_scores.get)
            
            # 从低相关度候选ETF中选择规模最大的一只[3,4](@ref)
            next_etf = max(low_correlation_candidates, key=lambda etf: volume_dict.get(etf, 0))
            
            selected_etfs.append(next_etf)
            remaining_etfs.remove(next_etf)
            
            if enable_progress_log:
                print(f"第{round_num}轮选择: {next_etf} ({name_mapping.get(next_etf, next_etf)})")
                print(f"  相关度和: {candidate_scores[next_etf]:.4f} (在{len(candidate_scores)}只候选ETF中排名第{sorted(candidate_scores.values()).index(candidate_scores[next_etf])+1})")
                volume_val = volume_dict.get(next_etf, 0)
                print(f"  成交量: {volume_val:.0f} (在{top_n_candidates}只低相关度候选ETF中最大)")
                
                # 打印详细相关度
                details = candidate_details[next_etf]
                for i, selected in enumerate(selected_etfs[:-1]):
                    print(f"   与{selected} ({name_mapping.get(selected, selected)})的相关度: {details[i]:.4f}")
    
    # 输出最终结果
    if enable_progress_log:
        print(f"\n最终选出的{len(selected_etfs)}只ETF组合（按选择顺序）:")
        for i, etf in enumerate(selected_etfs, 1):
            etf_name = name_mapping.get(etf, etf)
            etf_price = price_data_dict.get(etf, 'N/A')
            volume_val = volume_dict.get(etf, 0)
            forced_flag = " (强制包含)" if etf in forced_etfs else ""
            print(f"{i}. {etf} ({etf_name}) - 成交量: {volume_val:.0f}, 价格: {etf_price if isinstance(etf_price, str) else f'{etf_price:.2f}元'}{forced_flag}")
    
    # 返回包含代码和名称的完整信息
    etf_info = []
    for etf in selected_etfs:
        etf_info.append({
            'code': etf,
            'name': name_mapping.get(etf, etf),
            'volume': volume_dict.get(etf, 0),
            'price': price_data_dict.get(etf, None),
            'forced': etf in forced_etfs,
            'selection_order': selected_etfs.index(etf) + 1
        })
    
    return selected_etfs, etf_info, returns_data, price_data, params

# 获取ETF名称映射字典
def get_etf_name_mapping(etf_list, end_date):
    """获取ETF代码到名称的映射字典"""
    etf_df = get_all_securities(types=['etf'], date=end_date)
    name_mapping = {}
    for etf in etf_list:
        if etf in etf_df.index:
            name_mapping[etf] = etf_df.loc[etf, 'display_name']
        else:
            name_mapping[etf] = etf  # 如果找不到名称，使用代码本身
    return name_mapping

# 计算并绘制相关度矩阵热力图
def plot_correlation_heatmap(selected_etfs, returns_data, name_mapping, params):
    """绘制ETF组合的相关度矩阵热力图"""
    if not params['plot_correlation']:
        return None
        
    print("\n正在计算相关度矩阵并绘制热力图...")
    
    # 准备收益率数据
    returns_list = []
    valid_etfs = []
    etf_names = []  # 存储ETF的简称用于标签
    
    for etf in selected_etfs:
        if etf in returns_data:
            returns_list.append(returns_data[etf])
            valid_etfs.append(etf)
            # 创建简短的名称标签
            short_name = f"{etf[-6:]}\n{name_mapping.get(etf, etf)[:4]}"
            etf_names.append(short_name)
    
    if len(returns_list) < 2:
        print("ETF数量不足，无法计算相关矩阵")
        return None
    
    # 创建DataFrame并对齐数据
    returns_df = pd.concat(returns_list, axis=1, keys=valid_etfs)
    returns_df = returns_df.dropna()
    
    if len(returns_df) < 10:  # 确保有足够数据点
        print("数据点不足，无法计算可靠的相关矩阵")
        return None
    
    # 计算相关矩阵
    correlation_matrix = returns_df.corr()
    
    # 创建热力图
    plt.figure(figsize=params['plot_size'])
    
    # 使用Seaborn绘制热力图 
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # 掩码上半部分
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt=".3f", 
                cmap=params['color_palette'], 
                center=0,
                vmin=-1, vmax=1,
                square=True, 
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                xticklabels=etf_names,
                yticklabels=etf_names)
    
    plt.title('ETF组合相关度矩阵热力图', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    print("相关度矩阵:")
    print(correlation_matrix.round(4))
    
    # 计算平均相关度（排除对角线）
    matrix_values = correlation_matrix.values
    np.fill_diagonal(matrix_values, np.nan)
    avg_correlation = np.nanmean(matrix_values)
    print(f"\n组合平均相关度: {avg_correlation:.4f}")
    
    return correlation_matrix

# 绘制收益率曲线
def plot_returns_curves(selected_etfs, price_data, name_mapping, params):
    """绘制ETF的收益率曲线"""
    if not params['plot_returns']:
        return
        
    print("\n正在绘制ETF收益率曲线...")
    
    # 创建图形
    plt.figure(figsize=params['plot_size'])
    
    # 为每个ETF选择不同的颜色
    colors = plt.cm.get_cmap(params['color_palette'])(np.linspace(0, 1, len(selected_etfs)))
    
    # 绘制每个ETF的累积收益率曲线
    for i, etf in enumerate(selected_etfs):
        if etf in price_data:
            # 计算累积收益率（归一化到起始点=100）
            prices = price_data[etf]
            cumulative_returns = (prices / prices.iloc[0]) * 100
            
            # 创建标签（代码后6位+名称）
            label = f"{etf[-6:]} {name_mapping.get(etf, etf)}"
            
            plt.plot(cumulative_returns.index, cumulative_returns.values, 
                    label=label, linewidth=2, alpha=0.8, color=colors[i])
    
    plt.title('ETF累积收益率曲线（归一化）', fontsize=16, fontweight='bold')
    plt.xlabel('日期')
    plt.ylabel('累积收益率（起始点=100）')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 绘制日收益率分布图
    plt.figure(figsize=(params['plot_size'][0], params['plot_size'][1]//2))
    
    # 准备日收益率数据
    returns_list = []
    labels = []
    
    for etf in selected_etfs:
        if etf in price_data:
            prices = price_data[etf]
            daily_returns = prices.pct_change().dropna()
            returns_list.append(daily_returns)
            labels.append(f"{etf[-6:]} {name_mapping.get(etf, etf)}")
    
    # 绘制箱线图显示收益率分布
    plt.boxplot(returns_list, labels=labels)
    plt.title('ETF日收益率分布', fontsize=16, fontweight='bold')
    plt.xlabel('ETF')
    plt.ylabel('日收益率')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 执行代码（在聚宽研究环境中直接运行）
if __name__ == "__main__":
    # 导入heapq模块（用于高效获取Top N元素）[1,5](@ref)
    import heapq
    
    # 初始化所有参数
    params = initialize_parameters()
    
    # 可以在这里覆盖默认参数
    params.update({
        'n_etfs': 20,                     # 总共选择8只ETF
        'correlation_threshold': 0.5,    # 相关度阈值
        'min_data_days': 125,            # 最小数据天数
        'end_date': '2024-01-01',        # 回测结束日期
        'volume_percentile': 50,         # 成交量排名前50%
        'top_n_candidates': 5,          # 从相关度最低的10只中选择规模最大的
        'max_price_threshold': 50,       # 价格过滤阈值（元）
        'force_gold_first': True,        # 强制黄金ETF作为第一只
        
        # 自定义ETF参数
        'custom_etfs': ['518880.XSHG', '159941.XSHE', '511220.XSHG'],  # 自定义必须包含的ETF
        'use_template': '',  # 使用模板
    })
    
    # 构建低相关度ETF组合
    selected_etfs, etf_info, returns_data, price_data, params = build_low_correlation_etf_portfolio(params)
    
    # 获取名称映射（用于绘图）
    name_mapping = get_etf_name_mapping(selected_etfs, params['end_date'])
    
    # 绘制相关度矩阵热力图
    correlation_matrix = plot_correlation_heatmap(selected_etfs, returns_data, name_mapping, params)
    
    # 绘制收益率曲线
    plot_returns_curves(selected_etfs, price_data, name_mapping, params)
    
    # 以表格形式显示最终结果
    print("\n" + "=" * 60)
    print("最终ETF组合汇总")
    print("=" * 60)
    for info in etf_info:
        price_str = f"{info['price']:.2f}元" if info['price'] is not None else "N/A"
        forced_flag = " (强制包含)" if info['forced'] else ""
        print(f"{info['selection_order']:2d}. {info['code']} - {info['name']} - 成交量: {info['volume']:.0f} - 价格: {price_str}{forced_flag}")
    
    # 性能统计
    if correlation_matrix is not None:
        matrix_values = correlation_matrix.values
        np.fill_diagonal(matrix_values, np.nan)
        min_corr = np.nanmin(matrix_values)
        max_corr = np.nanmax(matrix_values)
        
        print(f"\n组合相关性统计:")
        print(f"- 平均相关度: {np.nanmean(matrix_values):.4f}")
        print(f"- 最小相关度: {min_corr:.4f}")
        print(f"- 最大相关度: {max_corr:.4f}")
        print(f"- 相关度范围: {max_corr - min_corr:.4f}")
        
        # 评估分散化效果
        avg_corr = np.nanmean(matrix_values)
        if avg_corr < 0.3:
            print("- 分散化效果: 优秀（平均相关度<0.3）")
        elif avg_corr < 0.5:
            print("- 分散化效果: 良好（平均相关度<0.5）")
        else:
            print("- 分散化效果: 一般（平均相关度≥0.5）")