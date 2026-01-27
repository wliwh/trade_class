from jqdata import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
import datetime
import os
from pathlib import Path

# --------------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------------
CONFIG = {
    'end_date': None,          # 计算截止日，None则为最近交易日
    'display_days': 50,        # 画图展示的天数，不大于50
    'p_vmax': 100,             # 热力图显示最大值
    'figsize_width': 16,       # 图片宽度
    'row_height_factor': 0.4,  # 每行数据对应的高度因子，用于动态调整图片高度
    'ma_window': 20,           # 均线窗口大小
    'plot_scheme': 1,          # 画图方案: 1, 2, 3
    'save_data': False,        # 是否保存数据到CSV
    'csv_filename': 'industries_score.csv' # 保存的文件名
}

# --------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

# --------------------------------------------------------------------------------
# Data Fetching & Processing
# --------------------------------------------------------------------------------

def get_bias_df(end_date, display_days):
    """
    计算全市场个股是否处于均线之上 (Close > MA)
    返回: Boolean DataFrame (True/False)
    """
    total_days = display_days + CONFIG['ma_window']
    tr_days = get_trade_days(end_date=end_date, count=total_days)
    
    if len(tr_days) < total_days:
        print(f"Warning: Not enough trading days. Found {len(tr_days)}, needed {total_days}")
        return pd.DataFrame()

    start_date = tr_days[0]
    stock_list = get_all_securities(date=start_date).index.tolist()
    
    # 批量获取价格数据，panel=False 效率更高
    df_price = get_price(stock_list, end_date=end_date, count=total_days, 
                         fields=['close'], panel=False)
    
    if df_price.empty:
        return pd.DataFrame()
        
    df_close = df_price.pivot(index='time', columns='code', values='close')
    
    # 计算均线
    df_ma = df_close.rolling(CONFIG['ma_window']).mean()
    
    # 判断是否大于均线 (截取后 display_days 天)
    df_bias = (df_close > df_ma).iloc[CONFIG['ma_window']:]
    
    return df_bias

def get_indus_energy(df_bias, end_date):
    """
    计算各行业的强势股比例
    优化策略: 使用 GroupBy 替代循环切片
    """
    if df_bias.empty:
        return pd.DataFrame()

    stock_list = df_bias.columns.tolist()
    
    # 1. 获取行业映射 (Stock -> Industry)
    # get_industry 支持批量查询，返回 {stock: {industry_schema: info}}
    # 注意：股票数量较多时，这一步可能需要几秒钟
    industry_info = get_industry(stock_list, date=end_date)
    
    stock_to_ind = {}
    for code, info in industry_info.items():
        # 使用申万一级行业 'sw_l1'
        if 'sw_l1' in info:
            stock_to_ind[code] = info['sw_l1']['industry_name']
        else:
            stock_to_ind[code] = 'Unknown'
            
    # 2. 按行业分组计算均值 (即 True 的比例)
    # axis=1 表示按列(股票)分组
    # mean() * 100 即百分比
    df_ind_pct = df_bias.groupby(stock_to_ind, axis=1).mean() * 100
    
    # 3. 排序列 (优先展示核心周期行业)
    priority_inds = ('银行', '煤炭', '钢铁', '有色')
    all_cols = df_ind_pct.columns.tolist()
    
    sorted_cols = [c for c in all_cols if c.startswith(priority_inds)] + \
                  [c for c in all_cols if not c.startswith(priority_inds)]
                  
    df_final = df_ind_pct[sorted_cols]
    
    # 按日期倒序 (最近的在上面)
    df_final.sort_index(ascending=False, inplace=True)
    
    return df_final

# --------------------------------------------------------------------------------
# Data Saving
# --------------------------------------------------------------------------------

def get_csv_last_date():
    """
    获取CSV文件中最新的日期
    """
    try:
        current_dir = Path(__file__).parent
        save_dir = current_dir.parent / 'data_save'
        csv_path = save_dir / CONFIG['csv_filename']
        
        if not csv_path.exists():
            return None
            
        df_old = pd.read_csv(csv_path, index_col=0)
        if df_old.empty:
            return None
            
        # 假设索引是日期字符串或datetime
        last_date = df_old.index.max()
        return last_date
    except Exception as e:
        print(f"Error reading CSV last date: {e}")
        return None

def calculate_fetch_days(end_date, display_days):
    """
    计算需要获取的数据天数。
    如果开启 save_data，则尝试获取从 CSV 最新日期之后到 end_date 的数据。
    取 (更新所需天数) 和 (展示所需天数) 的最大值。
    """
    days_to_fetch = display_days
    
    if CONFIG['save_data']:
        last_csv_date_str = get_csv_last_date()
        
        if last_csv_date_str:
            # 计算从 CSV 最新日期到 end_date 之间的交易日数量
            # end_date 已经是字符串格式 'YYYY-MM-DD'
            # last_csv_date_str 也是字符串
            
            # 获取两个日期之间的交易日
            # 注意：get_trade_days 包含起止日期
            # 我们需要的是 last_csv_date 之后的数据，所以从 last_csv_date 开始取，然后去掉第一天(如果就是它自己)
            # 或者简单点：获取从 last_csv_date 到 end_date 的所有交易日，然后减去 1 (如果 last_csv_date 本身是交易日)
            
            try:
                # 确保日期格式一致
                trade_days_range = get_trade_days(start_date=last_csv_date_str, end_date=end_date)
                
                if len(trade_days_range) > 0:
                    # 如果 last_csv_date 在范围内，说明它就是第一天，我们需要的是它之后的天数
                    # 实际上我们需要的是：csv里最新是10号，现在是20号。
                    # get_trade_days(10号, 20号) -> [10, 11, ..., 20]
                    # 我们需要更新的是 [11, ..., 20]。
                    # 所以 needed_days = len - 1。
                    # 如果结果是0或负数（比如 csv日期比end_date还新），则不需要额外更新，保持 display_days
                    
                    days_needed_for_update = len(trade_days_range) - 1
                    
                    # 额外的一点冗余防止边界问题，比如 csv 最新日期是非交易日等情况 (虽然不太可能因为是 get_trade_days 出来的)
                    # 但逻辑上 fetch_days 至少要能覆盖 [11...20] 这段时间。
                    # get_bias_df 里用的 get_price(count=total_days) 是往前推。
                    # 所以如果我们需要最近 N 天的数据，total_days 就得是 N。
                    
                    if days_needed_for_update > 0:
                        # 比较：是为了补数据需要的 days 多，还是画图需要的 days 多
                        days_to_fetch = max(days_needed_for_update, display_days)
                        print(f"Update mode: CSV date {last_csv_date_str}. Need {days_needed_for_update} days to update.")
                    else:
                        print(f"CSV is up to date ({last_csv_date_str}).")
            except Exception as e: 
                print(f"Error calculating fetch days: {e}")
                
    return days_to_fetch

def save_to_csv(df_new):
    """
    保存最新数据到 CSV 文件
    逻辑：读取现有 CSV，找到比 CSV 中最新日期还要新的数据，追加并覆盖保存
    """
    # 构造相对路径: ../data_save/filename
    # 假设 sign1.py 在 test1/micro/, 目标在 test1/data_save/
    # Path(__file__).parents[1] 是 test1
    try:
        current_dir = Path(__file__).parent
        save_dir = current_dir.parent / 'data_save'
        
        # 如果目录不存在则创建
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            
        csv_path = save_dir / CONFIG['csv_filename']
        
        if not csv_path.exists():
            print(f"File not found: {csv_path}. Creating new file.")
            df_new.to_csv(csv_path)
            return

        print(f"Updating data in: {csv_path}")
        df_old = pd.read_csv(csv_path, index_col=0)
        
        # 确保索引是 datetime 类型以便比较 (如果不是的话)
        # 通常 CSV 读出来索引是字符串，df_new 索引是 Timestamp
        # 统一转为字符串比较或者 Timestamp 比较
        
        # 简单起见，统一用字符串比较 (ISO格式 YYYY-MM-DD)
        last_date_old = df_old.index.max()
        
        # 筛选出日期大于旧表最新日期的新数据
        # 注意: df_new.index 可能是 Timestamp, 需转换
        df_new_str_index = df_new.copy()
        df_new_str_index.index = df_new_str_index.index.strftime('%Y-%m-%d')
        
        df_to_append = df_new_str_index[df_new_str_index.index > last_date_old]
        
        if df_to_append.empty:
            print("No new data to append.")
            return
            
        print(f"Appending {len(df_to_append)} rows.")
        
        # 对齐列: 确保新数据包含旧数据的所有列，缺失补 NaN
        # 同时也保留新数据特有的列 (如果有)
        
        # 关键修改: 保持原有 CSV 的列顺序
        # 1. 先取旧列
        all_columns = df_old.columns.tolist()
        
        # 2. 再追加新列 (如果有)
        new_cols = [c for c in df_to_append.columns if c not in all_columns]
        all_columns.extend(new_cols)
        
        # 重建 DataFrame 以包含所有列
        df_old = df_old.reindex(columns=all_columns)
        df_to_append = df_to_append.reindex(columns=all_columns)
        
        # 合并: 新数据在旧数据之上 (因为是倒叙排列)
        df_final = pd.concat([df_to_append, df_old], axis=0)
        
        # 再次确保按日期倒序排列 (最新日期在最上面)
        df_final.sort_index(ascending=False, inplace=True)
        
        df_final.to_csv(csv_path)
        print("Save complete.")
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")


# --------------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------------

def display_mkt(df_ind, df_bias, scheme=1):
    """
    根据不同方案展示市场热度
    scheme 1: 各行业的及格率 再汇总 (sum)
    scheme 2: 各行业及格率 + 行业平均值 + 全市场平均值
    scheme 3: 各行业及格率 + 全市场平均值
    """
    df_plot = df_ind.copy()
    p_vmax = 100
    
    if scheme == 1:
        # 方案 1：各行业的及格率 再汇总
        # 假设 df_ind 只包含行业列
        df_plot['总体'] = df_plot.sum(axis=1)
        # vmax 设置为行业数量 * 100，因为是总和
        p_vmax = (len(df_plot.columns) - 1) * 100
        cols_main = df_plot.columns[:-1] # 行业
        col_summary = '总体'
        
    elif scheme == 2:
        # 方案 2：各行业及格率 + 行业平均 + 全市场总体
        # 计算行业平均 (df_ind 是行业百分比)
        df_plot['平均'] = df_plot.mean(axis=1).astype(int)
        # 计算全市场总体 (基于 df_bias 所有股票)
        # 注意: df_bias 的 index 顺序可能和 df_plot 不一致(df_plot已倒序)，需对齐
        market_avg = (df_bias.sum(axis=1) / df_bias.shape[1]) * 100
        df_plot['总体'] = market_avg.reindex(df_plot.index).astype(int)
        
        p_vmax = 100
        cols_main = df_plot.columns[:-2] # 排除 '平均', '总体'
        col_summary = ['平均', '总体']
        
    elif scheme == 3:
        # 方案 3：各行业及格率 + 全市场总体
        market_avg = (df_bias.sum(axis=1) / df_bias.shape[1]) * 100
        df_plot['aver'] = market_avg.reindex(df_plot.index).astype(int)
        
        p_vmax = 100
        cols_main = df_plot.columns[:-1] # 排除 'aver'
        col_summary = 'aver'
    
    else:
        print(f"Unknown scheme: {scheme}")
        return

    # 绘图逻辑
    display_rows = len(df_plot)
    fig_height = max(6, display_rows * CONFIG['row_height_factor'])
    
    fig = plt.figure(figsize=(CONFIG['figsize_width'], fig_height))
    grid = plt.GridSpec(1, 10)
    cmap = sns.diverging_palette(200, 10, as_cmap=True)
    
    # 1. 主热力图 (行业)
    ax1 = fig.add_subplot(grid[:, :-1])
    ax1.xaxis.set_ticks_position('top')
    sns.heatmap(df_plot[cols_main], vmin=0, vmax=100, annot=True, fmt=".0f", cmap=cmap,
                annot_kws={'size': 10}, cbar=False, ax=ax1)
    ax1.set_yticklabels(df_plot.index.strftime('%Y-%m-%d'))
    ax1.set_title(f"Industry Breadth (Scheme {scheme})", y=1.005)
    
    # 2. 汇总热力图 (右侧)
    ax2 = fig.add_subplot(grid[:, -1])
    ax2.xaxis.set_ticks_position('top')
    
    # 处理 col_summary 是列表还是单个字符串
    summary_data = df_plot[col_summary] if isinstance(col_summary, list) else df_plot[[col_summary]]
    
    sns.heatmap(summary_data, vmin=0, vmax=p_vmax,
                annot=True, fmt=".0f", cmap=cmap, annot_kws={'size': 10}, cbar=False, ax=ax2)
    ax2.set_yticklabels([])
    ax2.set_title("Summary", y=1.005)

    plt.tight_layout()
    plt.show()
    
    # 3. 折线图 (仅展示总体趋势)
    plt.figure(figsize=(CONFIG['figsize_width'], 6))
    
    # 确保索引为 DatetimeIndex 以便正确绘图
    # 虽然通常已经是 DatetimeIndex，但为了稳健性转换一次
    plot_dates = pd.to_datetime(df_plot.index)
    
    if 'aver' in df_plot.columns:
        plt.plot(plot_dates, df_plot['aver'], label='Total/Market Avg')
    elif '总体' in df_plot.columns:
        plt.plot(plot_dates, df_plot['总体'], label='Total/Market Avg')
        
    if 'sum' in df_plot.columns:
        plt.plot(plot_dates, df_plot['sum'], label='Industry Avg', linestyle='--')
        
    plt.title(f'Market Breadth Trend (Scheme {scheme})')
    plt.legend()
    plt.grid(True)
    
    # 优化 X 轴显示，防止重叠
    import matplotlib.dates as mdates
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate() # 自动旋转日期标签
    
    plt.show()

# --------------------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------------------

if __name__ == '__main__':
    # 处理截止日期: 如果未指定，则默认为最近的一个交易日
    end_date_cfg = CONFIG['end_date']
    if not end_date_cfg:
        today = datetime.date.today()
        # 获取最近一个完整的交易日 (不包含今天)
        # 即使今天是交易日，数据可能还没收盘，所以取昨天截止
        yesterday = today - datetime.timedelta(days=1)
        trade_days = get_trade_days(end_date=yesterday, count=1)
        if len(trade_days) > 0:
            end_date = trade_days[-1]
            end_date = end_date.strftime('%Y-%m-%d')
        else:
            print("Error: Could not determine the latest trade day.")
            exit()
    else:
        end_date = end_date_cfg
            
    # 计算实际需要获取的天数
    fetch_days = calculate_fetch_days(end_date, CONFIG['display_days'])
    
    print(f"Starting analysis for End Date: {end_date}")
    print(f"Fetch Days: {fetch_days} (Display Days: {CONFIG['display_days']})")
    print(f"Using Plot Scheme: {CONFIG['plot_scheme']}")
    
    # 1.获取基础数据 (根据 fetch_days)
    df_bias_all = get_bias_df(end_date, fetch_days)
    
    # 2.计算行业热度 (基于完整获取的数据)
    if not df_bias_all.empty:
        df_ind_all = get_indus_energy(df_bias_all, end_date)
        
        if not df_ind_all.empty:
            
            # 3. 保存数据 (如果启用)
            # 保存使用的是完整获取的数据 (为了补全 CSV)
            if CONFIG['save_data']:
                df_to_save = df_ind_all.copy()
                market_avg = (df_bias_all.sum(axis=1) / df_bias_all.shape[1]) * 100
                df_to_save['aver'] = market_avg.reindex(df_to_save.index)
                df_to_save['sum'] = df_ind_all.mean(axis=1)
                save_to_csv(df_to_save)
            
            # 4. 展示
            # 画图只使用 CONFIG['display_days']
            # df_ind_all 是按日期倒序排列的 (最新在最上面)
            # 所以取前 display_days 行即可
            
            df_ind_display = df_ind_all.head(CONFIG['display_days'])
            
            # df_bias 也需要切片，因为 display_mkt 里有的 scheme 会用到 df_bias 算市场平均
            # df_bias_all 索引是升序还是降序? 
            # get_bias_df -> df_price -> pivot -> rolling -> iloc
            # get_price 返回的是时间升序 (旧 -> 新)。
            # get_indus_energy 里做了 sort_index(ascending=False)。
            # 所以 df_ind_all 是 降序 (新 -> 旧)。
            # 但 df_bias_all 是 升序 (旧 -> 新)。
            
            # 为了 display_mkt 正确工作 (它里面可能会用 df_bias 算平均然后 reindex 到 df_ind 的索引)，
            # 最好传进去切片后的 df_bias OR 在 display_mkt 里处理。
            # 简单起见，我们把 df_bias_all 切片成最近 display_days 天
            
            df_bias_display = df_bias_all.tail(CONFIG['display_days'])
            
            display_mkt(df_ind_display, df_bias_display, scheme=CONFIG['plot_scheme'])
            
        else:
            print("Error: Industry analysis failed.")
    else:
        print("Error: No market data retrieved.")