import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class BacktestAnalyzer:
    def __init__(self, registry):
        """
        registry: dict or str (path to registry file)
        """
        if isinstance(registry, str):
            if os.path.exists(registry):
                with open(registry, 'r', encoding='utf-8') as f:
                    self.registry = json.load(f)
            else:
                self.registry = {}
        else:
            self.registry = registry

    def _print_table_title(self, title):
        width = 80
        print("\n" + "="*width)
        print(f"{title.center(width)}")
        print("="*width)

    def compare_results(self, strategy_names=None):
        """
        对比回测结果 (实时获取指标)
        strategy_names: list, 要对比的策略名称列表 (e.g. ['wy03']). None=全部
        """
        if 'get_backtest' not in globals():
            print("[Error] 'get_backtest' function not found. Cannot fetch metrics.")
            return None

        # ... (Fetch Data Logic remains unchanged) ...
        data = []
        print("\nFetching metrics for comparison...")
        
        for task_hash, record in self.registry.items():
            # 1. 基本过滤
            bts_id = record.get('id')
            if not bts_id or record.get('status') != 'done':
                continue
            
            # 2. 策略名称过滤
            s_name = record.get('strategy')
            if strategy_names and s_name not in strategy_names:
                continue
            
            # 3. 动态获取指标
            try:
                bt = get_backtest(bts_id)
                metrics = bt.get_risk()
                if not metrics:
                    continue
                    
                row = {
                    'Test Name': record.get('test_name', 'unnamed'),
                    'Strategy': s_name,
                    'Return': f"{metrics.get('algorithm_return', 0):.2%}",
                    'Volatility': f"{metrics.get('algorithm_volatility', 0):.2%}",
                    'Ann. Return': f"{metrics.get('annual_algo_return', 0):.2%}",
                    'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                    'Max Drawdown Period': f"{metrics.get('max_drawdown_period', 'N/A')}",
                    'Alpha': f"{metrics.get('alpha', 0):.2f}",
                    'Beta': f"{metrics.get('beta', 0):.2f}",
                    'Sharpe': f"{metrics.get('sharpe', 0):.2f}",
                    'Sortino': f"{metrics.get('sortino', 0):.2f}",
                    'IF.': f"{metrics.get('information', 0):.2f}",
                    'Lose Count': f"{metrics.get('lose_count', 0)}",
                    'Win Count': f"{metrics.get('win_count', 0)}",
                    'Win Rate': f"{metrics.get('win_ratio', 0):.2%}",
                    'Turnover Rate': f"{metrics.get('turnover_rate', 0):.2f}",
                    'Avg. Trade Return': f"{metrics.get('avg_trade_return', 0):.2%}",
                    'Avg. Position Days': f"{metrics.get('avg_position_days', 0):.2f}",
                }
                data.append(row)
            except Exception as e:
                print(f"Error fetching metrics for {bts_id}: {e}")

        if not data:
            print("No completed backtests found for comparison.")
            return None

        # 4. 生成 DataFrame 并转置
        df_raw = pd.DataFrame(data)
        
        # 定义期望的行顺序 (Metrics)
        row_order = [
            'Return', 'Ann. Return', 'Volatility', 'Max Drawdown', 'Max Drawdown Period', 
            'Alpha', 'Beta', 'Sharpe', 'Sortino', 'IF.', 
            'Lose Count', 'Win Count', 'Win Rate', 'Turnover Rate', 
            'Avg. Trade Return', 'Avg. Position Days'
        ]
        
        # 创建复合标签作为列名，例如 "wy03 (time_1030)"
        df_raw['Label'] = df_raw.apply(lambda x: f"{x['Strategy']} ({x['Test Name']})", axis=1)
        
        # 设置 Label 为索引
        df_raw.set_index('Label', inplace=True)
        
        # 移除不需要转置的元数据列
        if 'Strategy' in df_raw.columns:
            del df_raw['Strategy']
        if 'Test Name' in df_raw.columns:
            del df_raw['Test Name']
            
        # 转置
        df_t = df_raw.T
        
        # 尝试按指定顺序排序行 (如果存在)
        existing_rows = [r for r in row_order if r in df_t.index]
        other_rows = [r for r in df_t.index if r not in row_order]
        df_t = df_t.reindex(existing_rows + other_rows)
        
        self._print_table_title(f"BACKTEST COMPARISON ({len(data)} tests)")
        print(df_t.to_string())
        print("="*80 + "\n")
        
        return df_t

    def plot_curves(self, strategy_names=None, log_scale=False, start_date=None, end_date=None):
        """
        画出收益曲线
        strategy_names: list, 策略名过滤
        log_scale: bool, 是否使用对数坐标
        start_date: str, 开始日期
        end_date: str, 结束日期
        """
        if 'get_backtest' not in globals():
            print("[Error] 'get_backtest' function not found. Cannot fetch results.")
            return

        plt.figure(figsize=(14, 7))
        has_data = False

        for task_hash, record in self.registry.items():
            # 1. 基本过滤
            bts_id = record.get('id')
            if not bts_id or record.get('status') != 'done':
                continue
                
            # 2. 策略名称过滤
            s_name = record.get('strategy')
            if strategy_names and s_name not in strategy_names:
                continue

            try:
                bt = get_backtest(bts_id)
                results = bt.get_results()
                if not results:
                    continue

                # 转换为 DataFrame
                df_res = pd.DataFrame(results)
                # time 字段转为 datetime
                df_res['time'] = pd.to_datetime(df_res['time'])
                df_res.set_index('time', inplace=True)
                
                # 计算净值 (Net Value)
                df_res['net_value'] = df_res['returns'] + 1

                # 过滤日期
                if start_date:
                    original_len = len(df_res)
                    df_res = df_res[df_res.index >= pd.to_datetime(start_date)]
                    # 如果指定了开始日期，且数据不为空，进行归一化处理（以此日为单位1）
                    if not df_res.empty and len(df_res) < original_len:
                         df_res['net_value'] = df_res['net_value'] / df_res['net_value'].iloc[0]

                if end_date:
                    df_res = df_res[df_res.index <= pd.to_datetime(end_date)]

                if df_res.empty:
                    print(f"No data for {s_name} in specified range.")
                    continue
                
                # 绘制收益曲线
                # 使用 test_name 作为 label
                label = f"{s_name}_{record.get('test_name', 'unnamed')}"
                plt.plot(df_res.index, df_res['net_value'], label=label)
                has_data = True
                
            except Exception as e:
                print(f"Error processing plot for {bts_id}: {e}")

        if has_data:
            plt.title('Backtest Equity Curves')
            plt.xlabel('Date')
            
            if log_scale:
                plt.yscale('log')
                plt.ylabel('Net Value (Log Scale)')
            else:
                plt.ylabel('Net Value')
                
            plt.legend(loc='upper left')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.show()
        else:
            print("No data available for plotting.")

    def show_monthly_returns(self, strategy_names=None, mode='individual', table_type='both', plot_type='year'):
        """
        展示月度/年度收益表及热力图
        mode: 'individual' (单独展示) | 'compare' (对比展示)
        table_type: 'table' (仅年度) | 'month' (仅月度) | 'both' (两者都输出) | 'none' (不输出) - 除 none 外，其他三个选项仅在 compare 模式有效，none 对于所有mode均有效
        plot_type: 'year' (仅年度) | 'none' (不画) - year 仅在 compare 模式有效，none 对于所有mode均有效
        """
        if 'get_backtest' not in globals():
            print("[Error] 'get_backtest' function not found.")
            return

        target_records = list()
        for task_hash, record in self.registry.items():
            bts_id = record.get('id')
            if not bts_id or record.get('status') != 'done':
                continue
            
            s_name = record.get('strategy')
            if strategy_names and s_name not in strategy_names:
                continue
            target_records.append((task_hash, record))
        
        if not target_records:
            print("No valid backtests found.")
            return

        if mode == 'compare':
            self._show_compare_mode(target_records, table_type, plot_type)
        else:
            self._show_individual_mode(target_records, table_type, plot_type)

    def _fetch_monthly_data(self, bts_id, full_name):
        try:
            bt = get_backtest(bts_id)
            risks = bt.get_period_risks()
            if not risks or 'algorithm_return' not in risks:
                return None
                
            df_src = risks['algorithm_return']
            if 'one_month' not in df_src.columns:
                return None
            
            return df_src['one_month']
        except Exception as e:
            print(f"Error fetching data for {full_name}: {e}")
            return None

    def _calculate_yearly_returns(self, monthly_series):
        """
        根据月度收益计算年度收益
        Input: pd.Series (index=Date, values=MonthlyReturn)
        Output: pd.Series (index=Year, values=YearlyReturn)
        """
        if monthly_series is None or monthly_series.empty:
            return pd.Series(dtype=float)
            
        # 转换为 DataFrame 以便处理
        df = monthly_series.to_frame(name='ret')
        df['Year'] = df.index.map(lambda x: int(x.split('-')[0]))
        
        # Group by Year and compound
        yearly = df.groupby('Year')['ret'].apply(lambda x: np.prod(1 + x) - 1)
        return yearly

    def _show_compare_mode(self, target_records, table_type, plot_type):
        monthly_data = {}
        yearly_data = {}
        
        for task_hash, record in target_records:
            bts_id = record['id']
            s_name = record.get('strategy')
            full_name = f"{s_name} ({record.get('test_name', '')})"
            
            series = self._fetch_monthly_data(bts_id, full_name)
            if series is not None:
                monthly_data[full_name] = series
                yearly_data[full_name] = self._calculate_yearly_returns(series)
        
        if not monthly_data:
            print("No return data available for comparison.")
            return

        # --- Monthly Comparison ---
        if table_type in ['month', 'both']:
            df_compare = pd.DataFrame(monthly_data)
            df_compare.sort_index(inplace=True)
            
            self._print_table_title("MONTHLY RETURN COMPARISON")
            print(df_compare.applymap(lambda x: f"{x:.2%}" if pd.notnull(x) else "").to_string())
            print("="*80 + "\n")
            
        # --- Yearly Comparison ---
        if table_type in ['table', 'both'] or plot_type == 'year':
            df_yearly = pd.DataFrame(yearly_data)
            df_yearly.sort_index(inplace=True)
            
            if table_type in ['table', 'both']:
                self._print_table_title("YEARLY RETURN COMPARISON")
                print(df_yearly.applymap(lambda x: f"{x:.2%}" if pd.notnull(x) else "").to_string())
                print("="*80 + "\n")
            
            if plot_type == 'year':
                try:
                    height = len(df_yearly) * 0.8
                    height = max(height, 4)
                    ax = df_yearly.plot(kind='barh', figsize=(12, height), width=0.8)
                    plt.title('Yearly Return Comparison')
                    plt.xlabel('Return')
                    plt.ylabel('Year')
                    plt.grid(True, axis='x', alpha=0.5)
                    plt.axvline(0, color='black', linewidth=0.8)
                    plt.gca().invert_yaxis()
                    #plt.legend(loc='best')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    # plt.xticks(rotation=0) # Not needed for numeric/barh
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Error plotting yearly comparison: {e}")

    def _show_individual_mode(self, target_records, table_type, plot_type):
        for task_hash, record in target_records:
            bts_id = record['id']
            s_name = record.get('strategy')
            full_name = f"{s_name} ({record.get('test_name', '')})"

            series = self._fetch_monthly_data(bts_id, full_name)
            if series is None:
                continue
            
            try:
                # Pivot and Calculate
                df_calc = series.to_frame(name='one_month')
                df_calc['Year'] = df_calc.index.map(lambda x: int(x.split('-')[0]))
                df_calc['Month'] = df_calc.index.map(lambda x: int(x.split('-')[1]))
                
                pivot = df_calc.pivot(index='Year', columns='Month', values='one_month')
                
                # Yearly Calc
                yearly_series = self._calculate_yearly_returns(series)
                pivot['Yearly'] = yearly_series
                
                # Print Table
                if table_type != 'none':
                    self._print_table_title(f"Monthly Returns: {full_name}")
                    
                    print_df = pivot.copy()
                    for c in print_df.columns:
                        print_df[c] = print_df[c].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                    
                    for m in range(1, 13):
                        if m not in print_df.columns:
                                print_df[m] = ""
                    
                    cols = list(range(1, 13)) + ['Yearly']
                    print_df = print_df[cols]
                    print(print_df.to_string())
                
                # Plot Heatmap
                if plot_type != 'none':
                    self._plot_monthly_heatmap(pivot, full_name)
                    
            except Exception as e:
                print(f"Error processing individual view for {bts_id}: {e}")

    def _plot_monthly_heatmap(self, pivot_df, title):
        """
        绘制月度收益热力图
        """
        # 准备数据 (Drop Yearly column for heatmap)
        data = pivot_df.drop(columns=['Yearly'], errors='ignore')
        
        # 补全 1-12 月以便对齐
        for m in range(1, 13):
            if m not in data.columns:
                data[m] = np.nan
        data = data[sorted(data.columns)] # Sort 1..12
        
        years = data.index.tolist()
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        values = data.values # shape (n_years, 12)
        
        fig, ax = plt.subplots(figsize=(10, len(years) * 0.8 + 2))
        
        # Plot Heatmap
        # 使用 RdYlGn (红绿) colormap, center=0
        # 为了让 0 显示为白色/中性，通常需要自定义 norm 或 cmap range
        # 这里简单使用 RdYlGn
        im = ax.imshow(values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
        
        # Label Ticks
        ax.set_xticks(np.arange(len(months)))
        ax.set_xticklabels(months)
        ax.set_yticks(np.arange(len(years)))
        ax.set_yticklabels(years)
        
        # Rotated x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(years)):
            for j in range(len(months)):
                val = values[i, j]
                if pd.notnull(val):
                    text = ax.text(j, i, f"{val:.1%}",
                                   ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(f"Monthly Returns - {title}")
        fig.tight_layout()
        plt.colorbar(im, ax=ax, label='Return')
        plt.show()