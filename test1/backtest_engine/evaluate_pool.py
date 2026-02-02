from jqdata import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PoolEvaluator:
    def __init__(self, backtest_id, etf_list, price_field='close'):
        """
        初始化评估器 (Backtest集成版)
        :param backtest_id: str, 回测ID (from create_backtest)
        :param etf_list: list, 标的池代码列表
        :param price_field: str, 使用的价格字段
        """
        print(f"正在加载回测数据 ID: {backtest_id}...")
        self.gt = get_backtest(backtest_id)
        
        # 1. 获取策略收益 (Strategy Returns)
        # get_results returns list of dicts: [{'time':..., 'returns':...}, ...]
        results_list = self.gt.get_results()
        df_results = pd.DataFrame(results_list)
        df_results['time'] = pd.to_datetime(df_results['time'])
        df_results.set_index('time', inplace=True)
        
        # 'returns' 是累计收益，需要转换为日收益率 (pct_change)
        # 注意: get_results 的 returns 是 "Cumulative Return" (如 0.01, 0.05...)
        # 但通常我们需要 "Daily Return"。 
        # API文档说 "return" 是 "将初期投入视为单位1计算的累积收益"。
        # 所以 Daily Return = (1 + Current_Cum) / (1 + Prev_Cum) - 1
        self.strategy_total_ret = df_results['returns']
        self.strategy_ret = (1 + self.strategy_total_ret).pct_change().fillna(0)
        
        # 修正第一天的收益 (如果是第一天就有收益，pct_change会由于shift为NaN而丢失)
        # 第一天的日收益 = 第一天的累计收益
        if len(self.strategy_ret) > 0:
            self.strategy_ret.iloc[0] = self.strategy_total_ret.iloc[0]

        # 归一化日期索引 (去除时间，只保留日期)
        self.strategy_ret.index = self.strategy_ret.index.normalize()

        # 2. 获取策略持仓 (Strategy Positions)
        # get_positions returns: [{'time':..., 'security':..., 'amount':...}, ...]
        # 默认返回所有历史持仓
        pos_list = self.gt.get_positions()
        if pos_list:
            df_pos = pd.DataFrame(pos_list)
            df_pos['time'] = pd.to_datetime(df_pos['time']).dt.normalize()
            
            # 每天可能持有多个（但在轮动策略通常只有1个主要持仓）
            # 筛选出 amount > 0 的 (非空仓)
            df_pos = df_pos[df_pos['amount'] > 0]
            
            # 简化：每天取仓位最重的一个 (或者简单取第一个，视策略而定)
            # 这里假设是单标的轮动
            daily_holdings = df_pos.sort_values('amount', ascending=False).groupby('time')['security'].first()
            self.strategy_positions = daily_holdings.reindex(self.strategy_ret.index).fillna("CASH")
        else:
            self.strategy_positions = pd.Series("CASH", index=self.strategy_ret.index)

        # 3. 获取标的池数据 (Pool Data)
        # 用策略收益的起止时间来获取行情
        # 注意：如果 strategy_ret 有缺失（如空仓期没记录），range 可能短
        # 但通常 Backtest 是连续日期的。如果不连续，我们以获取到的 Price Index 为主 (Market Calendar)
        start_date = self.strategy_ret.index[0]
        end_date = self.strategy_ret.index[-1]
        
        print(f"正在获取 {len(etf_list)} 个标的池资产行情...")
        raw_data = get_price(list(etf_list), start_date=start_date, end_date=end_date, frequency='daily', fields=[price_field])
        
        # 处理 JQData 返回格式
        if hasattr(raw_data, 'to_frame'): 
            self.pool_prices = raw_data[price_field]
        elif isinstance(raw_data, pd.DataFrame):
             # 简化处理，假设是 Panel 转的 DF 或直接是 DF
             if 'code' in raw_data.columns:
                 self.pool_prices = raw_data.pivot(index='time', columns='code', values=price_field)
             else:
                 self.pool_prices = raw_data
        
        # 确保日期对齐：以 Pool Index (交易所日历) 为准
        self.pool_prices.index = pd.to_datetime(self.pool_prices.index).normalize()
        
        # 强制对齐：Backtest 数据缺失的日期 (空仓 gap) 填补为 0 / CASH
        # intersection 可能会把 gap 扔掉，所以我们这里用 pool_prices.index 来 reindex strategy
        market_index = self.pool_prices.index
        
        # Clip strategy data to market range
        # (Start/End might vary slightly, so we intersect first to clean bounds, then reindex to fill internal gaps)
        # 实际上，只要取 Range 内的所有交易日即可
        
        # Reindex Strategy Returns: 填充 0 (假设 gap 是无收益的空仓)
        self.strategy_ret = self.strategy_ret.reindex(market_index).fillna(0.0)
        
        # Reindex Strategy Positions: 填充 "CASH" (假设 gap 是空仓)
        self.strategy_positions = self.strategy_positions.reindex(market_index).fillna("CASH")
        
        # 再次确保对齐 (理论上已经齐了)
        common_index = self.strategy_ret.index.intersection(self.pool_prices.index)
        
        self.strategy_ret = self.strategy_ret.loc[common_index]
        self.strategy_positions = self.strategy_positions.loc[common_index]
        self.pool_prices = self.pool_prices.loc[common_index]
        
        # 4. 计算池内指标
        self.pool_rets = self.pool_prices.pct_change().fillna(0)
        self.pool_index_ret = self.pool_rets.mean(axis=1)
        
        # 5. 计算净值
        self.strategy_nav = (1 + self.strategy_ret).cumprod()
        self.pool_index_nav = (1 + self.pool_index_ret).cumprod()

    def evaluate_rolling_returns(self, window=20, fig=False):
        """
        方法1: 滚动收益对比 (Rolling Return Comparison)
        计算策略与池内所有资产的N日滚动收益，并分析策略处于排名的百分位。
        """
        # 计算滚动收益
        strategy_roll = self.strategy_ret.rolling(window).apply(lambda x: (1+x).prod() - 1)
        pool_roll = self.pool_rets.rolling(window).apply(lambda x: (1+x).prod() - 1)
        
        # 合并数据
        combined = pool_roll.copy()
        combined['Strategy'] = strategy_roll
        
        # 计算策略在每天的排名 (pct=True 返回百分比排名)
        # 排名越高越好 (1.0 = best, 0.0 = worst)
        ranks = combined.rank(axis=1, pct=True, ascending=True)
        strategy_rank = ranks['Strategy']
        
        print(f"\n[滚动收益分析 (N={window}日)]")
        print(f"平均排名百分位: {strategy_rank.mean():.2%}")
        print(f"位列前25%的时间占比: {(strategy_rank >= 0.75).mean():.2%}")
        print(f"位列后25%的时间占比: {(strategy_rank <= 0.25).mean():.2%}")
        if fig:
            self.plot_rolling_returns(window)
        
        return strategy_rank

    def plot_rolling_returns(self, window=20):
        """
        方法1绘图: 滚动收益可视化
        绘制两个子图:
        1. 滚动收益对比: 策略 vs 标的池(最大/最小/平均)
        2. 排名分位数: 策略在池子中的排名 (1.0=最优)
        """
        # 计算数据
        strategy_roll = self.strategy_ret.rolling(window).apply(lambda x: (1+x).prod() - 1)
        pool_roll = self.pool_rets.rolling(window).apply(lambda x: (1+x).prod() - 1)
        
        # 计算标的池统计量
        pool_mean = pool_roll.mean(axis=1)
        pool_max = pool_roll.max(axis=1)
        pool_min = pool_roll.min(axis=1)
        
        # 计算排名
        combined = pool_roll.copy()
        combined['Strategy'] = strategy_roll
        ranks = combined.rank(axis=1, pct=True, ascending=True)['Strategy']

        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # 子图1: 收益率对比
        ax1.plot(strategy_roll, label='Strategy', color='red', linewidth=2)
        ax1.plot(pool_mean, label='Pool Average', color='gray', linestyle='--')
        # 绘制标的池范围 (Min - Max)
        ax1.fill_between(pool_roll.index, pool_min, pool_max, color='gray', alpha=0.2, label='Pool Range (Min-Max)')
        
        ax1.set_title(f'{window}-Day Rolling Returns Comparison')
        ax1.set_ylabel('Rolling Return')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # 子图2: 排名百分位
        ax2.plot(ranks, label='Strategy Rank Percentile', color='blue')
        ax2.axhline(0.75, color='green', linestyle=':', label='Top 25%')
        ax2.axhline(0.50, color='orange', linestyle=':', label='Median')
        ax2.axhline(0.25, color='red', linestyle=':', label='Bottom 25%')
        
        ax2.fill_between(ranks.index, 0.75, 1.0, color='green', alpha=0.1) # 优秀区
        ax2.fill_between(ranks.index, 0, 0.25, color='red', alpha=0.1)     # 差区
        
        ax2.set_title(f'Strategy Rank Percentile (1.0=Best)')
        ax2.set_ylabel('Percentile')
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc='lower right')
        ax2.grid(True)
        
        plt.tight_layout()
        print(f"\n[滚动绘图] 已生成 {window}日 滚动收益分析图。")

    def evaluate_holding_attribution(self):
        """
        方法2: 持仓周期归因 (Holding Period Attribution)
        自动使用 self.strategy_positions 进行分析
        """
        strategy_positions = self.strategy_positions
        
        # 识别持仓变化的节点，分割为不同的持仓段
        # shift(1) != current
        groups = (strategy_positions != strategy_positions.shift()).cumsum()
        
        results = []
        
        for g_id, group_data in strategy_positions.groupby(groups):
            start_date = group_data.index[0]
            end_date = group_data.index[-1]
            asset_held = group_data.iloc[0]
            
            # 获取该时间段内的区间收益
            # 使用 pct_change 数据计算累积收益: (1+r1)*(1+r2)... - 1
            # 这样能包含 start_date 当天的收益
            period_pool_rets = self.pool_rets.loc[start_date:end_date]
            if len(period_pool_rets) < 1:
                continue
            
            # 池内各资产的区间收益
            period_ret = (1 + period_pool_rets).prod() - 1
            
            # 策略持仓收益 Calculation
            if asset_held == "CASH":
                # 计算策略在此期间的实际收益 (通常接近0或货币基金收益)
                # 使用 cumprod 计算区间复合收益
                strat_period_trace = (1 + self.strategy_ret.loc[start_date:end_date]).cumprod()
                # 如果区间为空或只有1天，trace可能只有1个值
                if len(strat_period_trace) > 0:
                     held_ret = strat_period_trace.iloc[-1] - 1
                else:
                     held_ret = 0.0
            else:
                 # 依然使用标的理论收益 (Pure Signal)，也可改为 actual strategy return
                 # 这里保持一致性，使用标的收益
                 if asset_held in period_ret:
                     held_ret = period_ret[asset_held]
                 else:
                     continue

            # 池内最好收益
            best_asset = period_ret.idxmax()
            best_ret = period_ret.max()
            
            # 池内最差收益
            worst_asset = period_ret.idxmin()
            worst_ret = period_ret.min()
            
            # 平均收益
            avg_ret = period_ret.mean()
            
            # 判断是否"赢了"
            # 如果我是 CASH (空仓)，且 CASH > Pool_Avg，则视为防守成功 (Win)
            is_best = (asset_held == best_asset)
            # 如果空仓且所有资产都跌，Best可能是负的，如果 CASH(0) > Best(-2%)，那CASH其实就是 Best
            if asset_held == "CASH" and held_ret > best_ret:
                is_best = True
                
            results.append({
                'Start': start_date,
                'End': end_date,
                'Days': len(group_data),
                'Held_Asset': asset_held,
                'Held_Return': held_ret,
                'Pool_Best': best_asset,
                'Best_Return': best_ret,
                'Pool_Avg': avg_ret,
                'Is_Best': is_best,
                'Beats_Avg': (held_ret > avg_ret)
            })
            
        df_res = pd.DataFrame(results)
        
        if df_res.empty:
            print("\n[持仓归因] 未找到有效的持仓周期。")
            return df_res

        print(f"\n[持仓归因分析]")
        print(f"总持仓段数: {len(df_res)}")
        print(f"命中率 (选中最优标的): {df_res['Is_Best'].mean():.2%}")
        print(f"胜率 (跑赢池均值): {df_res['Beats_Avg'].mean():.2%}")
        print(f"平均超额收益 (vs 池均值): {(df_res['Held_Return'] - df_res['Pool_Avg']).mean():.2%}")
        
        return df_res

        return df_res

    def evaluate_switching_effect(self):
        """
        方法4: 换仓效果分析 (Switching Effectiveness)
        分析每一次换仓 (Asset A -> Asset B) 后，新持仓是否跑赢了旧持仓？
        """
        pos = self.strategy_positions
        # 找到换仓点: 今天持仓 != 昨天持仓
        # shift(1) 是昨天
        trades = pos[pos != pos.shift(1)]
        
        # 移除第一天 (因为没有"前一持仓")
        if len(trades) > 0 and trades.index[0] == self.strategy_ret.index[0]:
            trades = trades.iloc[1:]
            
        switching_results = []
        
        for date, new_asset in trades.items():
            # 获取前一天的持仓 (Old Asset)
            # 注意: pos.shift(1) 在 date 这一天的值就是 "Yesterday's Position"
            old_asset = pos.shift(1).loc[date]
            
            # 确定本次持仓的结束时间 (Next Switch Date)
            # 在 trades 中找到大于 date 的第一个日期
            next_dates = trades.index[trades.index > date]
            if len(next_dates) > 0:
                end_date = next_dates[0] 
                # 持仓期不包括下一次换仓日当天 (当天已经换了) -> 实际上是到 end_date 的前一天
                # 但为了计算简单，且通常日线数据包含当天，我们取 slice: prices.loc[date : end_date]
                # 然后计算区间涨幅。
            else:
                end_date = self.strategy_ret.index[-1]
            
            # 计算区间收益 (使用 pool_rets)
            period_pool_rets = self.pool_rets.loc[date:end_date]
            if len(period_pool_rets) < 1:
                continue
                
            # 计算期间各资产累计涨幅 (Compound Return)
            period_comp_ret = (1 + period_pool_rets).prod() - 1

            # 计算 "新资产" 在此期间的收益
            if new_asset == "CASH":
                ret_new = 0.0 # 简化为0
            elif new_asset in period_comp_ret:
                ret_new = period_comp_ret[new_asset]
            else:
                ret_new = 0.0
                
            # 计算 "旧资产" 在此期间的收益 (如果不卖会怎样?)
            if old_asset == "CASH":
                ret_old = 0.0
            elif old_asset in period_comp_ret:
                ret_old = period_comp_ret[old_asset]
            else:
                ret_old = 0.0
                
            switch_alpha = ret_new - ret_old
            
            switching_results.append({
                'Date': date,
                'Old_Asset': old_asset,
                'New_Asset': new_asset,
                'New_Return': ret_new,
                'Old_Return': ret_old,
                'Switch_Alpha': switch_alpha,
                'Is_Correct': (switch_alpha > 0)
            })
            
        df_switch = pd.DataFrame(switching_results)
        
        if df_switch.empty:
            print("\n[换仓分析] 未发现换仓操作。")
            return df_switch
            
        print(f"\n[换仓效果分析]")
        print(f"总换仓次数: {len(df_switch)}")
        print(f"换仓成功率 (新>旧): {df_switch['Is_Correct'].mean():.2%}")
        print(f"平均换仓Alpha: {df_switch['Switch_Alpha'].mean():.2%}")
        print(f"累计换仓Alpha: {df_switch['Switch_Alpha'].sum():.2%}")
        
        return df_switch

    def plot_relative_strength(self):
        """
        方法3: 相对强弱曲线 (Relative Strength Curve)
        绘制 策略净值 / 标的池等权净值
        """
        rs = self.strategy_nav / self.pool_index_nav
        
        plt.figure(figsize=(12, 6))
        plt.plot(rs, label='Relative Strength (Strategy / Pool Index)')
        plt.axhline(1.0, color='gray', linestyle='--')
        plt.title('Relative Strength Analysis')
        plt.legend()
        plt.grid(True)
        # plt.show() # Uncomment to show locally
        print("\n[相对强弱] 已生成RS曲线 (斜率>0 意味着存在Alpha)。")
        return rs

# ==========================================
# Example Usage (Dummy Data for Demo)
# ==========================================
if __name__ == "__main__":
    # Mocking JQData API for demonstration
    def get_price(security, start_date=None, end_date=None, frequency='daily', fields=None):
        dates = pd.date_range(start=start_date, end=end_date)
        # Random daily returns for pool
        df = pd.DataFrame(np.random.randn(len(dates), len(security))/100 + 1, index=dates, columns=security).cumprod()
        if fields and len(fields) == 1:
             return df
        return df

    # Mocking Backtest Object
    class MockBacktest:
        def get_results(self):
            # Mock Strategy Returns
            dates = pd.date_range(start='2024-01-01', periods=100)
            cum_ret = (np.random.randn(100)/100 + 1).cumprod() - 1
            return [{'time': d, 'returns': r} for d, r in zip(dates, cum_ret)]
        
        def get_positions(self):
            # Mock Positions: Hold '510300.XSHG' for first 50 days, then '510500.XSHG'
            dates = pd.date_range(start='2024-01-01', periods=100)
            pos = []
            for i, d in enumerate(dates):
                sec = '510300.XSHG' if i < 50 else '510500.XSHG'
                pos.append({'time': d, 'security': sec, 'amount': 100})
            return pos

    def get_backtest(bt_id):
        return MockBacktest()

    # Demo
    print("正在运行模拟数据的演示...")
    
    backtest_id = "mock_bt_id_12345"
    assets = ['510300.XSHG', '510500.XSHG']
    
    # Initialize with backtest_id
    evaluator = PoolEvaluator(backtest_id, assets, price_field='open')
    
    # Method 1
    evaluator.evaluate_rolling_returns(window=20,fig=True)
    
    # Method 2 (Auto-fetches positions)
    evaluator.evaluate_holding_attribution()
    
    # Method 4 (Switching Effect)
    evaluator.evaluate_switching_effect()
    
    # Method 3
    rs = evaluator.plot_relative_strength()
