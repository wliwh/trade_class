
# 策略文件路径映射
STRATEGY_FILES = {
    'wy03': 'test1/ETFs/ETF_wy03_opt.py',
    'long': 'test1/ETFs/ETF_long_modular.py',
    'yj15': 'test1/ETFs/ETF_yj15_modular.py'
}

def get_empty_placeholders(strategy_file_path):
    """从策略文件中提取所有以 EXECUTION_ 开头的占位符变量"""
    placeholders = {}
    try:
        with open(strategy_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('EXECUTION_') and '=' in line:
                    k, v = line.split('=', 1)
                    placeholders[k.strip()] = v.strip()
    except Exception as e:
        print(f"Error reading {strategy_file_path}: {e}")
    return placeholders

def generate_strategy_code(strategy_file_path, params):
    """
    读取策略文件并替换占位符参数
    params: dict, 例如 {'EXECUTION_TIME_PLACEHOLDER': "'14:30'"}
    """
    new_lines = []
    try:
        with open(strategy_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                # 检查当前行是否是我们要替换的参数定义行
                replaced = False
                if stripped.startswith('EXECUTION_') and '=' in stripped:
                    key = stripped.split('=', 1)[0].strip()
                    if key in params:
                        # 替换为新值
                        new_lines.append(f"{key} = {params[key]}\n")
                        replaced = True
                
                if not replaced:
                    new_lines.append(line)
        return ''.join(new_lines)
    except Exception as e:
        print(f"Error processing {strategy_file_path}: {e}")
        return ""

def run_strategy_backtest(strategy_name, test_name, params, start_date, end_date, initial_cash=100000):
    """
    运行单个回测任务
    """
    if strategy_name not in STRATEGY_FILES:
        print(f"Strategy {strategy_name} not found.")
        return None
        
    file_path = STRATEGY_FILES[strategy_name]
    
    # 1. 获取默认参数作为底稿
    current_params = get_empty_placeholders(file_path)
    
    # 2. 更新本次覆盖的参数
    # 注意：传入的params值必须是合法的Python代码字符串，例如字符串值需要带引号 "'10:30'"
    current_params.update(params)
    
    # 3. 生成新代码
    code = generate_strategy_code(file_path, current_params)
    
    if not code:
        return None
        
    print(f"Creating backtest [{test_name}] for {strategy_name}...")
    
    # 4. 创建回测
    try:
        bt_id = create_backtest(
            algorithm_id=None, # 提供 code 时不需要 algorithm_id，或者随便填一个占位
            start_date=start_date,
            end_date=end_date,
            frequency="day",
            initial_cash=initial_cash,
            code=code,
            name=f"{strategy_name}_{test_name}",
            python_version=3
        )
        print(f"Backtest created: {bt_id}")
        return bt_id
    except Exception as e:
        print(f"Failed to create backtest: {e}")
        return None

# ==========================================
# 批量回测与指标对比功能
# ==========================================
import time
import pandas as pd

def compare_backtests(backtest_configs):
    """
    批量运行回测并对比指标
    backtest_configs: list of dict, 每个 dict包含 run_strategy_backtest 所需参数
    例如:
    [
        {'strategy_name': 'wy03', 'test_name': '10:30', 'params': {...}, 'start_date': ..., 'end_date': ...},
        {'strategy_name': 'wy03', 'test_name': '14:30', 'params': {...}, 'start_date': ..., 'end_date': ...}
    ]
    """
    bt_results = []
    
    # 1. 提交所有回测任务
    print(f"Starting {len(backtest_configs)} backtests...")
    for config in backtest_configs:
        bt_id = run_strategy_backtest(
            strategy_name=config['strategy_name'],
            test_name=config['test_name'],
            params=config['params'],
            start_date=config['start_date'],
            end_date=config['end_date']
        )
        if bt_id:
            bt_results.append({
                'id': bt_id,
                'name': f"{config['strategy_name']}_{config['test_name']}",
                'status': 'running'
            })
            
    if not bt_results:
        print("No backtests started.")
        return

    # 2. 轮询等待完成
    print("\nWaiting for backtests to complete...")
    all_done = False
    while not all_done:
        all_done = True
        running_count = 0
        for task in bt_results:
            if task['status'] in ['done', 'failed', 'canceled']:
                continue
                
            try:
                gt = get_backtest(task['id'])
                status = gt.get_status()
                task['status'] = status
                
                if status == 'running' or status == 'none':
                    all_done = False
                    running_count += 1
                elif status == 'failed':
                    print(f"Task {task['name']} FAILED.")
            except Exception as e:
                print(f"Error checking status for {task['id']}: {e}")
                
        if not all_done:
            print(f"Running: {running_count}/{len(bt_results)}...", end='\r')
            time.sleep(5) # 每5秒检查一次
            
    print("\nAll backtests finished.\n")
    
    # 3. 收集指标并展示
    metrics_list = []
    for task in bt_results:
        if task['status'] != 'done':
            continue
            
        try:
            gt = get_backtest(task['id'])
            risk = gt.get_risk()
            
            # 提取核心指标，参考 test1/ETFs/Test_ETF_backtest.md 文档
            metrics = {
                'Name': task['name'],
                'Annual Return': f"{risk.get('annual_algo_return', 0):.2%}",
                'Total Return': f"{risk.get('algorithm_return', 0):.2%}",
                'Max Drawdown': f"{risk.get('max_drawdown', 0):.2%}",
                'Sharpe': f"{risk.get('sharpe', 0):.2f}",
                'Win Ratio': f"{risk.get('win_ratio', 0):.2%}",
                'Profit/Loss Ratio': f"{risk.get('profit_loss_ratio', 0):.2f}",
                'Alpha': f"{risk.get('alpha', 0):.2f}",
                'Beta': f"{risk.get('beta', 0):.2f}"
            }
            metrics_list.append(metrics)
        except Exception as e:
            print(f"Error getting metrics for {task['name']}: {e}")
            
    # 4. 生成对比表格
    if metrics_list:
        df = pd.DataFrame(metrics_list)
        # 调整列顺序
        cols = ['Name', 'Annual Return', 'Max Drawdown', 'Sharpe', 'Win Ratio', 'Profit/Loss Ratio', 'Total Return', 'Alpha', 'Beta']
        # 确保列存在
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        print("="*80)
        print("STRATEGY PERFORMANCE COMPARISON")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
    else:
        print("No successful backtest results to display.")

def test_run_backtest():
    # 定义要对比的测试配置
    configs = [
        # 测试 wy03 策略在 10:30 交易
        {
            'strategy_name': 'wy03',
            'test_name': 'time_1030',
            'params': {'EXECUTION_TIME_PLACEHOLDER': "'10:30'"},
            'start_date': '2025-01-01',
            'end_date': '2025-06-01'
        },
        # 测试 wy03 策略在 14:30 交易
        {
            'strategy_name': 'wy03',
            'test_name': 'time_1430',
            'params': {'EXECUTION_TIME_PLACEHOLDER': "'14:30'"},
            'start_date': '2025-01-01',
            'end_date': '2025-06-01'
        },
        # 测试 yj15 策略 (默认时间)
        {
            'strategy_name': 'yj15',
            'test_name': 'default_time',
            'params': {
                'EXECUTION_SOLD_TIME_PLACEHOLDER': "'09:30'",
                'EXECUTION_BUY_TIME_PLACEHOLDER': "'09:35'"
            },
            'start_date': '2025-01-01',
            'end_date': '2025-06-01'
        },
        # 测试 yj15 策略 (延迟时间)
        {
            'strategy_name': 'yj15',
            'test_name': 'delayed_1400',
            'params': {
                'EXECUTION_SOLD_TIME_PLACEHOLDER': "'14:00'",
                'EXECUTION_BUY_TIME_PLACEHOLDER': "'14:05'"
            },
            'start_date': '2025-01-01',
            'end_date': '2025-06-01'
        }
    ]
    
    # 执行批量对比
    compare_backtests(configs)