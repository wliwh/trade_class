import os
import json
import hashlib
import time
import pandas as pd
from datetime import datetime

# 缓存目录
CACHE_DIR = 'test1/ETFs/backtest_cache'

# 策略文件路径映射
STRATEGY_FILES = {
    'wy03': 'test1/ETFs/ETF_wy03_opt.py',
    'long': 'test1/ETFs/ETF_long_modular.py',
    'yj15': 'test1/ETFs/ETF_yj15_modular.py'
}

def get_task_hash(strategy_content, params, start_date, end_date, initial_cash, frequency):
    """计算任务唯一Hash"""
    # 构造唯一标识串：策略内容 + 排序后的参数 + 时间区间 + 资金
    param_str = json.dumps(params, sort_keys=True)
    raw_str = f"{strategy_content}|{param_str}|{start_date}|{end_date}|{initial_cash}|{frequency}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

def save_cache(task_hash, data):
    """保存回测结果到本地缓存"""
    if not os.path.exists(CACHE_DIR):
        try:
            os.makedirs(CACHE_DIR)
        except Exception as e:
            print(f"Error creating cache dir: {e}")
            return

    file_path = os.path.join(CACHE_DIR, f"{task_hash}.json")
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Cache saved: {file_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")

def load_cache(task_hash):
    """读取本地缓存"""
    file_path = os.path.join(CACHE_DIR, f"{task_hash}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load cache: {e}")
    return None

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

def run_strategy_backtest(strategy_name, test_name, params, start_date, end_date, initial_cash=100000, frequency="day", force_run=False):
    """
    运行单个回测任务 (支持缓存)
    """
    if strategy_name not in STRATEGY_FILES:
        print(f"Strategy {strategy_name} not found.")
        return None
        
    file_path = STRATEGY_FILES[strategy_name]
    
    # 1. 读取策略文件内容用于Hash计算和代码生成
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            strategy_content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # 2. 获取完整参数 (底稿 + 覆盖)
    current_params = get_empty_placeholders(file_path)
    current_params.update(params)
    
    # 3. 计算Hash
    task_hash = get_task_hash(strategy_content, current_params, start_date, end_date, initial_cash, frequency)
    
    # 4. 检查缓存
    if not force_run:
        cached_data = load_cache(task_hash)
        if cached_data:
            print(f"Found cached result for [{test_name}] ({task_hash})")
            return {
                'status': 'cached',
                'name': f"{strategy_name}_{test_name}",
                'hash': task_hash,
                'metrics': cached_data.get('metrics', {})
            }

    # 5. 生成新代码
    code = generate_strategy_code(file_path, current_params)
    if not code:
        return None
        
    print(f"Creating backtest [{test_name}] for {strategy_name}...")
    
    # 6. 创建回测
    try:
        bt_id = create_backtest(
            algorithm_id=None, # 提供 code 时不需要 algorithm_id
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            initial_cash=initial_cash,
            code=code,
            name=f"{strategy_name}_{test_name}",
            python_version=3
        )
        print(f"Backtest created: {bt_id}")
        return {
            'status': 'running',
            'id': bt_id,
            'name': f"{strategy_name}_{test_name}",
            'hash': task_hash,
            'meta': {
                'strategy_name': strategy_name,
                'test_name': test_name,
                'params': params,
                'frequency': frequency,
                'start_date': start_date,
                'end_date': end_date,
                'initial_cash': initial_cash,
                'create_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
    except Exception as e:
        print(f"Failed to create backtest: {e}")
        return None

# ==========================================
# 批量回测与指标对比功能
# ==========================================


def compare_backtests(backtest_configs):
    """
    批量运行回测并对比指标 (支持持久化)
    """
    bt_results = []
    
    # 1. 提交所有回测任务
    print(f"Starting {len(backtest_configs)} backtests (checking cache)...")
    for config in backtest_configs:
        res = run_strategy_backtest(
            strategy_name=config['strategy_name'],
            test_name=config['test_name'],
            params=config['params'],
            start_date=config['start_date'],
            end_date=config['end_date']
        )
        if res:
            bt_results.append(res)
            
    if not bt_results:
        print("No backtests started.")
        return

    # 2. 区分正在运行的和已缓存的
    running_tasks = [t for t in bt_results if t['status'] == 'running']
    cached_tasks = [t for t in bt_results if t['status'] == 'cached']

    # 3. 轮询等待运行中的任务完成
    if running_tasks:
        print(f"\nWaiting for {len(running_tasks)} backtests to complete...")
        all_done = False
        while not all_done:
            all_done = True
            pending_count = 0
            for task in running_tasks:
                if task.get('final_status') in ['done', 'failed', 'canceled']:
                    continue
                    
                try:
                    gt = get_backtest(task['id'])
                    status = gt.get_status()
                    
                    if status == 'done':
                        task['final_status'] = 'done'
                        # 获取结果并保存缓存
                        risk = gt.get_risk()
                        cache_data = {
                            'hash': task['hash'],
                            'meta': task['meta'],
                            'metrics': risk,
                            'status': 'done'
                        }
                        save_cache(task['hash'], cache_data)
                        task['metrics'] = risk
                        print(f"Task {task['name']} finished and cached.")
                        
                    elif status in ['failed', 'canceled', 'deleted']:
                        task['final_status'] = status
                        print(f"Task {task['name']} ended with status: {status}")
                        
                    else:
                        all_done = False
                        pending_count += 1
                except Exception as e:
                    print(f"Error checking status for {task['id']}: {e}")
                    
            if not all_done:
                print(f"Running: {pending_count}/{len(running_tasks)}...", end='\r')
                time.sleep(5)
        print("\nAll running backtests finished.\n")
    
    # 4. 收集所有指标并展示
    metrics_list = []
    # 合并 cached_tasks 和 finished running_tasks
    all_tasks = cached_tasks + running_tasks
    
    for task in all_tasks:
        params_metrics = task.get('metrics')
        if not params_metrics:
            continue
            
        try:
            risk = params_metrics
            metrics = {
                'Name': task['name'],
                'Annual Return': f"{risk.get('annual_algo_return', 0):.2%}",
                'Total Return': f"{risk.get('algorithm_return', 0):.2%}",
                'Max Drawdown': f"{risk.get('max_drawdown', 0):.2%}",
                'Sharpe': f"{risk.get('sharpe', 0):.2f}",
                'Win Ratio': f"{risk.get('win_ratio', 0):.2%}",
                'Profit/Loss Ratio': f"{risk.get('profit_loss_ratio', 0):.2f}",
                'Total Returns': f"{risk.get('algorithm_return', 0):.2%}", # Field name consistency check, user used 'Total Return' in original, JQ has 'algorithm_return'
                'Alpha': f"{risk.get('alpha', 0):.2f}",
                'Beta': f"{risk.get('beta', 0):.2f}"
            }
            # Fix column name consistency from original code
            # Original: 'Total Return': f"{risk.get('algorithm_return', 0):.2%}"
            metrics['Total Return'] = metrics.pop('Total Returns')
            
            metrics_list.append(metrics)
        except Exception as e:
            print(f"Error formulating metrics for {task['name']}: {e}")
            
    # 5. 生成对比表格
    if metrics_list:
        df = pd.DataFrame(metrics_list)
        cols = ['Name', 'Annual Return', 'Max Drawdown', 'Sharpe', 'Win Ratio', 'Profit/Loss Ratio', 'Total Return', 'Alpha', 'Beta']
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        print("="*80)
        print("STRATEGY PERFORMANCE COMPARISON (Cached + New)")
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