import os
import json
import hashlib
import time
import pandas as pd
from datetime import datetime

# 缓存目录
CACHE_DIR = 'test1/ETFs/backtest_cache'
ACTIVE_TASKS_FILE = os.path.join(CACHE_DIR, 'active_tasks.json')

# 策略文件路径映射
STRATEGY_FILES = {
    'wy03': 'test1/ETFs/ETF_wy03_opt.py',
    'long': 'test1/ETFs/ETF_long_modular.py',
    'yj15': 'test1/ETFs/ETF_yj15_modular.py'
}

def ensure_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)

def get_task_hash(strategy_content, params, start_date, end_date, initial_cash, frequency):
    """计算任务唯一Hash"""
    param_str = json.dumps(params, sort_keys=True)
    raw_str = f"{strategy_content}|{param_str}|{start_date}|{end_date}|{initial_cash}|{frequency}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

class ActiveTaskManager:
    """管理活跃任务的持久化存储"""
    @staticmethod
    def _load_active_tasks():
        ensure_cache_dir()
        if os.path.exists(ACTIVE_TASKS_FILE):
            try:
                with open(ACTIVE_TASKS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading active tasks: {e}")
        return {}

    @staticmethod
    def _save_active_tasks(tasks):
        ensure_cache_dir()
        try:
            with open(ACTIVE_TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving active tasks: {e}")

    @classmethod
    def get_task(cls, task_hash):
        tasks = cls._load_active_tasks()
        return tasks.get(task_hash)

    @classmethod
    def add_task(cls, task_hash, backtest_id, meta):
        tasks = cls._load_active_tasks()
        tasks[task_hash] = {
            'id': backtest_id,
            'meta': meta,
            'status': 'running',
            'update_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        cls._save_active_tasks(tasks)

    @classmethod
    def remove_task(cls, task_hash):
        tasks = cls._load_active_tasks()
        if task_hash in tasks:
            del tasks[task_hash]
            cls._save_active_tasks(tasks)

def save_cache(task_hash, data):
    """保存回测结果到本地缓存"""
    ensure_cache_dir()
    file_path = os.path.join(CACHE_DIR, f"{task_hash}.json")
    
    # 特殊处理 monthly_metrics: 保存CSV并转为dict以便JSON序列化
    if 'monthly_metrics' in data and isinstance(data['monthly_metrics'], dict):
        for key, val in data['monthly_metrics'].items():
            if isinstance(val, pd.DataFrame):
                # 1. 保存 CSV
                # csv_path = os.path.join(CACHE_DIR, f"{task_hash}_{key}.csv")
                # val.to_csv(csv_path, encoding='utf-8_sig')
                # 2. 原地替换为 dict (JSON兼容)
                data['monthly_metrics'][key] = val.to_dict(orient='list')
                data['monthly_metrics'][key]['_index'] = val.index.tolist()
            else:
                data['monthly_metrics'][key] = val
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return file_path
    except Exception as e:
        print(f"Failed to save cache: {e}")
        return None

def load_cache_meta(file_path):
    """读取本地缓存文件"""
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

def run_strategy_backtest(
    strategy_name, 
    test_name, 
    params, 
    start_date, 
    end_date, 
    initial_cash=100000, 
    frequency="day", 
    block=True
):
    """
    运行单个回测任务 
    block: True=阻塞直到完成, False=立即返回任务状态
    Returns:
        {
            'status': 'done'|'running'|'failed',
            'path': file_path (if done),
            'id': backtest_id (if running),
            'hash': task_hash
        }
    """
    if strategy_name not in STRATEGY_FILES:
        print(f"Strategy {strategy_name} not found.")
        return None
        
    file_path = STRATEGY_FILES[strategy_name]
    
    # 1. 准备工作
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            strategy_content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    current_params = get_empty_placeholders(file_path)
    current_params.update(params)
    task_hash = get_task_hash(strategy_content, current_params, start_date, end_date, initial_cash, frequency)
    
    cache_path = os.path.join(CACHE_DIR, f"{task_hash}.json")
    
    # 2. Level 1: Check Local Cache
    if os.path.exists(cache_path):
        print(f"[{test_name}] Found local cache.")
        return {'status': 'done', 'path': cache_path, 'hash': task_hash}

    # 3. Level 2: Check Active Tasks
    active_task = ActiveTaskManager.get_task(task_hash)
    bt_id = None
    
    if active_task:
        bt_id = active_task['id']
        print(f"[{test_name}] Found active task: {bt_id}, checking status...")
        
        try:
            gt = get_backtest(bt_id)
            status = gt.get_status()
            
            if status == 'done':
                # Task finished, retrieve and save
                result_data = {
                    'metrics': gt.get_risk(),
                    'daily_results': gt.get_results(),
                    'monthly_metrics': gt.get_period_risks('month'),
                    'meta': active_task['meta'],
                    'hash': task_hash
                }
                saved_path = save_cache(task_hash, result_data)
                ActiveTaskManager.remove_task(task_hash)
                print(f"[{test_name}] Task finished and saved.")
                return {'status': 'done', 'path': saved_path, 'hash': task_hash}
                
            elif status in ['failed', 'canceled', 'deleted']:
                print(f"\n[CRITICAL] Backtest {test_name} ({bt_id}) FAILED with status: {status}")
                ActiveTaskManager.remove_task(task_hash)
                # Fall through to create new task
                bt_id = None 
                
            else: # running, none, paused
                if not block:
                    return {'status': 'running', 'id': bt_id, 'hash': task_hash}
                # If blocking, we need to wait. Handled below.
                
        except Exception as e:
            print(f"Error checking backtest {bt_id}: {e}")
            ActiveTaskManager.remove_task(task_hash) # clean up bad record
            bt_id = None

    # 4. Create New Task (if needed)
    if bt_id is None:
        code = generate_strategy_code(file_path, current_params)
        if not code: return None
        
        print(f"[{test_name}] Creating NEW backtest for {strategy_name}...")
        try:
            bt_id = create_backtest(
                algorithm_id=None,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                initial_cash=initial_cash,
                code=code,
                name=f"{strategy_name}_{test_name}",
                python_version=3
            )
            meta = {
                'strategy_name': strategy_name,
                'test_name': test_name,
                'params': params,
                'frequency': frequency,
                'start_date': start_date,
                'end_date': end_date,
                'initial_cash': initial_cash,
                'create_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            ActiveTaskManager.add_task(task_hash, bt_id, meta)
            
            if not block:
                return {'status': 'running', 'id': bt_id, 'hash': task_hash}
                
        except Exception as e:
            print(f"Failed to create backtest: {e}")
            return None

    # 5. Blocking Wait (Only reachable if block=True and task is running)
    print(f"[{test_name}] Waiting for backtest {bt_id} to complete...")
    while True:
        try:
            gt = get_backtest(bt_id)
            status = gt.get_status()
            
            if status == 'done':
                # Need to fetch meta again if we just recovered an active task or use the one we just created
                # To be safe, rely on what we saved in active tasks or reconstruct
                active_task = ActiveTaskManager.get_task(task_hash)
                meta = active_task['meta'] if active_task else {}
                
                try:
                    monthly_metrics = gt.get_period_risks('month')
                except:
                    monthly_metrics = None
                    
                result_data = {
                    'metrics': gt.get_risk(),
                    'daily_results': gt.get_results(),
                    'monthly_metrics': monthly_metrics,
                    'meta': meta,
                    'hash': task_hash
                }
                saved_path = save_cache(task_hash, result_data)
                ActiveTaskManager.remove_task(task_hash)
                print(f"[{test_name}] Done.")
                return {'status': 'done', 'path': saved_path, 'hash': task_hash}
                
            elif status in ['failed', 'canceled', 'deleted']:
                print(f"\n[CRITICAL] Backtest {test_name} ({bt_id}) FAILED with status: {status}")
                ActiveTaskManager.remove_task(task_hash)
                return {'status': 'failed', 'hash': task_hash}
                
            time.sleep(5)
        except Exception as e:
            print(f"Error while waiting for {bt_id}: {e}")
            time.sleep(5)


# ==========================================
# 批量回测与指标对比功能
# ==========================================

def compare_backtests(backtest_configs):
    """
    批量运行回测并对比指标
    """
    # 1. Start/Resume all tasks (Non-blocking)
    print(f"Initializing {len(backtest_configs)} tasks...")
    
    # We keep track of configs that are not yet done
    pending_configs = backtest_configs[:] 
    finished_results = []
    
    while pending_configs:
        still_pending = []
        
        for config in pending_configs:
            # Re-call run_strategy_backtest. 
            # If it's running, it checks status. 
            # If it finishes, it saves data and returns 'done'.
            res = run_strategy_backtest(
                strategy_name=config['strategy_name'],
                test_name=config['test_name'],
                params=config['params'],
                start_date=config['start_date'],
                end_date=config['end_date'],
                block=False 
            )
            
            if res:
                res['name'] = f"{config['strategy_name']}_{config['test_name']}"
                
                if res['status'] == 'done':
                    finished_results.append(res)
                    # print(f"Task {res['name']} finished.") # Optional, run_strategy_backtest already logs
                elif res['status'] in ['failed', 'canceled', 'deleted']:
                    # Already logged critical error in run_strategy_backtest
                    pass 
                else:
                    # Still running (or other status), keep in pending
                    still_pending.append(config)
            else:
                 # Error creating/checking, stop tracking this one
                 print(f"Skipping failed config: {config.get('test_name')}")

        pending_configs = still_pending
        
        # Only sleep if we still have pending tasks
        if pending_configs:
            time.sleep(5)
    
    print("\nAll tasks completed validation.\n")

    # 3. Collect Results
    metrics_list = []
    
    for task in finished_results:
        data = load_cache_meta(task['path'])
        if not data or 'metrics' not in data:
            continue
            
        risk = data['metrics']
        try:
            metrics = {
                'Name': task['name'],
                'Annual Return': f"{risk.get('annual_algo_return', 0):.2%}",
                'Max Drawdown': f"{risk.get('max_drawdown', 0):.2%}",
                'Sharpe': f"{risk.get('sharpe', 0):.2f}",
                'Win Ratio': f"{risk.get('win_ratio', 0):.2%}",
                'Profit/Loss Ratio': f"{risk.get('profit_loss_ratio', 0):.2f}",
                'Total Return': f"{risk.get('algorithm_return', 0):.2%}",
                'Alpha': f"{risk.get('alpha', 0):.2f}",
                'Beta': f"{risk.get('beta', 0):.2f}"
            }
            metrics_list.append(metrics)
        except Exception as e:
             print(f"Error parsing metrics for {task['name']}: {e}")

    # 4. Display Table
    if metrics_list:
        df = pd.DataFrame(metrics_list)
        # cols = ['Name', 'Annual Return', 'Max Drawdown', 'Sharpe', 'Win Ratio', 'Profit/Loss Ratio', 'Total Return', 'Alpha', 'Beta']
        # df = df[cols] 
        # (Auto column order is usually fine, specific order needs check if cols exist)
        
        print("="*100)
        print("STRATEGY PERFORMANCE COMPARISON")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
    else:
        print("No successful results to display.")

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