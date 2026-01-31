import os
import json
import hashlib
import time
from datetime import datetime

# ==========================================
# 配置部分
# ==========================================

# 注册表文件路径 (保存已运行回测的记录)
REGISTRY_FILE = 'test1/ETFs/backtest_registry.json'

# 策略文件路径映射
STRATEGY_FILES = {
    'wy03': 'test1/ETFs/ETF_wy03_opt.py',
    'long': 'test1/ETFs/ETF_long_modular.py',
    'yj15': 'test1/ETFs/ETF_yj15_modular.py'
}

# ==========================================
# 核心功能类
# ==========================================

class SimpleBacktestManager:
    def __init__(self, registry_path=REGISTRY_FILE):
        self.registry_path = registry_path
        self.registry = self._load_registry()

    def _ensure_dir(self):
        directory = os.path.dirname(self.registry_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _load_registry(self):
        """加载回测注册表"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load registry: {e}")
        return {}

    def _save_registry(self):
        """保存回测注册表"""
        self._ensure_dir()
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[Error] Failed to save registry: {e}")

    def get_strategy_content(self, file_path):
        """读取策略文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"[Error] Cannot read strategy file {file_path}: {e}")
            return None

    def prepare_strategy_code(self, content, params):
        """
        替换策略代码中的占位符参数
        params: dict, 例如 {'EXECUTION_TIME_PLACEHOLDER': "'14:30'"}
        """
        if not params:
            return content
            
        new_lines = []
        for line in content.splitlines(keepends=True):
            stripped = line.strip()
            # 简单替换逻辑：查找 EXECUTION_ 开头的赋值语句
            replaced = False
            if stripped.startswith('EXECUTION_') and '=' in stripped:
                key = stripped.split('=', 1)[0].strip()
                if key in params:
                    # 替换为新值，保持 python 语法 (key = value)
                    new_lines.append(f"{key} = {params[key]}\n")
                    replaced = True
            
            if not replaced:
                new_lines.append(line)
        
        return ''.join(new_lines)

    def calculate_hash(self, strategy_content, params, start_date, end_date, initial_cash, frequency):
        """计算唯一任务Hash (策略代码+参数+回测设置)"""
        # 排序参数以确保一致性
        param_str = json.dumps(params, sort_keys=True)
        raw_str = f"{strategy_content}|{param_str}|{start_date}|{end_date}|{initial_cash}|{frequency}"
        return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

    def run_backtest(self, config):
        """
        运行单个回测配置
        config: dict, 包含 strategy_name, test_name, params, start_date, end_date 等
        """
        s_name = config.get('strategy_name')
        t_name = config.get('test_name', 'unnamed')
        
        if s_name not in STRATEGY_FILES:
            print(f"[{t_name}] Error: Strategy '{s_name}' not defined in STRATEGY_FILES.")
            return

        file_path = STRATEGY_FILES[s_name]
        raw_content = self.get_strategy_content(file_path)
        if not raw_content:
            return

        # 准备参数
        params = config.get('params', {})
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        initial_cash = config.get('initial_cash', 100000)
        frequency = config.get('frequency', 'day')

        # 1. 生成最终待运行的代码
        final_code = self.prepare_strategy_code(raw_content, params)
        
        # 2. 计算 Hash
        task_hash = self.calculate_hash(final_code, params, start_date, end_date, initial_cash, frequency)

        # 3. 检查注册表
        if task_hash in self.registry:
            existing = self.registry[task_hash]
            print(f"[{t_name}] Skipped: Already exists. (ID: {existing.get('id')}, Created: {existing.get('create_time')})")
            return existing['id']

        # 4. 创建新回测 (调用 JoinQuant API)
        print(f"[{t_name}] Creating new backtest...")
        try:
            #以此检查环境是否支持 create_backtest
            if 'create_backtest' not in globals():
                print(f"[{t_name}] Error: 'create_backtest' function not found. Are you running in JoinQuant Research env?")
                return None

            full_name = f"{s_name}_{t_name}"
            bt_id = create_backtest(
                algorithm_id=None, # 使用 code 模式
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                initial_cash=initial_cash,
                code=final_code,
                name=full_name,
                python_version=3
            )
            
            # 5. 更新注册表
            record = {
                'id': bt_id,
                'name': full_name,
                'strategy': s_name,
                'test_name': t_name,
                'params': params,
                'range': [start_date, end_date],
                'create_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'status': 'created' 
            }
            self.registry[task_hash] = record
            self._save_registry()
            
            print(f"[{t_name}] Success: Created backtest ID {bt_id}")
            return bt_id

        except Exception as e:
            print(f"[{t_name}] Failed to create backtest: {e}")
            return None

# ==========================================
# 使用示例 / 主程序
# ==========================================

def main():
    manager = SimpleBacktestManager()
    
    # 定义任务列表
    configs = [
        # 1. wy03: 10:30 交易
        {
            'strategy_name': 'wy03',
            'test_name': 'time_1030',
            'params': {'EXECUTION_TIME_PLACEHOLDER': "'10:30'"},
            'start_date': '2025-01-01',
            'end_date': '2025-06-01'
        },
        # 2. wy03: 14:30 交易
        {
            'strategy_name': 'wy03',
            'test_name': 'time_1430',
            'params': {'EXECUTION_TIME_PLACEHOLDER': "'14:30'"},
            'start_date': '2025-01-01',
            'end_date': '2025-06-01'
        },
        # 3. yj15: 默认时间
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
        # 4. yj15: 延迟时间
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

    print(f"Processing {len(configs)} backtest configurations...")
    for cfg in configs:
        manager.run_backtest(cfg)
    print("All done.")

if __name__ == "__main__":
    main()
