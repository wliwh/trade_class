from enum import Enum
from pathlib import Path

INDICATOR_CONFIG_PATH = \
    Path.joinpath(Path(__file__).parent, '..', 'cator.json')

DateTime_FMT = r'%Y-%m-%d %H:%M'

class Update_Cond(Enum):
    NoNeed_Update = 1
    Updated = 2
    Updating = 3
    Wrong_Update = 4

BASIC_INDICATOR_CONFIG = dict(
    zh = "中文标题",
    freq = 'day',
    fpath = False,
    itempath = False,           # 子项名称文件的路径
    dbpath = False,             # 数据库文件路径

    update_method = 'append',
    morn_or_night = '#9',
    max_date_idx = False,       # 当前数据最新日期
    next_update_time = False,   # 超过这一时间进行更新
    update_kwargs = dict(),
    update_kwargs_note = dict(),
    
    item_name = False,
    item_name_type = 'str',
    append_item_args = False,
    # quantile_periods = [20, 60, 120],
    
    warning_cond = False,
    warning_info = False
)


