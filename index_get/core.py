import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union, Optional, Callable, Iterable

from config import (INDICATOR_CONFIG_PATH,
                    BASIC_INDICATOR_CONFIG,
                    Update_Cond,
                    DateTime_FMT)
from common.trade_date import (get_trade_day,
                               get_delta_trade_day,
                               get_next_update_time)
from common.walk_conds import lexpr2str, to_numeric
# from get_baidu_search import bd_search_nearday


def read_conf_file(pth):
    with open(pth, 'r', encoding='UTF-8') as f:
        try:
            jread = json.load(f)
        except json.JSONDecodeError as e:
            jread = dict()
    return jread


class IndicatorGetter(object):
    _indicator_config_pth = INDICATOR_CONFIG_PATH
    _indicator_child_dict = dict()

    def __init__(self, cator_name: str, update_fun=None) -> None:
        self.cator_name = cator_name
        self.now_datetime = datetime.now().strftime(DateTime_FMT)
        self.project_dir = Path(Path(__file__).parent, '..')
        self.update_fun = update_fun
        self.uppth = None
        jread = read_conf_file(IndicatorGetter._indicator_config_pth)
        if cator_name in jread:
            self.cator_conf = jread[cator_name]
        else:
            self.cator_conf = BASIC_INDICATOR_CONFIG

    def get_cator_conf(self):
        return self.cator_conf

    def set_cator_conf(self, to_file: bool = False, **kwargs):
        self.cator_conf.update(**kwargs)
        if to_file:
            jres = read_conf_file(IndicatorGetter._indicator_config_pth)
            with open(IndicatorGetter._indicator_config_pth, 'w', encoding='UTF-8') as wf:
                jres[self.cator_name] = self.cator_conf
                json.dump(jres, wf, indent=2, ensure_ascii=False)

    def update_db_data(self):
        pass

    def append_db_date(self):
        pass

    def append_data(self, fun, aitems):
        conf = self.cator_conf
        if isinstance(conf['dbpath'], str):
            self.uppth = Path(self.project_dir, conf['dbpath'])
            return self.append_db_data()
        else:
            self.uppth = Path(self.project_dir, conf['fpath'])
        if aitems and isinstance(fun, Callable):
            app_data = fun(aitems, conf['itempath'])
            if not app_data: return None
            p1 = pd.read_csv(self.uppth,index_col=0)
            p1 = pd.concat([p1,app_data],axis=0)
            p1.sort_index(inplace=True)
            p1.to_csv(self.uppth, mode='w',float_format='%.3f')
        else:
            return None

    def update_data(self, down_fun=None):
        conf = self.cator_conf
        upfun = self.update_fun if self.update_fun is not None else down_fun
        nupdate_time = conf['next_update_time']
        if nupdate_time is False: nupdate_time = '2001-01-01'
        if conf['freq'] is False: return Update_Cond.NoNeed_Update
        if isinstance(conf['dbpath'], str):
            self.uppth = Path(self.project_dir, conf['dbpath'])
            return self.update_db_data()
        else:
            self.uppth = Path(self.project_dir, conf['fpath'])
        if self.now_datetime <= nupdate_time:
            return Update_Cond.Updated
        else:
            # TODO: 添加函数/添加款项名字？
            ## append_data = self.append_data(append_fun)
            up_kwargs = conf['update_kwargs']
            if isinstance(up_kwargs,dict):
                data = upfun(**up_kwargs)
            elif isinstance(up_kwargs, list):
                data = upfun(*[conf[u] for u in up_kwargs])
            else:
                data = upfun(conf[up_kwargs])
            near_up_date = data.index.max()
            conf['max_date_idx'] = near_up_date if isinstance(near_up_date, str) else near_up_date.strftime('%Y-%m-%d')
            conf['next_update_time'] = get_next_update_time(
                conf['max_date_idx'], conf['morn_or_night'], date_fmt=DateTime_FMT)
            if conf['update_method'] == 'append':
                data.to_csv(self.uppth, mode='a', header=False)
            else:
                data.to_csv(self.uppth, mode='w',float_format='%.3f')
            self.set_cator_conf(True, **conf)
            return Update_Cond.Updating

    def set_warn_info(self):
        conf = self.cator_conf
        warn_cond = conf['warning_cond']
        near_trade_date = get_delta_trade_day(conf['max_date_idx'], 0, date_fmt='%Y-%m-%d')
        if near_trade_date is None:
            near_trade_date = get_delta_trade_day(conf['max_date_idx'], -1, date_fmt='%Y-%m-%d')
        print('@@@', self.uppth)
        data = pd.read_csv(self.uppth, index_col=0)
        data = to_numeric(data.loc[near_trade_date])
        if isinstance(warn_cond, bool): return
        if isinstance(warn_cond, str):
            tt = data.query(warn_cond)
        else:
            keys, qstr = lexpr2str(warn_cond)
            tt = data.query(qstr)
        if not tt.empty:
            print(f"{self.cator_name} warning info update.")
            cond_lst = [tt.loc[tt.index[i], list(keys)].to_dict() for i in range(len(tt))]
            cond_lst.insert(0,near_trade_date)
            self.set_cator_conf(True, warning_info=cond_lst)
        else:
            self.set_cator_conf(True, warning_info=False)

    def get_warn_info(self):
        return self.cator_conf['warning_info']

if __name__ == '__main__':
    p1 = IndicatorGetter('baidu_search')
    # p1.update_data(bd_search_nearday)
    print(p1.cator_conf['max_date_idx'])
    # p1.get_warn_info()
    pass
