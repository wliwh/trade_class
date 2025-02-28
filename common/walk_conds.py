from functools import lru_cache, wraps
import pandas as pd
from typing import List, Iterable, Union
import re

def walk_conds_lru(table: pd.DataFrame, condition: dict):
    """ 使用迭代条件过滤表格
        condition 可从table中过滤出满足条件的款项, 该条件是嵌套的, 有一定层次
        其语义如下
        imp := '&' | '|';
        item := Dict[<Str>:<Str>]
              | Dict[<Str>:<Numeric>]
              | Dict[<Str>:List[<Numeric>,<Numeric>]]
              | Dict[<Str>:List[<Bool>,<Numeric>]]
              | Dict[<Str>:List[<Numeric>,<Bool>]]
        condition := List[imp,(item|condition)...]

        其中一个基本语义单元item有两种大的类型, 都是仅有一项的字典
        例如 `{'item':'50ETF'}` 表示在table中过滤出item列为50ETF的行
        而其中含有数字的语义单元, 则表示过滤该列大于或小于设定的数. 例如
        `{'Q20':80}` 表示过滤table表Q20列大于80的行; `{'Q20':[90,10]}` 表示找出大于90或小于10的行; 若过滤条件为小于某数, 则可设定 `{<Str>:[False,<Numeric>]}`
    """
    # TODO: 对于索引非数字的情况
    def item_det(idx: frozenset, d: dict):
        tab = table.loc[list(idx)]
        k, v = next(iter(d.items()))
        if isinstance(v, str):
            return frozenset(tab.index[tab[k] == v])
        elif isinstance(v, (int, float)):
            return frozenset(tab.index[tab[k] >= v])
        elif isinstance(v, List):
            assert len(v) == 2, "List type item-value have wrong length."
            a1, a2 = v
            if isinstance(a1, bool):
                return frozenset(tab.index[tab[k] <= a2])
            elif isinstance(a2, bool):
                return frozenset(tab.index[tab[k] >= a1])
            elif a1 > a2:
                return frozenset(tab.index[(tab[k] >= a1) | (tab[k] <= a2)])
            elif a1 < a2:
                return frozenset(tab.index[(tab[k] >= a1) & (tab[k] <= a2)])

    def getitem(p, idx):
        for i in idx:
            p = p[i]
        return p

    @lru_cache()
    def walk(idx: frozenset, ci: tuple):
        c = getitem(condition, ci)
        if isinstance(c, dict):
            return item_det(idx, c)
        elif c[0] == '&':
            return frozenset.intersection(*(walk(idx, ci+(i,)) for i in range(1, len(c))))
        else:
            return frozenset.union(*(walk(idx, ci+(i,)) for i in range(1, len(c))))

    return walk(frozenset(table.index), tuple())


def lexpr2str(l:Iterable) -> str:

    warning_keys = list()
    def node(d:dict) -> str:
        k,v = next(iter(d.items()))
        if k not in warning_keys: warning_keys.append(k)
        if isinstance(v, str):
            return f"({k} == '{v}')"
        elif isinstance(v, (int, float)):
            return f"({k} > {v})"
        elif isinstance(v, Iterable):
            assert len(v) == 2, "List type item-value have wrong length."
            v1,v2=v
            if isinstance(v1,bool):
                return f"({k}<={v2})"
            elif isinstance(v2,bool):
                return f"({k}>={v1})"
            else:
                return f"(({k} < {v1}) or ({k} > {v2}))"
        else:
            return ""

    def getitem(p, idx):
        for i in idx:
            p = p[i]
        return p

    @lru_cache()
    def run_exp(idx:int=0):
        if idx==0: wks = Sent
        else: wks = Sent[idx]
        sgn = ' or ' if wks[0].strip()=='|' else ' and '
        return sgn.join((node(wks[i]) if isinstance(wks[i], dict) else '('+run_exp(i)+')' for i in range(1,len(wks))))
    
    def run_exp2(idx:tuple=()):
        wks = getitem(Sent, idx)
        sgn = ' or ' if wks[0].strip()=='|' else ' and '
        return sgn.join((node(wks[i]) if isinstance(wks[i], dict) else '('+run_exp2(idx+(i,))+')' for i in range(1, len(wks))))

    Sent = l
    sents = run_exp2()
    return warning_keys, sents


def to_numeric(values:Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    将 DataFrame 或者 Series 尽可能地转为数字
    """
    ignore = ['code', 'keyword', 'type', 'symbol', 'item']

    def convert(o: Union[str, int, float]) -> Union[str, float, int]:
        if not re.findall('\d', str(o)):
            return o
        try:
            if str(o).isalnum():
                o = int(o)
            else:
                o = float(o)
        except:
            pass
        return o

    if isinstance(values, pd.DataFrame):
        for column in values.columns:
            if column not in ignore:
                values.loc[:,column] = values.loc[:,column].apply(convert)
    elif isinstance(values, pd.Series):
        for index in values.index:
            if index not in ignore:
                values[index] = convert(values[index])
    return values


if __name__=='__main__':
    ss = ["|",
      ["&",{"code": "50ETF"},
           ["|",{"cp_div": [0.95,1.845]},
                {"high": [15.5,22.8]}]
      ],
      ["&",{"code": "300ETF"},{"high": 23}]
      ]
    print(lexpr2str(ss)[0])
    pass