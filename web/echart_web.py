import os, sys
import pandas as pd
from pyecharts.charts import Tab
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.baidu_utils import Search_Name_Path
from draw_chart.draw_bsearch import draw_future_echart
from web.warning_page import Warning_Infos


Bsearch_Page_Name = {0:'国内',1:'香港',2:'海外',3:'大宗'}
Draw_Echarts_Path = os.path.join(os.path.dirname(__file__),'..','draw_chart')


def echart_warn_page(allpage:bool=False, infos=Warning_Infos):
    p = pd.read_csv(Search_Name_Path,index_col=0)
    tend_dic = defaultdict(list)
    if allpage:
        p = p.loc[p['type']>=0]
        p['date'] = p['neardate'].map(lambda x:str(int(x[:4])-1)+x[4:])
        for nm, row in p.transpose().to_dict().items():
            tend_dic[row['type']].append((nm, row['code'], row['date'], row['neardate']))
    else:
        for i in infos['搜索指数'][1:]:
            nm = i['keyword']
            date, code, tp = p.loc[p.index==nm,['neardate','code','type']].values[0]
            beg_date = str(int(date[:4])-1)+date[4:]
            if tp>=0: 
                tend_dic[tp].append((nm, code, beg_date, date))
    tends = dict()
    for k,v in tend_dic.items():
        tab = Tab()
        for nm,cd,beg,end in v:
            tend = draw_future_echart(cd,nm,False,beg,end,k)
            tab.add(tend,nm+'-'+cd)
        tab.render(os.path.join(Draw_Echarts_Path,Bsearch_Page_Name[k]+'.html'))
        # tends[Bsearch_Page_Name[k]] = tab
    # return tends


def echart_page(infos, set_id=0):
    tab = Tab()
    p = pd.read_csv(Search_Name_Path,index_col=0)
    for i in infos['搜索指数'][1:]:
        nm = i['keyword']
        date, code, tp = p.loc[p.index==nm,['neardate','code','type']].values[0]
        if tp==set_id: 
            beg_date = str(int(date[:4])-1)+date[4:]
            tend = draw_future_echart(code, nm, False, beg_date,date, tp)
            return tend
            # tab.add(tend, f'{nm}-{code}')
    # return tab

