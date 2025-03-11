
import os, sys
from datetime import datetime
# import streamlit as st
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from web.warning_page import echart_warn_page



if __name__ == '__main__':
    # main_page(Warning_Infos)
    # pg = st.navigation([
    #     st.Page(main_page, title='汇总'),
    #     st.Page("/home/hh01/Documents/trade_class/web/bs0.py",title='德国')
    # ])
    # pg.run()
    echart_warn_page(True)
    pass