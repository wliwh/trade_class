from get_high_low import high_low_legu_indicator
# from get_north_flow import north_flow_indicator
from get_qvix_value import qvix_day_indicator
from get_baidu_search import bsearch_indicator
from get_index_score import index_score_indicator
from get_index_value import global_index_indicator
from config import _logger


hl = high_low_legu_indicator()
hl.update_data() 
hl.set_warn_info()

# nt = north_flow_indicator()
# nt.update_data()
# nt.set_warn_info()

qd = qvix_day_indicator()
qd.update_data()
qd.set_warn_info()

glb = global_index_indicator()
glb.update_data()
glb.set_warn_info()

sco = index_score_indicator()
sco.update_data()
sco.set_warn_info()

try:
    bd = bsearch_indicator()
    bd.update_data()
    bd.set_warn_info()
    # bd.set_warn_info('2024-01-01')
except FileNotFoundError as e:
    _logger.error('bd_search cooks file not found.')
        # pass