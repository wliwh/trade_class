from get_high_low import high_low_legu_indicator
from get_north_flow import north_flow_indicator
from get_qvix_value import qvix_day_indicator
from get_baidu_search import bsearch_indicator


hl = high_low_legu_indicator()
print(hl.update_data())
hl.set_warn_info()

nt = north_flow_indicator()
print(nt.update_data())
nt.set_warn_info()

# qd = qvix_day_indicator()
# print(qd.update_data())
# qd.set_warn_info()

bd = bsearch_indicator()
print(bd.update_data())
bd.set_warn_info()