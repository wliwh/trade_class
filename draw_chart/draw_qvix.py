from index_get.get_funds_value import basic_index_getter
import pandas as pd
import matplotlib.pyplot as plt

szzs = basic_index_getter('000001','2019-01-04','2024-04-19')
sz50 = basic_index_getter('000016','2019-01-04','2024-04-19')
hs300 = basic_index_getter('000300','2019-01-04','2024-04-19')
zz1000 = basic_index_getter('000852', '2019-01-04','2024-04-19')

corr_mat = pd.concat([
    szzs['close'].rolling(30).corr(sz50.close),
    szzs['close'].rolling(30).corr(hs300.close),
    szzs['close'].rolling(30).corr(zz1000.close)
],axis=1)
corr_mat.columns = ('sz50','hs300','zz1000')

corr_mat.loc['2022-08-26':'2023-01-05'].plot()
plt.show()