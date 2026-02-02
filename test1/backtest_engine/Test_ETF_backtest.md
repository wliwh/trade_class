
## 设计方案

程序要完成下面几个功能：

1. 修改策略的设置的可以更该的参数；已完成
2. 创建回测，并将回测结果保存到本地；考虑下面几种情况：
    a. 发现同样的回测结果已经保存，直接返回本地结果的保存地址，不再创建回测
    b. 发现要创建一个新的回测，创建回测后
        - 监测该回测直到结束，保存结果到本地，阻塞式
        - 立即返回，不阻塞，再次调用时，查看回测状态，如果回测已结束，保存回测结果到本地，并返回本地结果的保存地址，如果回测未结束，则什么都不做，等待下次调用


## 回测函数说明

create\_backtest 通过一个策略ID从研究中创建回测

```
create_backtest(algorithm_id, start_date, end_date, frequency="day", initial_cash=10000, initial_positions=None, extras=None, name=None, code="", benchmark=None, python_version=2, use_credit=False)
```

通过一个策略ID从研究中创建回测，只能在研究中使用，目前不支持在回测及模拟交易中使用；

**参数：**

*   algorithm\_id: 策略ID，从策略编辑页的 url 中获取, 比如 '/algorithm/index/edit?algorithmId=xxxx'，则策略ID为: xxxx。
*   start\_date: 回测开始日期
*   end\_date: 回测结束日期
*   frequency: 数据频率，支持 day, minute, tick
*   initial\_cash: 初始资金
*   extras: 额外参数，一个 dict， 用于设置全局的 g 变量，如 extras={'x':1, 'y':2}，则回测中 g.x = 1, g.y = 2，需要注意的是，该参数的值是`在 initialize 函数执行之后`才设置给 g 变量的，所以这会覆盖掉 initialize 函数中 g 变量同名属性的值
*   name: 回测名, 用于指定回测名称, 如果没有指定则默认采用策略名作为回测名
*   initial\_positions: 初始持仓。持仓会根据价格换成现金加到初始资金中，如果没有给定价格则默认获取股票最近的价格。格式如下:
    ```
    initial_positions = [
        {
            'security':'000001.XSHE',
            'amount':'100',
        },
        {
            'security':'000063.XSHE',
            'amount':'100',
            'avg_cost': '1.0'
        },
    ]
    ```
*   code：策略代码。现在支持从研究中传入策略代码进行回测。指定之后将使用传入的代码来创建回测。
*   benchmark: 为回测设置基准。默认为None，表示使用策略中原有set\_benchmark设置的基准。若不为None，则表示使用当前传入的基准覆盖原策略的基准。benchmark支持的基准同set\_benchmark
*   python\_version: 创建回测的python的版本，已废弃参数,目前只支持python3内核
*   use\_credit:是否允许消耗积分新建回测。当每个自然日内编译运行、回测超过免费时间时，继续运行每30分钟需消耗2积分。默认为False，表示不允许消耗积分新建回测，设为True表示接受消耗积分新建回测。需注意，对于已经在运行中的回测，此配置不生效。

**返回：**

一个字符串, 即 backtest\_id

**示例一：**

```
algorithm_id = "xxxx"
extra_vars = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
initial_positions = [
    {
        'security':'000001.XSHE',
        'amount':'100',
    },
    {
        'security':'000063.XSHE',
        'amount':'100',
        'avg_cost': '1.0'
    },
]

params = {
    "algorithm_id": algorithm_id,
    "start_date": "2015-10-01",
    "end_date": "2016-07-31",
    "frequency": "day",
    "initial_cash": "1000000",
    "initial_positions": initial_positions,
    "extras": extra_vars,
}

created_bt_id = create_backtest(**params)
print(created_bt_id)

```

**示例二，在研究中指定回测用策略代码：**

```
code = """
# 导入函数库
from jqdata import *

# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 输出内容到日志 log.info()
    log.info('初始函数开始运行且全局只运行一次')
    # 过滤掉order系列API产生的比info级别低的log
    # log.set_level('order', 'info')

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
      # 开盘前运行
    run_daily(before_market_open, time='09:00', reference_security='000300.XSHG') 
      # 开盘时运行
    run_daily(market_open, time='09:30', reference_security='000300.XSHG')
      # 收盘后运行
    run_daily(after_market_close, time='15:30', reference_security='000300.XSHG')

## 开盘前运行函数     
def before_market_open(context):
    # 输出运行时间
    log.info('函数运行时间(before_market_open)：'+str(context.current_dt.time()))

    # 给微信发送消息（添加模拟交易，并绑定微信生效）
    send_message('美好的一天~')

    # 要操作的股票：平安银行（g.为全局变量）
    g.security = '000001.XSHE'

## 开盘时运行函数
def market_open(context):
    log.info('函数运行时间(market_open):'+str(context.current_dt.time()))
    security = g.security
    # 获取股票的收盘价
    close_data = attribute_history(security, 5, '1d', ['close'])
    # 取得过去五天的平均价格
    MA5 = close_data['close'].mean()
    # 取得上一时间点价格
    current_price = close_data['close'][-1]
    # 取得当前的现金
    cash = context.portfolio.available_cash

    # 如果上一时间点价格高出五天平均价1%, 则全仓买入
    if current_price > 1.01*MA5:
        # 记录这次买入
        log.info("价格高于均价 1%%, 买入 %s" % (security))
        # 用所有 cash 买入股票
        order_value(security, cash)
    # 如果上一时间点价格低于五天平均价, 则空仓卖出
    elif current_price < MA5 and context.portfolio.positions[security].closeable_amount > 0:
        # 记录这次卖出
        log.info("价格低于均价, 卖出 %s" % (security))
        # 卖出所有股票,使这只股票的最终持有量为0
        order_target(security, 0)

## 收盘后运行函数  
def after_market_close(context):
    log.info(str('函数运行时间(after_market_close):'+str(context.current_dt.time())))
    #得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info('成交记录：'+str(_trade))
    log.info('一天结束')
    log.info('##############################################################')

"""

algorithm_id = "xxxx"
extra_vars = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
initial_positions = [
    {
        'security':'000001.XSHE',
        'amount':'100',
    },
    {
        'security':'000063.XSHE',
        'amount':'100',
        'avg_cost': '1.0'
    },
]

params = {
    "algorithm_id": algorithm_id,
    "start_date": "2015-10-01",
    "end_date": "2016-07-31",
    "frequency": "day",
    "initial_cash": "1000000",
    "initial_positions": initial_positions,
    "extras": extra_vars,
}

created_bt_id = create_backtest(code=code, **params)

```

---------------------

get\_backtest 研究中获取回测与模拟交易信息

```
gt = get_backtest(backtest_id)

```

研究中获取回测与模拟交易信息，只能在研究中使用，目前不支持在回测及模拟交易中使用；使用方法及参考教程：

*   [参数优化及并行回测参考教程](https://www.joinquant.com/help/api/help?name=faq#%E5%8F%82%E6%95%B0%E4%BC%98%E5%8C%96%E5%8F%8A%E5%B9%B6%E8%A1%8C%E5%9B%9E%E6%B5%8B)
*   [区分create\_backtest(策略ID)和get\_backtest(回测ID)](https://www.joinquant.com/help/api/help?name=faq#%E5%8F%82%E6%95%B0%E4%BC%98%E5%8C%96%E5%8F%8A%E5%B9%B6%E8%A1%8C%E5%9B%9E%E6%B5%8B)
*   [create\_backtest与get\_backtest](https://www.joinquant.com/view/community/detail/1673d7bd3b883ad403a189288389efd8?type=1)

**参数：**

*   backtest\_id: 回测ID，从回测详情页以及模拟交易详情页的 url 中获取, 比如 '/algorithm/backtest/detail?backtestId=xxxx'以及'/algorithm/live/index?backtestId=xxxx'，则回测ID为:xxxx。

**返回：**

*   gt.get\_status():**获取回测状态**. 返回一个字符串，其含义分别为：
    *   none: 未开始
    *   running: 正在进行
    *   done: 完成
    *   failed: 失败
    *   canceled: 取消
    *   paused: 暂停
    *   deleted: 已删除
*   gt.get\_params()：**获得回测参数**. 返回一个 dict, 包含调用 create\_backtest 时传入的所有信息. (注： algorithm\_id，initial\_positions，extras 只有在研究中创建的回测才能取到)
    输出示例如下：
    ```json
    {'algorithm_id': '6e21bbaee8cc3f8423def84436a2bf49',
    'end_date': '2026-01-10 23:59:59',
    'extras': {},
    'frequency': 'day',
    'initial_cash': '100000',
    'initial_positions': [],
    'initial_value': None,
    'name': 'ETF_wy03_opt',
    'package_version': '1.0',
    'python_version': '3',
    'start_date': '2025-10-01 00:00:00',
    'subportfolios': [{'account_type': 'stock',
    'set_subportfolios': True,
    'starting_cash': 100000.0,
    'subAccountId': 0}]}
    ```
*   gt.get\_results()：**获得收益曲线**. 返回一个 list，每个交易日是一个 dict，键的含义如下：
    *   time: 时间
    *   returns: 收益
    *   benchmark\_returns: 基准收益
    *   上面的收益都是将初期投入视为单位1计算的累积收益，
    *   如果没有收益则返回一个空的 list
    *   例如：
    ```json
    [
        {'benchmark_returns': 0.021063078958976522,
        'returns': -0.010974204999999904,
        'time': '2025-10-09 16:00:00'},
        {'benchmark_returns': 0.0260255844728714,
        'returns': -0.053674204999999864,
        'time': '2025-10-10 16:00:00'},
        ...
    ]
    ```
*   gt.get\_positions(start\_date=None, end\_date=None)：**获得持仓详情**. 返回一个 list，默认取所有回测时间段内的数据。每个交易日为一个 dict，键的含义为：
    *   time: 时间
    *   amount: 持仓数量,
    *   avg\_cost: 开场均价,
    *   closeable\_amount: 可平仓数量,
    *   daily\_gains: 当日收益,
    *   gains: 累积收益,
    *   hold\_cost: 持仓成本（期货）,
    *   margin: 保证金,
    *   price: 当前价格,
    *   security: 标的代码,
    *   security\_name: 标的名,
    *   side: 仓位方向,
    *   today\_amount: 今开仓量
    *   如果没有持仓则返回一个空的 list
*   gt.get\_orders(start\_date=None, end\_date=None)：**获得交易详情**. 返回一个 list，默认取所有回测时间段内的数据。每个交易日为一个 dict，键的含义为：
    *   time: 时间
    *   action: 开平仓，'open'/'close',
    *   amount: 数量,
    *   commission: 手续费,
    *   filled: 已成交量,
    *   gains: 收益,
    *   limit\_price: 限价单委托价,
    *   match\_time: 最新成交时间,
    *   price: 成交价,
    *   security: 标的代码,
    *   security\_name: 标的名,
    *   side: 仓位方向,
    *   status: 订单状态,
    *   time: 委托时间,
    *   type: 委托方式，市价单/限价单
    *   如果没有交易则返回一个空的 list
*   gt.get\_records()：**获得所有 record 记录**. 返回一个 list，每个交易日为一个 dict，键是 time 以及调用 record() 函数时设置的值. 不设置record时返回一个空的 list
*   gt.get\_risk()：**获得总的风险指标**. 返回一个 dict，键是各类收益指标数据，如果没有风险指标则返回一个空的 dict.
    输出示例如下：
    ```json
    {'__version': 101,
    'algorithm_return': -0.12025479399999961,
    'algorithm_volatility': 0.2706099326034811,
    'alpha': -0.5093623235091826,
    'annual_algo_return': -0.38907469939513395,
    'annual_bm_return': 0.20524742681695063,
    'avg_excess_return': -0.0025672256449780525,
    'avg_position_days': 21.666666666666668,
    'avg_trade_return': -0.030596692152069782,
    'benchmark_return': 0.04973533303925959,
    'benchmark_volatility': 0.2200948291403348,
    'beta': 0.4858630821706263,
    'day_win_ratio': 0.47692307692307695,
    'excess_return': -0.1619361773287118,
    'excess_return_max_drawdown': 0.2320812539294742,
    'excess_return_max_drawdown_period': ['2025-10-09', '2025-12-04'],
    'excess_return_sharpe': -1.9633016846388105,
    'information': -2.175977104037733,
    'lose_count': 3,
    'max_drawdown': 0.1979117267999997,
    'max_drawdown_period': ['2025-10-09', '2025-11-19'],
    'max_leverage': 0.0,
    'period_label': '2026-01',
    'profit_loss_ratio': 0.27995397116152365,
    'sharpe': -1.5855837044379588,
    'sortino': -2.8391283026633487,
    'trading_days': 65,
    'treasury_return': 0.010082191780821918,
    'turnover_rate': 0.07067065611837263,
    'win_count': 1,
    'win_ratio': 0.25}
    ```
*   gt.get\_period\_risks()：**获得分月计算的风险指标**. 返回一个 dict，键是各类指标, 值为一个 pandas.DataFrame. 如果没有风险指标则返回一个空的 dict.
    *   键包括：'algorithm_return', 'benchmark_return', 'alpha', 'beta', 'sharpe', 'sortino', 'information', 'algorithm_volatility', 'benchmark_volatility', 'max_drawdown'。
    *   第一个键'algorithm_return'对应的值类型为pandas.DataFrame，其形式如下
    ```
    one_month	three_month	six_month	twelve_month
    2025-10	-0.134110	NaN	NaN	NaN
    2025-11	-0.038404	NaN	NaN	NaN
    2025-12	0.025574	-0.146069	NaN	NaN
    2026-01	0.030230	0.016001	NaN	NaN
    ```
*   gt.get\_balances(start\_date=None, end\_date=None): **获取回测每日市值**. 返回一个 list，默认取所有回测时间段内的数据。每个交易日为一个 dict
    *   其键包括：'net_value', 'cash', 'time', 'total_value', 'aval_cash'。 返回值示例：
    ```json
    [{'net_value': 98902.58,
    'cash': 235.08,
    'time': '2025-10-09 16:00:00',
    'total_value': 98902.58,
    'aval_cash': 235.08},
    {'net_value': 94632.58,
    'cash': 235.08,
    'time': '2025-10-10 16:00:00',
    'total_value': 94632.58,
    'aval_cash': 235.08},
    ...
    ]
    ```

**示例：**

```
gt = get_backtest("xxxx")

gt.get_status()        # 获取回测状态
gt.get_params()        # 获取回测参数
gt.get_results()       # 获取收益曲线
gt.get_positions()     # 获取所有持仓列表
gt.get_orders()        # 获取交易列表
gt.get_records()       # 获取所有record()记录
gt.get_risk()          # 获取总的风险指标
gt.get_period_risks()  # 获取分月计算的风险指标
gt.get_balances()      # 获取回测每日市值
```