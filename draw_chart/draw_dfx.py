from get_funds_value import get_bond_index
from pyecharts import options as opts
from pyecharts.charts import Scatter, Timeline
from pyecharts.faker import Faker
from pyecharts.commons.utils import JsCode


Au = {
    '黄金': (7.01, 41.2)
}
WFX = {
    '大额存单-2020': (3.9, 0),
    '大额存单-2024': (2.35, 0),
    '货币基金': (2.1, 0),
    # '券商新客理财': (6, 0),
}

def draw_dfx_ret(beg='2013'):
    ret = list()
    Bonds = {
        '中证全债': 'H11001',
        '短期国债': 'H11099',
        '长期国债': 'H11077',
        '超长期国债': '931080',
        '同业存单AAA': '931059',
        '信用债高等级': '931203',
        '沪城投债': 'H11098',
        '纯债债基': '930609',
    }
    for n,v in Bonds.items():
        ret.append([n]+list(get_bond_index(v,beg+'-12-31','2024-03-29')))
    return ret


def draw_year2now(rg=('2013','2018','2020')):
    t1 = Timeline()
    for y in rg:
        bd = draw_dfx_ret(y)
        c = (
            Scatter()
            .add_xaxis(xaxis_data=[b[1] for b in bd])
            .add_yaxis(
                series_name="债券",
                y_axis=[[b[2],b[0]] for b in bd],
                symbol_size=20,
                label_opts=opts.LabelOpts(
                    distance=2,
                    position='top',
                    formatter=JsCode("function(x) {return x.value[2];}"),
            ))
            .set_series_opts()
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)
                ),
                yaxis_opts=opts.AxisOpts(
                    type_="value",
                    axistick_opts=opts.AxisTickOpts(is_show=True),
                    splitline_opts=opts.SplitLineOpts(is_show=True),
                ),
                tooltip_opts=opts.TooltipOpts(
                    is_show=True,
                    formatter=JsCode(
                        "function (x) {return x.value[0] + ', ' + x.value[1];}"
                    )),
            )
            # .render("basic_scatter_chart.html")
        )
        t1.add(c,'{}年至今'.format(int(y)+1))
    t1.render('basic_scatter_chart.html')


# draw_year2now()