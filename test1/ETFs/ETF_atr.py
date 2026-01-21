# 策略名称：ETF收益率稳定性轮动策略（带短期动量过滤和ATR动态止损）- 修复未来函数版本
# 策略作者：屌丝逆袭量化
# 优化时间：2025-12-30
# 修复内容：修复所有未来函数问题，使用前1分钟价格

import numpy as np
import math
import pandas as pd
from datetime import datetime, timedelta

# 初始化函数，设置策略参数
def initialize(context):
    # ==================== 实盘交易设置 ====================
    set_option("avoid_future_data", True)
    set_option("use_real_price", True)
    
    # 设置滑点
    set_slippage(FixedSlippage(0.0001), type="fund")
    set_slippage(FixedSlippage(0.003), type="stock")
    
    # 设置交易成本
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0002,
            close_commission=0.0002,
            close_today_commission=0,
            min_commission=5,
        ),
        type="fund",
    )
    
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0,
            close_commission=0,
            close_today_commission=0,
            min_commission=0,
        ),
        type="mmf",
    )
    
    # 设置日志级别
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    
    log.info("策略2优化版初始化完成 - 已修复未来函数问题")
    set_benchmark("513100.XSHG")
    
    # ==================== 策略参数设置 ====================
    g.etf_pool = [
        "159915.XSHE",  # 创业板ETF
        "518880.XSHG",  # 黄金ETF
        "513100.XSHG",  # 纳指ETF
        "511220.XSHG",  # 城投债ETF
    ]
    
    g.etf_pool0 = [
        # 大宗商品ETF
        "518880.XSHG", "159980.XSHE", "159985.XSHE", "501018.XSHG",
        # 国际ETF
        "513100.XSHG", "513500.XSHG", "513520.XSHG", "513030.XSHG", "513080.XSHG",
        # 香港ETF
        "159920.XSHE",
        # 中国ETF
        "510300.XSHG", "510500.XSHG", "510050.XSHG", "510210.XSHG", "159915.XSHE",
        "588080.XSHG", "159995.XSHE", "513050.XSHG", "159852.XSHE", "159845.XSHE",
        "515030.XSHG", "159806.XSHE", "516160.XSHG", "159928.XSHE",
        # 防御ETF
        "511010.XSHG", "511880.XSHG",
    ]
    
    # 策略参数
    g.lookback_days = 25
    g.holdings_num = 1
    g.stop_loss = 0.95
    g.loss = 0.97
    g.defensive_etf = "511880.XSHG"
    g.min_score_threshold = 0.0
    g.max_score_threshold = 6.0
    g.min_money = 5000
    
    # 新增参数
    g.use_short_momentum_filter = True
    g.short_lookback_days = 12
    g.short_momentum_threshold = 0.0
    
    g.use_atr_stop_loss = True
    g.atr_period = 14
    g.atr_multiplier = 2
    g.atr_trailing_stop = False
    g.atr_exclude_defensive = True
    
    g.use_ma_filter = False
    g.ma_short_period = 5
    g.ma_long_period = 25
    g.ma_filter_condition = "above"
    
    g.use_rsi_filter = False
    g.rsi_period = 6
    g.rsi_lookback_days = 1
    g.rsi_threshold = 95
    
    g.use_macd_filter = False
    g.macd_fast_period = 12
    g.macd_slow_period = 26
    g.macd_signal_period = 9
    g.macd_filter_condition = "bullish"
    
    g.use_volume_filter = False
    g.volume_lookback_days = 7
    g.volume_threshold = 2.0
    g.volume_exclude_defensive = True
    
    g.use_bollinger_filter = False
    g.bollinger_period = 20
    g.bollinger_std = 2.0
    g.bollinger_lookback_days = 3
    
    # 持仓管理
    g.positions = {}
    g.position_highs = {}
    g.position_stop_prices = {}
    
    # ==================== 交易调度 ====================
    run_daily(etf_trade, time='14:20')
    run_daily(check_positions, time='09:30')
    run_daily(check_atr_stop_loss, time='09:30')

# ==================== 辅助函数：价格获取 ====================
def get_previous_minute_price(security, context):
    """
    获取前一分钟的价格，避免未来函数
    关键修复：使用前1分钟的价格，而不是当前价格
    """
    try:
        # 获取前1分钟的分钟数据
        # 注意：如果当前是9:31，前1分钟就是9:30的数据
        end_time = context.current_dt
        start_time = end_time - timedelta(minutes=2)  # 多取1分钟确保有数据
        
        # 获取分钟数据
        minute_data = get_price(
            security, 
            start_date=start_time, 
            end_date=end_time, 
            frequency='1m', 
            fields=['close'],
            skip_paused=False,
            fq='pre',
            panel=False
        )
        
        if minute_data is None or len(minute_data) < 2:
            # 如果没有足够的分钟数据，使用日线数据的昨天收盘价
            hist_data = attribute_history(security, 2, '1d', ['close'], skip_paused=True)
            if not hist_data.empty:
                return hist_data['close'].iloc[-1]
            return 0
        
        # 获取前1分钟的收盘价（倒数第二根K线）
        # 最后一根K线是当前分钟的，可能不完整
        if len(minute_data) >= 2:
            return minute_data['close'].iloc[-2]
        else:
            return minute_data['close'].iloc[-1]
            
    except Exception as e:
        log.warn(f"获取{security}前1分钟价格失败: {e}")
        # 失败时返回0，后续会处理
        return 0

def get_previous_minute_volume(security, context):
    """
    获取前一分钟的成交量
    """
    try:
        end_time = context.current_dt
        start_time = end_time - timedelta(minutes=2)
        
        minute_data = get_price(
            security,
            start_date=start_time,
            end_date=end_time,
            frequency='1m',
            fields=['volume'],
            skip_paused=False,
            fq='pre',
            panel=False
        )
        
        if minute_data is None or len(minute_data) < 2:
            # 使用日线数据
            hist_data = attribute_history(security, 2, '1d', ['volume'], skip_paused=True)
            if not hist_data.empty:
                return hist_data['volume'].iloc[-1]
            return 0
        
        if len(minute_data) >= 2:
            return minute_data['volume'].iloc[-2]
        else:
            return minute_data['volume'].iloc[-1]
            
    except Exception as e:
        log.warn(f"获取{security}前1分钟成交量失败: {e}")
        return 0

# ==================== 技术指标函数 ====================
def calculate_atr(security, period=14, context=None):
    """
    计算ATR指标，使用历史日线数据
    """
    try:
        needed_days = period + 20
        hist_data = attribute_history(security, needed_days, '1d', 
                                     ['high', 'low', 'close'], skip_paused=True)
        
        if len(hist_data) < period + 1:
            return 0, [], False, f"数据不足{period+1}天"
        
        high_prices = hist_data['high'].values
        low_prices = hist_data['low'].values
        close_prices = hist_data['close'].values
        
        tr_values = np.zeros(len(high_prices))
        
        for i in range(1, len(high_prices)):
            tr1 = high_prices[i] - low_prices[i]
            tr2 = abs(high_prices[i] - close_prices[i-1])
            tr3 = abs(low_prices[i] - close_prices[i-1])
            tr_values[i] = max(tr1, tr2, tr3)
        
        atr_values = np.zeros(len(tr_values))
        for i in range(period, len(tr_values)):
            atr_values[i] = np.mean(tr_values[i-period+1:i+1])
        
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0
        valid_atr = atr_values[period:] if len(atr_values) > period else atr_values
        
        return current_atr, valid_atr, True, "计算成功"
    
    except Exception as e:
        log.warn(f"计算{security} ATR时出错: {e}")
        return 0, [], False, f"计算出错:{str(e)}"

def calculate_bollinger_bands(prices, period=20, std_dev=2.0):
    """
    计算布林带指标
    """
    if len(prices) < period:
        return [], [], []
    
    middle_band = np.zeros(len(prices))
    upper_band = np.zeros(len(prices))
    lower_band = np.zeros(len(prices))
    
    for i in range(period - 1, len(prices)):
        window = prices[i-period+1:i+1]
        middle = np.mean(window)
        std = np.std(window)
        
        middle_band[i] = middle
        upper_band[i] = middle + std_dev * std
        lower_band[i] = middle - std_dev * std
    
    return middle_band[period-1:], upper_band[period-1:], lower_band[period-1:]

def calculate_rsi(prices, period=6):
    """
    计算RSI指标
    """
    if len(prices) < period + 1:
        return []
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)
    
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])
    
    rsi_values = np.zeros(len(prices))
    rsi_values[:period] = 50
    
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
        
        if avg_losses[i] == 0:
            rsi_values[i] = 100
        else:
            rs = avg_gains[i] / avg_losses[i]
            rsi_values[i] = 100 - (100 / (1 + rs))
    
    return rsi_values[period:]

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    计算MACD指标
    """
    if len(prices) < slow_period + signal_period:
        return [], [], []
    
    def calculate_ema(data, period):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        alpha = 2 / (period + 1)
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    dif = ema_fast - ema_slow
    dea = calculate_ema(dif, signal_period)
    macd_bar = dif - dea
    
    start_idx = slow_period + signal_period - 1
    return dif[start_idx:], dea[start_idx:], macd_bar[start_idx:]

# ==================== 过滤函数 ====================
def check_bollinger_filter(etf, context):
    """
    检查布林带过滤条件，使用前1分钟价格
    """
    try:
        needed_days = g.bollinger_period + g.bollinger_lookback_days + 10
        price_data = attribute_history(etf, needed_days, '1d', ['close'], skip_paused=True)
        
        if len(price_data) < g.bollinger_period:
            return True, f"数据不足{g.bollinger_period}天"
        
        close_prices = price_data['close'].values
        current_price = get_previous_minute_price(etf, context)
        
        if current_price <= 0:
            return True, "无法获取有效价格"
        
        middle_band, upper_band, lower_band = calculate_bollinger_bands(
            close_prices, g.bollinger_period, g.bollinger_std
        )
        
        if len(upper_band) < g.bollinger_lookback_days:
            return True, f"布林带数据不足{g.bollinger_lookback_days}天"
        
        recent_upper_band = upper_band[-g.bollinger_lookback_days:]
        recent_close_prices = close_prices[-(len(middle_band)-len(upper_band)+g.bollinger_lookback_days):][-g.bollinger_lookback_days:]
        
        breakthrough_occurred = False
        for i in range(len(recent_close_prices)):
            if recent_close_prices[i] > recent_upper_band[i]:
                breakthrough_occurred = True
                break
        
        if len(close_prices) >= 5:
            ma5 = np.mean(close_prices[-5:])
        else:
            ma5 = np.mean(close_prices)
        
        if breakthrough_occurred and current_price < ma5:
            return False, f"近{g.bollinger_lookback_days}日曾突破布林带上轨，且当前价{current_price:.3f}<MA5({ma5:.3f})"
        else:
            return True, "布林带检查通过"
    
    except Exception as e:
        log.warn(f"检查{etf}布林带时出错: {e}")
        return True, f"检查出错:{str(e)}"

def check_volume_anomaly(etf, context):
    """
    检查成交量是否异常，使用前1分钟成交量
    """
    if g.volume_exclude_defensive and etf == g.defensive_etf:
        return True, 0.0, 0, 0, "防御ETF豁免成交量检查"
    
    try:
        volume_lookback = g.volume_lookback_days + 5
        volume_data = attribute_history(etf, volume_lookback, '1d', ['volume'], skip_paused=True)
        
        if len(volume_data) < g.volume_lookback_days:
            return True, 0.0, 0, 0, f"数据不足{g.volume_lookback_days}天"
        
        # 获取前1分钟的成交量
        recent_volume = get_previous_minute_volume(etf, context)
        
        if len(volume_data) >= g.volume_lookback_days + 1:
            avg_volume = volume_data['volume'].iloc[-(g.volume_lookback_days+1):-1].mean()
        else:
            avg_volume = volume_data['volume'].iloc[:-1].mean()
        
        if avg_volume <= 0:
            return True, 0.0, recent_volume, avg_volume, f"历史均量异常:{avg_volume:.0f}"
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
        
        if volume_ratio > g.volume_threshold:
            return False, volume_ratio, recent_volume, avg_volume, f"成交量异常:近1分钟{recent_volume:.0f} > 近{g.volume_lookback_days}日均值{avg_volume:.0f}的{g.volume_threshold}倍"
        else:
            return True, volume_ratio, recent_volume, avg_volume, f"成交量正常:比值{volume_ratio:.2f}"
    
    except Exception as e:
        log.warn(f"检查{etf}成交量时出错: {e}")
        return True, 0.0, 0, 0, f"检查出错:{str(e)}"

# ==================== 核心计算函数 ====================
def calculate_momentum_metrics(etf, context):
    """
    计算ETF动量得分，使用前1分钟价格避免未来函数
    """
    try:
        lookback = max(g.lookback_days, g.short_lookback_days, g.ma_long_period,
                      g.rsi_period + g.rsi_lookback_days,
                      g.macd_slow_period + g.macd_signal_period,
                      g.volume_lookback_days,
                      g.bollinger_period + g.bollinger_lookback_days) + 20
        
        # 获取历史日线数据
        prices = attribute_history(etf, lookback, '1d', ['close'], skip_paused=True)
        
        if len(prices) < lookback:
            return None
        
        # 关键修复：使用前1分钟价格作为当前价格
        current_price = get_previous_minute_price(etf, context)
        if current_price <= 0:
            return None
        
        close_prices = prices["close"].values
        price_series = np.append(close_prices, current_price)
        
        # ========== 计算MA指标 ==========
        if len(price_series) >= g.ma_long_period:
            ma5 = np.mean(price_series[-g.ma_short_period:])
            ma25 = np.mean(price_series[-g.ma_long_period:])
            
            if g.ma_filter_condition == "above":
                ma_condition_met = ma5 >= ma25
                condition_desc = f"MA{g.ma_short_period}>={g.ma_long_period}"
            else:
                ma_condition_met = ma5 <= ma25
                condition_desc = f"MA{g.ma_short_period}<={g.ma_long_period}"
            
            ma_ratio = ma5 / ma25 - 1
        else:
            ma5 = 0
            ma25 = 0
            ma_condition_met = True
            ma_ratio = 0
            condition_desc = "数据不足"
        
        # ========== 计算RSI指标 ==========
        rsi_filter_pass = True
        current_rsi = 0
        max_rsi = 0
        rsi_info = "未启用RSI过滤或数据不足"
        
        if g.use_rsi_filter and len(price_series) >= g.rsi_period + g.rsi_lookback_days:
            rsi_values = calculate_rsi(price_series, g.rsi_period)
            
            if len(rsi_values) >= g.rsi_lookback_days:
                recent_rsi = rsi_values[-g.rsi_lookback_days:]
                rsi_ever_above_threshold = np.any(recent_rsi > g.rsi_threshold)
                current_below_ma5 = current_price < ma5 if ma5 > 0 else False
                
                if rsi_ever_above_threshold and current_below_ma5:
                    rsi_filter_pass = False
                    max_rsi = np.max(recent_rsi)
                    current_rsi = recent_rsi[-1] if len(recent_rsi) > 0 else 0
                    log.info(f"⛔ RSI过滤: {etf} 近{g.rsi_lookback_days}日RSI曾达{max_rsi:.1f}，当前价{current_price:.3f}<MA5，RSI={current_rsi:.1f}")
                else:
                    max_rsi = np.max(recent_rsi) if len(recent_rsi) > 0 else 0
                    current_rsi = recent_rsi[-1] if len(recent_rsi) > 0 else 0
                    rsi_info = f"RSI(max={max_rsi:.1f}, current={current_rsi:.1f})"
        
        # ========== 计算MACD指标 ==========
        macd_filter_pass = True
        dif_value = 0
        dea_value = 0
        macd_bar = 0
        macd_info = "未启用MACD过滤或数据不足"
        
        if g.use_macd_filter and len(price_series) >= g.macd_slow_period + g.macd_signal_period:
            dif_values, dea_values, macd_bars = calculate_macd(
                price_series, 
                g.macd_fast_period, 
                g.macd_slow_period, 
                g.macd_signal_period
            )
            
            if len(dif_values) > 0:
                dif_value = dif_values[-1]
                dea_value = dea_values[-1]
                macd_bar = macd_bars[-1]
                
                if g.macd_filter_condition == "bullish":
                    macd_condition_met = dif_value > dea_value
                    condition_desc = f"DIF({dif_value:.4f})>DEA({dea_value:.4f})"
                else:
                    macd_condition_met = dif_value < dea_value
                    condition_desc = f"DIF({dif_value:.4f})<DEA({dea_value:.4f})"
                
                macd_filter_pass = macd_condition_met
                macd_info = f"MACD(DIF={dif_value:.4f}, DEA={dea_value:.4f}, BAR={macd_bar:.4f})"
                
                if not macd_filter_pass:
                    log.info(f"📉 MACD过滤: {etf} 不满足{condition_desc}，MACD柱={macd_bar:.4f}")
        
        # ========== 检查成交量异常 ==========
        volume_filter_pass = True
        volume_ratio = 0
        recent_volume = 0
        avg_volume = 0
        volume_info = "未启用成交量过滤"
        
        if g.use_volume_filter:
            volume_filter_pass, volume_ratio, recent_volume, avg_volume, volume_info = check_volume_anomaly(
                etf, context
            )
            
            if not volume_filter_pass:
                log.info(f"📊 成交量过滤: {etf} {volume_info}")
        
        # ========== 检查布林带过滤条件 ==========
        bollinger_filter_pass = True
        bollinger_info = "未启用布林带过滤"
        
        if g.use_bollinger_filter:
            bollinger_filter_pass, bollinger_info = check_bollinger_filter(etf, context)
            
            if not bollinger_filter_pass:
                log.info(f"📈 布林带过滤: {etf} {bollinger_info}")
        
        # ========== 计算短期动量 ==========
        if len(price_series) >= g.short_lookback_days + 1:
            short_return = price_series[-1] / price_series[-(g.short_lookback_days + 1)] - 1
            short_annualized = (1 + short_return) ** (250 / g.short_lookback_days) - 1
        else:
            short_return = 0
            short_annualized = 0
        
        # ========== 计算长期动量得分 ==========
        recent_days = min(g.lookback_days, len(price_series) - 1)
        if recent_days >= 10:
            recent_price_series = price_series[-(recent_days+1):]
            y = np.log(recent_price_series)
            x = np.arange(len(y))
            weights = np.linspace(1, 2, len(y))
            
            slope, intercept = np.polyfit(x, y, 1, w=weights)
            annualized_returns = math.exp(slope * 250) - 1
            
            ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
            ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot else 0
            
            score = annualized_returns * r_squared
            
            # 近3日跌幅检查
            if len(price_series) >= 4:
                day1_ratio = price_series[-1] / price_series[-2]
                day2_ratio = price_series[-2] / price_series[-3]
                day3_ratio = price_series[-3] / price_series[-4]
                
                if min(day1_ratio, day2_ratio, day3_ratio) < g.loss:
                    score = 0
                    log.info(f"⚠️ {etf} 近3日有单日跌幅超{((1-g.loss)*100):.0f}%，已排除")
        else:
            annualized_returns = 0
            r_squared = 0
            score = 0
        
        return {
            'etf': etf,
            'annualized_returns': annualized_returns,
            'r_squared': r_squared,
            'score': score,
            'current_price': current_price,
            'short_return': short_return,
            'short_annualized': short_annualized,
            'short_momentum_pass': short_return >= g.short_momentum_threshold,
            'ma5': ma5,
            'ma25': ma25,
            'ma_condition_met': ma_condition_met,
            'ma_ratio': ma_ratio,
            'rsi_filter_pass': rsi_filter_pass,
            'current_rsi': current_rsi,
            'max_recent_rsi': max_rsi,
            'macd_filter_pass': macd_filter_pass,
            'dif': dif_value,
            'dea': dea_value,
            'macd_bar': macd_bar,
            'volume_filter_pass': volume_filter_pass,
            'volume_ratio': volume_ratio,
            'bollinger_filter_pass': bollinger_filter_pass,
            'bollinger_info': bollinger_info
        }
    except Exception as e:
        log.warn(f"计算{etf}动量指标时出错: {e}")
        return None

# ==================== 持仓管理函数 ====================
def check_positions(context):
    """每日开盘后检查持仓状态"""
    for security in context.portfolio.positions:
        position = context.portfolio.positions[security]
        if position.total_amount > 0:
            security_name = get_security_name(security)
            log.info(f"📊 持仓检查: {security} {security_name}, 数量: {position.total_amount}, 成本: {position.avg_cost:.3f}, 当前价: {position.price:.3f}")

def get_security_name(security):
    """获取证券名称"""
    current_data = get_current_data()
    return current_data[security].name if security in current_data else security

# ==================== ATR动态止损 ====================
def check_atr_stop_loss(context):
    """检查并执行ATR动态止损"""
    if not g.use_atr_stop_loss:
        return
    
    for security in list(context.portfolio.positions.keys()):
        if security not in g.etf_pool:
            continue
            
        position = context.portfolio.positions[security]
        if position.total_amount <= 0:
            continue
        
        if g.atr_exclude_defensive and security == g.defensive_etf:
            continue
        
        try:
            # 使用前1分钟价格作为当前价格
            current_price = get_previous_minute_price(security, context)
            if current_price <= 0:
                continue
            
            cost_price = position.avg_cost
            current_atr, _, success, _ = calculate_atr(security, g.atr_period)
            
            if not success:
                continue
            
            if security not in g.position_highs:
                g.position_highs[security] = current_price
            else:
                g.position_highs[security] = max(g.position_highs[security], current_price)
            
            position_high = g.position_highs[security]
            
            if g.atr_trailing_stop:
                atr_stop_price = position_high - g.atr_multiplier * current_atr
            else:
                atr_stop_price = cost_price - g.atr_multiplier * current_atr
            
            g.position_stop_prices[security] = atr_stop_price
            
            if current_price <= atr_stop_price:
                success = smart_order_target_value(security, 0, context)
                if success:
                    security_name = get_security_name(security)
                    loss_percent = (current_price/cost_price - 1) * 100
                    atr_stop_type = "跟踪" if g.atr_trailing_stop else "固定"
                    log.info(f"🚨 ATR动态止损({atr_stop_type})卖出: {security} {security_name}，成本: {cost_price:.3f}，现价: {current_price:.3f}，ATR: {current_atr:.3f}，止损价: {atr_stop_price:.3f}，亏损: {loss_percent:.2f}%")
                    
                    if security in g.position_highs:
                        del g.position_highs[security]
                    if security in g.position_stop_prices:
                        del g.position_stop_prices[security]
        
        except Exception as e:
            log.warn(f"检查{security} ATR止损时出错: {e}")

# ==================== 智能下单函数 ====================
def smart_order_target_value(security, target_value, context):
    """
    智能下单函数，使用前1分钟价格
    """
    current_data = get_current_data()
    
    if current_data[security].paused:
        log.info(f"{security} {get_security_name(security)}: 今日停牌，跳过交易")
        return False
    
    # 使用前1分钟价格作为参考
    current_price = get_previous_minute_price(security, context)
    if current_price == 0:
        log.info(f"{security} {get_security_name(security)}: 无法获取有效价格")
        return False
    
    # 检查涨跌停
    if current_price >= current_data[security].high_limit:
        log.info(f"{security} {get_security_name(security)}: 当前涨停，跳过买入")
        return False
    
    if current_price <= current_data[security].low_limit:
        log.info(f"{security} {get_security_name(security)}: 当前跌停，跳过卖出")
        return False
    
    # 计算目标数量
    target_amount = int(target_value / current_price)
    target_amount = (target_amount // 100) * 100
    if target_amount <= 0 and target_value > 0:
        target_amount = 100
    
    # 获取当前持仓
    current_position = context.portfolio.positions.get(security, None)
    current_amount = current_position.total_amount if current_position else 0
    amount_diff = target_amount - current_amount
    
    # 检查最小交易金额
    trade_value = abs(amount_diff) * current_price
    if 0 < trade_value < g.min_money:
        log.info(f"{security} {get_security_name(security)}: 交易金额{trade_value:.2f}小于最小交易额{g.min_money}")
        return False
    
    # 检查T+1限制
    if amount_diff < 0:
        closeable_amount = current_position.closeable_amount if current_position else 0
        if closeable_amount == 0:
            log.info(f"{security} {get_security_name(security)}: 当天买入不可卖出(T+1)")
            return False
        amount_diff = -min(abs(amount_diff), closeable_amount)
    
    # 执行下单
    if amount_diff != 0:
        order_result = order(security, amount_diff)
        if order_result:
            if security not in g.positions:
                g.positions[security] = 0
            g.positions[security] = target_amount
            
            if amount_diff > 0 and security in g.etf_pool:
                g.position_highs[security] = current_price
                
                if g.use_atr_stop_loss and not (g.atr_exclude_defensive and security == g.defensive_etf):
                    current_atr, _, success, _ = calculate_atr(security, g.atr_period)
                    if success:
                        if g.atr_trailing_stop:
                            g.position_stop_prices[security] = current_price - g.atr_multiplier * current_atr
                        else:
                            g.position_stop_prices[security] = current_price - g.atr_multiplier * current_atr
            
            security_name = get_security_name(security)
            if amount_diff > 0:
                log.info(f"📥 买入 {security} {security_name}，数量: {amount_diff}，价格: {current_price:.3f}，金额: {trade_value:.2f}")
            else:
                log.info(f"📤 卖出 {security} {security_name}，数量: {abs(amount_diff)}，价格: {current_price:.3f}，金额: {trade_value:.2f}")
            return True
        else:
            log.warn(f"下单失败: {security} {get_security_name(security)}，数量: {amount_diff}")
            return False
    
    return False

def is_defensive_etf_available(context):
    """检查防御性ETF是否可交易"""
    defensive_etf = g.defensive_etf
    
    if defensive_etf not in g.etf_pool:
        return False
    
    current_data = get_current_data()
    current_price = get_previous_minute_price(defensive_etf, context)
    
    if current_data[defensive_etf].paused:
        log.info(f"防御性ETF {defensive_etf} {get_security_name(defensive_etf)} 今日停牌")
        return False
    
    if current_price >= current_data[defensive_etf].high_limit:
        log.info(f"防御性ETF {defensive_etf} {get_security_name(defensive_etf)} 当前涨停")
        return False
    
    if current_price <= current_data[defensive_etf].low_limit:
        log.info(f"防御性ETF {defensive_etf} {get_security_name(defensive_etf)} 当前跌停")
        return False
    
    return True

# ==================== 主交易函数 ====================
def get_ranked_etfs(context):
    """获取排名ETF，使用前1分钟价格"""
    etf_metrics = []
    for etf in g.etf_pool:
        metrics = calculate_momentum_metrics(etf, context)
        if metrics is not None:
            if g.use_short_momentum_filter and not metrics['short_momentum_pass']:
                log.info(f"📉 排除短期动量不足的ETF: {etf}，短期动量: {metrics['short_return']:.4f}")
                continue
            
            if g.use_ma_filter and not metrics['ma_condition_met']:
                log.info(f"📊 排除MA条件不符的ETF: {etf}，MA{g.ma_short_period}: {metrics['ma5']:.3f}，MA{g.ma_long_period}: {metrics['ma25']:.3f}")
                continue
            
            if g.use_rsi_filter and not metrics['rsi_filter_pass']:
                continue
            
            if g.use_macd_filter and not metrics['macd_filter_pass']:
                continue
            
            if g.use_volume_filter and not metrics['volume_filter_pass']:
                continue
            
            if g.use_bollinger_filter and not metrics['bollinger_filter_pass']:
                continue
            
            if 0 < metrics['score'] < g.max_score_threshold:
                etf_metrics.append(metrics)
            else:
                log.info(f"排除异常值ETF: {etf}，得分: {metrics['score']:.4f}")
    
    etf_metrics.sort(key=lambda x: x['score'], reverse=True)
    return etf_metrics

def etf_trade(context):
    """ETF轮动交易主函数"""
    ranked_etfs = get_ranked_etfs(context)
    
    log.info("=== ETF趋势指标分析 ===")
    for metrics in ranked_etfs:
        etf_name = get_security_name(metrics['etf'])
        bollinger_status = metrics['bollinger_info'] if g.use_bollinger_filter else "未启用"
        log.info(f"{metrics['etf']} {etf_name}: 年化={metrics['annualized_returns']:.4f}, R²={metrics['r_squared']:.4f}, 得分={metrics['score']:.4f}, 短期动量={metrics['short_return']:.4f}, MA{g.ma_short_period}={metrics['ma5']:.3f}, MA{g.ma_long_period}={metrics['ma25']:.3f}, 成交量比={metrics['volume_ratio']:.2f}, RSI={metrics['current_rsi']:.1f}, MACD(DIF={metrics['dif']:.4f}), 布林带={bollinger_status}, 当前价={metrics['current_price']:.3f}")
    
    target_etf = None
    if ranked_etfs and ranked_etfs[0]['score'] >= g.min_score_threshold:
        target_etf = ranked_etfs[0]['etf']
        top_metrics = ranked_etfs[0]
        etf_name = get_security_name(target_etf)
        log.info(f"🎯 正常模式，选择得分最高的ETF: {target_etf} {etf_name}，得分: {top_metrics['score']:.4f}，短期动量: {top_metrics['short_return']:.4f}")
    else:
        if is_defensive_etf_available(context):
            target_etf = g.defensive_etf
            etf_name = get_security_name(target_etf)
            log.info(f"🛡️ 进入防御模式，选择防御ETF: {target_etf} {etf_name}")
        else:
            log.info("💤 进入空仓模式")
    
    target_etfs = [target_etf] if target_etf else []
    
    # 固定百分比止损
    for security in list(context.portfolio.positions.keys()):
        position = context.portfolio.positions[security]
        if security in g.etf_pool and position.total_amount > 0:
            current_price = get_previous_minute_price(security, context)
            cost_price = position.avg_cost
            
            if current_price <= cost_price * g.stop_loss:
                success = smart_order_target_value(security, 0, context)
                if success:
                    security_name = get_security_name(security)
                    loss_percent = (current_price/cost_price-1)*100
                    log.info(f"🚨 固定百分比止损卖出: {security} {security_name}，成本: {cost_price:.3f}，现价: {current_price:.3f}，亏损: {loss_percent:.2f}%")
                    
                    if security in g.position_highs:
                        del g.position_highs[security]
                    if security in g.position_stop_prices:
                        del g.position_stop_prices[security]
    
    # 调仓逻辑
    total_value = context.portfolio.total_value
    target_value = total_value if target_etfs else 0
    
    current_positions = set(context.portfolio.positions.keys())
    target_etfs_set = set(target_etfs)
    
    for security in current_positions:
        if security in g.etf_pool and security not in target_etfs_set:
            position = context.portfolio.positions[security]
            if position.total_amount > 0:
                success = smart_order_target_value(security, 0, context)
                if success:
                    security_name = get_security_name(security)
                    log.info(f"📤 卖出: {security} {security_name} (不在目标列表中)")
                    
                    if security in g.position_highs:
                        del g.position_highs[security]
                    if security in g.position_stop_prices:
                        del g.position_stop_prices[security]
    
    for etf in target_etfs:
        current_value = 0
        if etf in context.portfolio.positions:
            position = context.portfolio.positions[etf]
            if position.total_amount > 0:
                current_value = position.total_amount * get_previous_minute_price(etf, context)
        
        if abs(current_value - target_value) > target_value * 0.05 or current_value == 0:
            success = smart_order_target_value(etf, target_value, context)
            if success:
                action = "买入" if current_value < target_value else "调仓"
                etf_name = get_security_name(etf)
                log.info(f"📦 {action}: {etf} {etf_name}，目标金额: {target_value:.2f}")

def trade(context):
    """主交易函数，为了兼容性保留"""
    etf_trade(context)