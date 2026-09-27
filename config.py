"""
全局配置：G10 主要货币、对应经济体、汇率代码。

说明：
- 2 年期国债收益率不走 yfinance（Yahoo Finance 没有这些国家的 2 年期收益率代码，
  US02Y / DE02Y / JP02Y 等是 TradingView 的代码），改用各国央行 / 财政部的官方免费接口，
  具体见 data_sources.py。
- 汇率走 yfinance，用于计算货币强弱。
"""

# 各经济体：代码 -> 元数据
# tenor_note：该国实际使用的期限（挪威没有 2 年期基准，用 3 年期代替）
COUNTRIES = {
    "US": {"name": "美国", "ccy": "USD", "tenor_note": "2Y"},
    "DE": {"name": "欧元区（德国）", "ccy": "EUR", "tenor_note": "2Y"},
    "GB": {"name": "英国", "ccy": "GBP", "tenor_note": "2Y"},
    "JP": {"name": "日本", "ccy": "JPY", "tenor_note": "2Y"},
    "AU": {"name": "澳大利亚", "ccy": "AUD", "tenor_note": "2Y"},
    "CA": {"name": "加拿大", "ccy": "CAD", "tenor_note": "2Y"},
    "CH": {"name": "瑞士", "ccy": "CHF", "tenor_note": "2Y"},
    "NZ": {"name": "新西兰", "ccy": "NZD", "tenor_note": "2Y"},
    "SE": {"name": "瑞典", "ccy": "SEK", "tenor_note": "2Y"},
    "NO": {"name": "挪威", "ccy": "NOK", "tenor_note": "3Y*"},
}

# yfinance 汇率代码：货币 -> (Ticker, 报价方向)
# "direct"  表示报价为 1 单位该货币 = 多少美元（如 EURUSD）
# "inverse" 表示报价为 1 美元 = 多少单位该货币（如 USDJPY，Yahoo 代码 JPY=X）
FX_TICKERS = {
    "EUR": ("EURUSD=X", "direct"),
    "GBP": ("GBPUSD=X", "direct"),
    "AUD": ("AUDUSD=X", "direct"),
    "NZD": ("NZDUSD=X", "direct"),
    "JPY": ("JPY=X", "inverse"),
    "CAD": ("CAD=X", "inverse"),
    "CHF": ("CHF=X", "inverse"),
    "SEK": ("SEK=X", "inverse"),
    "NOK": ("NOK=X", "inverse"),
}

# 默认参数
DEFAULT_MOMENTUM_DAYS = 30      # 动量回看天数（自然日）
DEFAULT_HISTORY_MONTHS = 6      # 历史走势窗口（月）
HTTP_TIMEOUT = 25               # 单个请求超时（秒）

# 缓存时间（秒）：官方收益率每天只更新一次，30 分钟重新检查一次足够；
# 汇率是盘中行情，5 分钟刷新一次
YIELD_TTL_SECONDS = 30 * 60
FX_TTL_SECONDS = 5 * 60

# 页面自动刷新间隔（分钟），0 = 关闭
AUTO_REFRESH_OPTIONS = [0, 5, 15, 30, 60]
DEFAULT_AUTO_REFRESH = 15

# 超过这么多个工作日没有新数据，就标记为“可能过期”
STALE_BUSINESS_DAYS = 3

# 页面显示的时区
DISPLAY_TZ = "Europe/London"

# 部分官方网站会拒绝没有浏览器 UA / Accept-Language 的请求（如新西兰联储、瑞士央行），
# 云端服务器的出口 IP 段也更容易被这类反爬机制拦截，所以尽量把请求头伪装得更像真实浏览器。
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}
