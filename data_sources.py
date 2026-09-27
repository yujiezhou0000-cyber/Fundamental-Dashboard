"""
数据获取层：各国 2 年期国债收益率（官方接口）+ G10 汇率（yfinance）。

设计要点：
1. 每个国家配置一个“数据源列表”，按顺序尝试，前一个失败自动换下一个；
2. 每个抓取函数都包在 try/except 里，任何一个接口断连只会让该国显示为“获取失败”，
   不会让整个程序崩溃；
3. 所有函数统一返回 pandas.Series：索引为日期（Timestamp，已排序去重），值为收益率（%）。
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd
import requests

from config import COUNTRIES, FX_TICKERS, HTTP_HEADERS, HTTP_TIMEOUT

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
_session = requests.Session()
_session.headers.update(HTTP_HEADERS)


def http_get(url: str, **kwargs) -> requests.Response:
    """带超时和状态码检查的 GET 请求，失败直接抛异常，由上层捕获。"""
    resp = _session.get(url, timeout=HTTP_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp


def _clean(s: pd.Series, start: date | None = None) -> pd.Series:
    """统一清洗：转数值、去空值、按日期排序去重、截取起始日期之后。"""
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if s.empty:
        raise ValueError("清洗后没有有效数据")
    # 简单合理性检查：收益率应在 -5% 到 30% 之间
    if (s.abs() > 30).any():
        raise ValueError("数值超出合理范围，可能解析错了列")
    return s.astype(float)


def _find_header_line(lines: list[str], must_contain: list[str]) -> int:
    """在文本行中找到同时包含指定关键字的表头行，返回行号。"""
    for i, line in enumerate(lines):
        low = line.lower()
        if all(k.lower() in low for k in must_contain):
            return i
    raise ValueError(f"找不到包含 {must_contain} 的表头行")


# ---------------------------------------------------------------------------
# 美国：FRED DGS2 -> 美国财政部收益率曲线 -> yfinance 2 年期收益率期货（近似）
# ---------------------------------------------------------------------------
def fetch_us_fred(start: date) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2&cosd={start:%Y-%m-%d}"
    df = pd.read_csv(io.StringIO(http_get(url).text))
    date_col = df.columns[0]                      # observation_date 或 DATE
    return _clean(df.set_index(date_col)["DGS2"], start)


def fetch_us_treasury(start: date) -> pd.Series:
    parts = []
    for year in range(start.year, date.today().year + 1):
        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            f"daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
            f"&field_tdr_date_value={year}&page&_format=csv"
        )
        df = pd.read_csv(io.StringIO(http_get(url).text))
        parts.append(df.set_index(pd.to_datetime(df["Date"]))["2 Yr"])
    return _clean(pd.concat(parts), start)


def fetch_us_yfinance(start: date) -> pd.Series:
    """Yahoo 上唯一能直接拿到的美国 2 年期收益率：CBOT 2 年期收益率期货 2YY=F（近似值）。"""
    import yfinance as yf

    df = yf.download("2YY=F", start=start, progress=False, auto_adjust=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return _clean(close, start)


# ---------------------------------------------------------------------------
# 欧元区：德国联邦银行（德债 2 年期）-> 欧洲央行 AAA 级欧元区国债 2 年期即期利率
# ---------------------------------------------------------------------------
def parse_sdmx_json(js: dict) -> pd.Series:
    """
    解析 SDMX-JSON（德国央行 sdmx_json、挪威央行 sdmx-json 已实测可用）。
    兼容 1.0（dataSets / structure 在顶层）和 2.0（data.dataSets / data.structures）两种布局，
    取第一条序列；缺失值（周末等）记为空。
    """
    root = js.get("data", js)
    ds_ = root["dataSets"][0]
    struct = root.get("structure") or (root.get("structures") or [None])[0] or js.get("structure")
    periods = [v.get("id") or v.get("start") for v in struct["dimensions"]["observation"][0]["values"]]
    series = ds_.get("series") or {}
    if not series:
        raise ValueError("SDMX-JSON 里没有序列")
    obs = next(iter(series.values()))["observations"]
    dates, vals = [], []
    for k, arr in obs.items():
        v = arr[0] if isinstance(arr, list) and arr else None
        dates.append(str(periods[int(k)])[:10])
        vals.append(v)
    return pd.Series(vals, index=pd.to_datetime(dates))


def _parse_date_value_lines(text: str, sep: str = ",") -> pd.Series:
    """解析“日期,数值,...”格式的行，跳过元数据行（德国央行 CSV 前几行是说明）。"""
    dates, vals = [], []
    for line in text.splitlines():
        parts = [p.strip().strip('"') for p in line.split(sep)]
        if len(parts) < 2 or not re.match(r"^\d{4}-\d{2}-\d{2}$", parts[0]):
            continue
        dates.append(parts[0])
        vals.append(parts[1])
    if not dates:
        raise ValueError("没有找到 日期,数值 形式的数据行")
    return pd.Series(vals, index=pd.to_datetime(dates))


def fetch_de_bundesbank(start: date) -> pd.Series:
    """德国央行：上市联邦证券收益率曲线（Svensson 法）2.0 年剩余期限，日度。"""
    key = "D.I.ZST.ZI.EUR.S1311.B.A604.R02XX.R.A.A._Z._Z.A"
    base = f"https://api.statistiken.bundesbank.de/rest/data/BBSIS/{key}"
    try:
        js = http_get(f"{base}?format=sdmx_json&startPeriod={start:%Y-%m-%d}").json()
        return _clean(parse_sdmx_json(js), start)
    except Exception:
        text = http_get(f"{base}?format=csv&lang=en&startPeriod={start:%Y-%m-%d}").content.decode("utf-8-sig", "replace")
        return _clean(_parse_date_value_lines(text), start)


def fetch_eu_ecb(start: date) -> pd.Series:
    url = (
        "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y"
        f"?format=csvdata&startPeriod={start:%Y-%m-%d}"
    )
    df = pd.read_csv(io.StringIO(http_get(url).text))
    return _clean(df.set_index("TIME_PERIOD")["OBS_VALUE"], start)


# ---------------------------------------------------------------------------
# 英国：英国央行收益率曲线文件（名义即期曲线中的 2 年期）
# 英国央行数据库 IADB 没有 2 年期序列，只能从收益率曲线 zip 包里取
# ---------------------------------------------------------------------------
BOE_ZIPS = [
    "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/latest-yield-curve-data.zip",
    "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/glcnominalddata.zip",
]


def _boe_spot_2y_from_xlsx(content: bytes) -> pd.Series:
    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    name = next((n for n in sheets if "spot" in n.lower()), None)
    if name is None:
        raise ValueError("文件里没有 spot curve 工作表")
    raw = sheets[name]
    # 找到期限表头行：该行中能找到数值 2（年）
    for r in range(min(15, len(raw))):
        row = pd.to_numeric(raw.iloc[r], errors="coerce")
        hits = np.where(np.isclose(row.values.astype(float), 2.0, equal_nan=False))[0]
        if len(hits) and row.notna().sum() > 5:
            col = hits[0]
            body = raw.iloc[r + 1:, [0, col]].dropna()
            body = body[pd.to_datetime(body.iloc[:, 0], errors="coerce", format="mixed").notna()]
            return pd.Series(body.iloc[:, 1].values, index=pd.to_datetime(body.iloc[:, 0], format="mixed"))
    raise ValueError("找不到 2 年期所在的列")


def fetch_gb_boe(start: date) -> pd.Series:
    parts, errors = [], []
    for url in BOE_ZIPS:
        try:
            zf = zipfile.ZipFile(io.BytesIO(http_get(url).content))
            for n in zf.namelist():
                low = n.lower()
                if "nominal" in low and "daily" in low and low.endswith((".xlsx", ".xls")):
                    try:
                        parts.append(_boe_spot_2y_from_xlsx(zf.read(n)))
                    except Exception as e:  # 单个文件解析失败不影响其他文件
                        errors.append(f"{n}: {e}")
        except Exception as e:
            errors.append(f"{url}: {e}")
    if not parts:
        raise ValueError("英国央行数据解析失败：" + " | ".join(errors[:3]))
    return _clean(pd.concat(parts), start)


# ---------------------------------------------------------------------------
# 日本：日本财务省 JGB 利率（当月文件 + 历史文件）
# ---------------------------------------------------------------------------
MOF_URLS = [
    "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/jgbcme.csv",
    "https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/historical/jgbcme_all.csv",
]


def _parse_mof(text: str) -> pd.Series:
    lines = text.splitlines()
    h = _find_header_line(lines, ["Date", "2Y"])
    df = pd.read_csv(io.StringIO("\n".join(lines[h:])), na_values=["-", ""])
    idx = pd.to_datetime(df["Date"], format="%Y/%m/%d", errors="coerce")
    return pd.Series(df["2Y"].values, index=idx).loc[idx.notna().values]


def fetch_jp_mof(start: date) -> pd.Series:
    parts, errors = [], []
    for url in MOF_URLS:
        try:
            resp = http_get(url)
            parts.append(_parse_mof(resp.content.decode("utf-8", errors="replace")))
        except Exception as e:
            errors.append(f"{url}: {e}")
    if not parts:
        raise ValueError("日本财务省数据获取失败：" + " | ".join(errors))
    return _clean(pd.concat(parts), start)


# ---------------------------------------------------------------------------
# 澳大利亚：澳洲联储 F2 表（FCMYGBAG2 = 2 年期国债）
# ---------------------------------------------------------------------------
def fetch_au_rba(start: date) -> pd.Series:
    url = "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv"
    text = http_get(url).content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    h = _find_header_line(lines, ["Series ID"])
    df = pd.read_csv(io.StringIO("\n".join(lines[h:])))
    col = "FCMYGBAG2" if "FCMYGBAG2" in df.columns else None
    if col is None:
        raise ValueError(f"找不到 FCMYGBAG2 列，现有列：{list(df.columns)[:8]}")
    idx = pd.to_datetime(df.iloc[:, 0], format="%d-%b-%Y", errors="coerce")
    return _clean(pd.Series(df[col].values, index=idx).loc[idx.notna().values], start)


# ---------------------------------------------------------------------------
# 加拿大：加拿大央行 Valet 接口（BD.CDN.2YR.DQ.YLD = 2 年期基准国债）
# ---------------------------------------------------------------------------
def fetch_ca_boc(start: date) -> pd.Series:
    sid = "BD.CDN.2YR.DQ.YLD"
    url = f"https://www.bankofcanada.ca/valet/observations/{sid}/json?start_date={start:%Y-%m-%d}"
    obs = http_get(url).json()["observations"]
    return _clean(pd.Series({o["d"]: o.get(sid, {}).get("v") for o in obs}), start)


# ---------------------------------------------------------------------------
# 瑞士：瑞士央行数据门户 rendoblid（联邦债券收益率，日度）
# ---------------------------------------------------------------------------
def fetch_ch_snb(start: date) -> pd.Series:
    """瑞士央行 rendoblid。实测带 fromDate 的请求可能返回空序列，所以空了就不带日期再取一次。"""
    errors = []
    for url in (f"https://data.snb.ch/api/cube/rendoblid/data/csv/en?fromDate={start:%Y-%m-%d}",
                "https://data.snb.ch/api/cube/rendoblid/data/csv/en"):
        try:
            return _clean(_parse_snb_csv(http_get(url).content.decode("utf-8-sig", errors="replace")), start)
        except Exception as e:
            errors.append(f"{url.split('?')[-1] if '?' in url else '全量'}: {e}")
    raise ValueError("瑞士央行数据为空或解析失败：" + " | ".join(errors))


def _parse_snb_csv(text: str) -> pd.Series:
    lines = text.splitlines()
    h = _find_header_line(lines, ["Date", "Value"])
    df = pd.read_csv(io.StringIO("\n".join(lines[h:])), sep=";")
    dims = [c for c in df.columns if c not in ("Date", "Value")]
    # 2 年期的项目代码一般是 "2J"（德语 Jahre）；兼容 "2Y"
    mask = pd.Series(False, index=df.index)
    for d in dims:
        mask |= df[d].astype(str).str.fullmatch(r"(?i)2\s*[JY]")
    if not mask.any():
        codes = sorted({str(v) for d in dims for v in df[d].unique()})[:20]
        raise ValueError(f"找不到 2 年期代码，现有代码：{codes}")
    sub = df[mask]
    return pd.Series(sub["Value"].values, index=pd.to_datetime(sub["Date"]))


# ---------------------------------------------------------------------------
# 新西兰：新西兰联储 B2 批发利率日度收盘文件（政府债券 2 年期）
# ---------------------------------------------------------------------------
RBNZ_URLS = [
    "https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/series/b/b2/hb2-daily-close.xlsx",
    "https://www.rbnz.govt.nz/-/media/project/sites/rbnz/files/statistics/series/b/b2/hb2-daily.xlsx",
]


def _rbnz_2y(content: bytes) -> pd.Series:
    raw = pd.read_excel(io.BytesIO(content), sheet_name=0, header=None)
    head = raw.head(12).astype(str).apply(lambda c: " ".join(c).lower())
    # 表头里同时出现 “2 year” 和 “bond / govt” 的那一列
    cand = [i for i, t in head.items()
            if re.search(r"\b2[\s-]*(year|yr)", t) and re.search(r"bond|govt|government", t)]
    if not cand:
        raise ValueError("找不到政府债券 2 年期所在的列")
    col = cand[0]
    idx = pd.to_datetime(raw.iloc[:, 0], errors="coerce", format="mixed")
    ok = idx.notna().values
    return pd.Series(raw.iloc[:, col].values[ok], index=idx[ok])


def fetch_nz_rbnz(start: date) -> pd.Series:
    errors = []
    for url in RBNZ_URLS:
        try:
            return _clean(_rbnz_2y(http_get(url).content), start)
        except Exception as e:
            errors.append(f"{url.rsplit('/', 1)[-1]}: {e}")
    raise ValueError("新西兰联储数据获取失败：" + " | ".join(errors))


# ---------------------------------------------------------------------------
# 瑞典：瑞典央行 SWEA 接口（SEGVB2YC = 2 年期国债）
# ---------------------------------------------------------------------------
def fetch_se_riksbank(start: date) -> pd.Series:
    url = f"https://api.riksbank.se/swea/v1/Observations/SEGVB2YC/{start:%Y-%m-%d}/{date.today():%Y-%m-%d}"
    data = http_get(url).json()
    return _clean(pd.Series({d["date"]: d["value"] for d in data}), start)


# ---------------------------------------------------------------------------
# 挪威：挪威央行通用国债收益率（没有 2 年期，用 3 年期）
# ---------------------------------------------------------------------------
def fetch_no_norgesbank(start: date) -> pd.Series:
    errors = []
    try:
        url = (f"https://data.norges-bank.no/api/data/GOVT_GENERIC_RATES/B.3Y.GBON."
               f"?format=sdmx-json&startPeriod={start:%Y-%m-%d}&locale=en")
        return _clean(parse_sdmx_json(http_get(url).json()), start)
    except Exception as e:
        errors.append(f"sdmx-json: {e}")
    for key in ("B.3Y.GBON.", "B..GBON."):
        try:
            url = (f"https://data.norges-bank.no/api/data/GOVT_GENERIC_RATES/{key}"
                   f"?format=csv&startPeriod={start:%Y-%m-%d}&locale=en")
            text = http_get(url).content.decode("utf-8-sig", errors="replace")
            sep = ";" if text.count(";") > text.count(",") else ","
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if "TENOR" in df.columns:
                df = df[df["TENOR"].astype(str).str.upper().str.startswith("3Y")]
            return _clean(df.set_index("TIME_PERIOD")["OBS_VALUE"], start)
        except Exception as e:
            errors.append(f"{key}: {e}")
    raise ValueError("挪威央行数据获取失败：" + " | ".join(errors))


# ---------------------------------------------------------------------------
# 数据源注册表：按顺序尝试
# ---------------------------------------------------------------------------
YIELD_SOURCES: dict[str, list[tuple[str, Callable[[date], pd.Series]]]] = {
    "US": [("美联储 FRED（DGS2）", fetch_us_fred),
           ("美国财政部收益率曲线", fetch_us_treasury),
           ("yfinance 2YY=F（期货，近似）", fetch_us_yfinance)],
    "DE": [("德国央行（德债 2Y）", fetch_de_bundesbank),
           ("欧洲央行 AAA 欧元区 2Y（近似德债）", fetch_eu_ecb)],
    "GB": [("英国央行收益率曲线", fetch_gb_boe)],
    "JP": [("日本财务省", fetch_jp_mof)],
    "AU": [("澳洲联储 F2", fetch_au_rba)],
    "CA": [("加拿大央行 Valet", fetch_ca_boc)],
    "CH": [("瑞士央行数据门户", fetch_ch_snb)],
    "NZ": [("新西兰联储 B2", fetch_nz_rbnz)],
    "SE": [("瑞典央行 SWEA", fetch_se_riksbank)],
    "NO": [("挪威央行（3Y 通用收益率）", fetch_no_norgesbank)],
}


@dataclass
class YieldResult:
    code: str
    series: pd.Series | None
    source: str | None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.series is not None and not self.series.empty


def get_yield(code: str, start: date) -> YieldResult:
    """按顺序尝试该国所有数据源，返回第一个成功的结果；全部失败则记录错误。"""
    errors = []
    for name, fn in YIELD_SOURCES[code]:
        try:
            s = fn(start)
            s.name = code
            return YieldResult(code, s, name, errors)
        except Exception as e:  # 任何异常都只记录，不中断
            errors.append(f"{name}：{type(e).__name__}: {str(e)[:160]}")
    return YieldResult(code, None, None, errors)


def load_all_yields(start: date) -> dict[str, YieldResult]:
    return {code: get_yield(code, start) for code in COUNTRIES}


# ---------------------------------------------------------------------------
# 汇率（yfinance）：换算成“1 单位该货币值多少美元”
# ---------------------------------------------------------------------------
def load_fx_usd_values(start: date) -> tuple[pd.DataFrame, dict[str, str]]:
    import yfinance as yf

    out, errors = {}, {}
    for ccy, (ticker, kind) in FX_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=False)
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna()
            if close.empty:
                raise ValueError("没有数据")
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            out[ccy] = close if kind == "direct" else 1.0 / close
        except Exception as e:
            errors[ccy] = f"{ticker}：{type(e).__name__}: {str(e)[:120]}"
    fx = pd.DataFrame(out).sort_index()
    if not fx.empty:
        fx["USD"] = 1.0
    return fx, errors
