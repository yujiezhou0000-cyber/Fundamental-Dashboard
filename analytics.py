"""
指标计算层：动量、鹰鸽矩阵、配对推荐、利差、货币强弱。
这里只做计算，不做任何网络请求，方便单独测试。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import COUNTRIES


def value_on_or_before(s: pd.Series, when: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """取某日（含）之前最近一个交易日的值；没有则返回 None。"""
    sub = s[s.index <= when]
    if sub.empty:
        return None
    return sub.index[-1], float(sub.iloc[-1])


def momentum(s: pd.Series, days: int) -> dict:
    """最新值与 N 个自然日前的值之差（基点 bp）。"""
    last_date, last_val = s.index[-1], float(s.iloc[-1])
    past = value_on_or_before(s, last_date - pd.Timedelta(days=days))
    if past is None:
        return dict(last_date=last_date, last=last_val, past_date=None, past=np.nan, chg_bp=np.nan)
    return dict(last_date=last_date, last=last_val, past_date=past[0], past=past[1],
                chg_bp=(last_val - past[1]) * 100)


def hawk_dove_matrix(yields: dict, days: int) -> pd.DataFrame:
    """
    模块一：鹰鸽矩阵。按收益率从高到低排序，
    最高者标“最鹰派（利差最高）”，最低者标“最鸽派（利差最低）”。
    yields: {code: YieldResult}
    """
    rows = []
    for code, res in yields.items():
        meta = COUNTRIES[code]
        base = dict(代码=code, 经济体=meta["name"], 货币=meta["ccy"], 期限=meta["tenor_note"])
        if not res.ok:
            rows.append({**base, "状态": "获取失败"})
            continue
        m = momentum(res.series, days)
        rows.append({**base,
                     "最新收益率(%)": m["last"],
                     f"{days}日前(%)": m["past"],
                     f"{days}日变化(bp)": m["chg_bp"],
                     "最新日期": m["last_date"].date(),
                     "数据源": res.source,
                     "状态": "正常"})
    df = pd.DataFrame(rows)
    ok = df[df["状态"] == "正常"].sort_values("最新收益率(%)", ascending=False).reset_index(drop=True)
    bad = df[df["状态"] != "正常"]
    ok["排名"] = np.arange(1, len(ok) + 1)
    ok["鹰鸽标签"] = ""
    if len(ok) >= 2:
        ok.loc[0, "鹰鸽标签"] = "最鹰派（利差最高）"
        ok.loc[len(ok) - 1, "鹰鸽标签"] = "最鸽派（利差最低）"
    chg = f"{days}日变化(bp)"
    if chg in ok and ok[chg].notna().any():
        ok["动量"] = np.where(ok[chg] > 5, "上行", np.where(ok[chg] < -5, "下行", "持平"))
    return pd.concat([ok, bad], ignore_index=True)


def best_pairs(matrix: pd.DataFrame, days: int) -> dict:
    """
    模块二：配对推荐。
    - 利差最分化：收益率最高的货币 vs 最低的货币
    - 动量最分化：近 N 日收益率上行最多 vs 下行最多（或上行最少）
    挪威用的是 3 年期，默认不参与配对，避免期限不一致。
    """
    ok = matrix[(matrix["状态"] == "正常") & (matrix["货币"] != "NOK")]
    if len(ok) < 2:
        return {}
    hi, lo = ok.iloc[0], ok.iloc[-1]
    out = dict(
        long_ccy=hi["货币"], short_ccy=lo["货币"],
        long_yield=hi["最新收益率(%)"], short_yield=lo["最新收益率(%)"],
        spread_bp=(hi["最新收益率(%)"] - lo["最新收益率(%)"]) * 100,
    )
    chg = f"{days}日变化(bp)"
    m = ok.dropna(subset=[chg]).sort_values(chg, ascending=False)
    if len(m) >= 2:
        out.update(mom_long=m.iloc[0]["货币"], mom_short=m.iloc[-1]["货币"],
                   mom_long_chg=m.iloc[0][chg], mom_short_chg=m.iloc[-1][chg])
    return out


def spread_series(a: pd.Series, b: pd.Series, max_fill_days: int = 3) -> pd.DataFrame:
    """
    模块三：两国收益率对齐后计算利差（bp）。
    各国假期不同，先按并集日期对齐，再最多向前填充 3 天。
    """
    df = pd.concat({"a": a, "b": b}, axis=1).sort_index().ffill(limit=max_fill_days).dropna()
    df["spread_bp"] = (df["a"] - df["b"]) * 100
    return df


def currency_strength(fx: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    货币强弱：每个货币对美元的 N 日涨跌幅，再减去所有 G10 货币（含美元=0）的平均值，
    得到“相对一篮子货币”的强弱。正数 = 强于一篮子。
    fx: 列为货币，值为 1 单位该货币值多少美元（美元列恒为 1）。
    """
    if fx.empty:
        return pd.DataFrame()
    last = fx.ffill().iloc[-1]
    past_idx = fx.index[fx.index <= fx.index[-1] - pd.Timedelta(days=days)]
    if len(past_idx) == 0:
        return pd.DataFrame()
    past = fx.ffill().loc[past_idx[-1]]
    ret = (last / past - 1) * 100
    strength = ret - ret.mean()
    return (pd.DataFrame({"对美元涨跌(%)": ret, "相对一篮子强弱(%)": strength})
            .sort_values("相对一篮子强弱(%)", ascending=False))


def strength_index(fx: pd.DataFrame, rebase_days: int) -> pd.DataFrame:
    """把每个货币相对一篮子的强弱做成时间序列（起点 = 100），用于走势图。"""
    if fx.empty:
        return pd.DataFrame()
    fx = fx.ffill().dropna()
    start = fx.index[-1] - pd.Timedelta(days=rebase_days)
    fx = fx[fx.index >= start]
    logret = np.log(fx / fx.iloc[0])
    rel = logret.sub(logret.mean(axis=1), axis=0)
    return 100 * np.exp(rel)


def macro_vs_fx(matrix: pd.DataFrame, strength: pd.DataFrame, days: int) -> pd.DataFrame:
    """把收益率动量（相对 G10 平均）和汇率强弱放在一起，看汇率有没有跟着利差走。"""
    chg = f"{days}日变化(bp)"
    ok = matrix[matrix["状态"] == "正常"].set_index("货币")
    if ok.empty or strength.empty or chg not in ok:
        return pd.DataFrame()
    rel_chg = ok[chg] - ok[chg].mean()
    df = pd.DataFrame({"收益率相对动量(bp)": rel_chg}).join(strength["相对一篮子强弱(%)"], how="inner")
    return df.dropna()
