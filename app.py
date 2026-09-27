"""
外汇宏观利差与货币强弱监控看板（Streamlit）

运行：
    pip install -r requirements.txt
    streamlit run app.py

页面结构：
    模块一  央行鹰鸽矩阵（2 年期收益率水平 + 近 N 日动量）
    模块二  最佳做多 / 做空配对推荐
    模块三  两国收益率走势对比 + 利差柱状图 + 利差直方图
    模块四  G10 货币强弱（汇率动量）以及它和收益率动量的关系
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

import analytics as an
import charts as ch
import data_sources as ds
from config import (AUTO_REFRESH_OPTIONS, COUNTRIES, DEFAULT_AUTO_REFRESH, DEFAULT_HISTORY_MONTHS,
                    DEFAULT_MOMENTUM_DAYS, DISPLAY_TZ, FX_TTL_SECONDS, MANUAL_CHECK_URLS,
                    STALE_BUSINESS_DAYS, YIELD_TTL_SECONDS)

st.set_page_config(page_title="外汇宏观利差监控", layout="wide")
TZ = ZoneInfo(DISPLAY_TZ)


# ---------------------------------------------------------------------------
# 数据加载（带缓存）。收益率 30 分钟过期、汇率 5 分钟过期，过期后下一次刷新会重新抓取。
# 缓存函数只返回可序列化的简单结构，并附上抓取时间。
# ---------------------------------------------------------------------------
@st.cache_data(ttl=YIELD_TTL_SECONDS, show_spinner="正在从各国央行 / 财政部获取 2 年期收益率……")
def cached_yields(start: date):
    res = ds.load_all_yields(start)
    return {k: (v.series, v.source, v.errors) for k, v in res.items()}, datetime.now(TZ)


@st.cache_data(ttl=FX_TTL_SECONDS, show_spinner="正在从 yfinance 获取汇率……")
def cached_fx(start: date):
    fx, errs = ds.load_fx_usd_values(start)
    return fx, errs, datetime.now(TZ)


def restore(raw: dict) -> dict:
    return {k: ds.YieldResult(k, s, src, errs) for k, (s, src, errs) in raw.items()}


def business_days_since(last: pd.Timestamp) -> int:
    """最新数据日期到今天之间隔了几个工作日（周末不算）。"""
    return int(np.busday_count(last.date(), datetime.now(TZ).date()))


# ---------------------------------------------------------------------------
# 侧边栏（放在自动刷新区域之外，改参数会触发整页重算）
# ---------------------------------------------------------------------------
st.sidebar.header("参数")
days = st.sidebar.slider("动量回看天数（自然日）", 10, 90, DEFAULT_MOMENTUM_DAYS, step=5)
months = st.sidebar.slider("历史走势窗口（月）", 3, 12, DEFAULT_HISTORY_MONTHS)
refresh_min = st.sidebar.selectbox(
    "自动刷新", AUTO_REFRESH_OPTIONS, index=AUTO_REFRESH_OPTIONS.index(DEFAULT_AUTO_REFRESH),
    format_func=lambda m: "关闭" if m == 0 else f"每 {m} 分钟")
if st.sidebar.button("立即刷新数据"):
    st.cache_data.clear()
st.sidebar.caption(f"官方收益率每天更新一次（部分有 1 个交易日延迟），程序每 {YIELD_TTL_SECONDS // 60} 分钟检查一次；"
                   f"汇率每 {FX_TTL_SECONDS // 60} 分钟更新一次。")


def render(days: int, months: int, refresh_min: int) -> None:
    """整个看板的主体。放在 st.fragment 里，按设定的间隔自动重跑。"""
    # 多取 45 天，保证窗口起点前也有数据可以计算动量
    start = date.today() - timedelta(days=int(months * 30.5) + 45)
    raw, y_time = cached_yields(start)
    yields = restore(raw)
    fx, fx_errors, fx_time = cached_fx(start)
    now = datetime.now(TZ)

    st.title("外汇宏观利差与货币强弱监控")
    n_ok = sum(r.ok for r in yields.values())
    auto = "关闭" if refresh_min == 0 else f"每 {refresh_min} 分钟"
    st.caption(f"页面刷新于 {now:%Y-%m-%d %H:%M:%S}（伦敦时间）· 收益率抓取于 {y_time:%H:%M} · "
               f"汇率抓取于 {fx_time:%H:%M} · 自动刷新：{auto} · 成功获取 {n_ok}/{len(yields)} 个经济体 · "
               "数据仅供研究，不构成投资建议")

    # ---- 数据源状态：来源、最新日期、是否过期 ----
    status_rows, stale = [], []
    for code, r in yields.items():
        meta = COUNTRIES[code]
        if r.ok:
            last = r.series.index[-1]
            lag = business_days_since(last)
            state = "✅ 最新" if lag <= STALE_BUSINESS_DAYS else f"⚠️ 可能过期（{lag} 个工作日无新数据）"
            if lag > STALE_BUSINESS_DAYS:
                stale.append(meta["name"])
            status_rows.append(dict(经济体=meta["name"], 货币=meta["ccy"], 数据源=r.source,
                                    最新数据日期=f"{last:%Y-%m-%d}", 状态=state, 备注="；".join(r.errors),
                                    手动查看=""))
        elif code in MANUAL_CHECK_URLS:
            # 官网 + 第三方兜底源都会被云端服务器的出口 IP 拦截，是对方基础设施层面的
            # 限制，不是代码 bug，也没有再绕过去的空间了——与其在页面上堆一串没人看得懂
            # 的报错堆栈，不如给一个手动查看当前数值的链接
            status_rows.append(dict(经济体=meta["name"], 货币=meta["ccy"], 数据源="—", 最新数据日期="—",
                                    状态="⚠️ 云端被拦截", 备注="该国数据源会拦截云端服务器，需手动查看→",
                                    手动查看=MANUAL_CHECK_URLS[code]))
        else:
            status_rows.append(dict(经济体=meta["name"], 货币=meta["ccy"], 数据源="—", 最新数据日期="—",
                                    状态="❌ 获取失败", 备注="；".join(r.errors), 手动查看=""))
    fx_state = ("❌ " + "；".join(f"{c} {e}" for c, e in fx_errors.items())) if fx_errors else \
        (f"✅ 截至 {fx.index[-1]:%Y-%m-%d}" if not fx.empty else "❌ 没有数据")
    status_rows.append(dict(经济体="G10 汇率", 货币="—", 数据源="Yahoo Finance（yfinance）",
                            最新数据日期=f"{fx.index[-1]:%Y-%m-%d}" if not fx.empty else "—", 状态=fx_state,
                            备注="", 手动查看=""))
    with st.expander("数据来源与更新状态" + ("（有数据可能过期）" if stale else ""), expanded=bool(stale)):
        st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True,
                    column_config={"手动查看": st.column_config.LinkColumn("手动查看", display_text="打开官网 ↗")})
    if n_ok == 0:
        st.error("所有收益率数据源都获取失败，请检查网络后点击侧边栏的“立即刷新数据”。")
        return
    _body(yields, fx, days, months)


def _body(yields: dict, fx: pd.DataFrame, days: int, months: int) -> None:

    matrix = an.hawk_dove_matrix(yields, days)
    chg_col = f"{days}日变化(bp)"

    # ---------------------------------------------------------------------------
    # 模块一：央行鹰鸽矩阵
    # ---------------------------------------------------------------------------
    st.header("模块一：央行鹰鸽矩阵")
    c1, c2 = st.columns([3, 2])
    with c1:
        ok_rows = matrix[matrix["状态"] == "正常"]
        cols = [c for c in ["排名", "经济体", "货币", "期限", "最新收益率(%)", f"{days}日前(%)", chg_col,
                            "动量", "鹰鸽标签", "最新日期", "数据源"] if c in ok_rows]
        show = ok_rows[cols]
        st.dataframe(
            show.style.format({"最新收益率(%)": "{:.2f}", f"{days}日前(%)": "{:.2f}", chg_col: "{:+.0f}",
                               "排名": "{:.0f}"}, na_rep="—")
                .background_gradient(subset=[chg_col], cmap="RdBu_r", vmin=-50, vmax=50),
            width="stretch", hide_index=True,
            column_config={"经济体": st.column_config.TextColumn(width="medium")})
        failed = matrix[matrix["状态"] != "正常"]
        if not failed.empty:
            st.warning("未获取到：" + "、".join(failed["经济体"]) + "（原因见侧边栏“数据源状态”）")
        st.caption("按收益率从高到低排序。2 年期收益率主要反映市场对未来两年政策利率的预期，"
                   "水平越高越“鹰”，近期上行越多说明市场在加码加息预期。3Y* = 挪威没有 2 年期基准，"
                   "用 3 年期代替，不参与配对推荐。")
    with c2:
        st.plotly_chart(ch.hawk_dove_heatmap(matrix, days), width="stretch")

    # ---------------------------------------------------------------------------
    # 模块二：最佳做多 / 做空配对
    # ---------------------------------------------------------------------------
    st.header("模块二：最佳做多 / 做空配对推荐")
    pairs = an.best_pairs(matrix, days)
    if pairs:
        L, S = pairs["long_ccy"], pairs["short_ccy"]
        st.success(f"当前宏观利差最分化组合推荐：做多 {L}/{S} 或 做空 {S}/{L}")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{L} 2 年期", f"{pairs['long_yield']:.2f}%")
        m2.metric(f"{S} 2 年期", f"{pairs['short_yield']:.2f}%")
        m3.metric("利差", f"{pairs['spread_bp']:.0f} bp")
        if "mom_long" in pairs:
            st.info(f"动量最分化组合：{pairs['mom_long']} 近 {days} 日收益率变化 {pairs['mom_long_chg']:+.0f} bp，"
                    f"{pairs['mom_short']} 变化 {pairs['mom_short_chg']:+.0f} bp。"
                    f"如果看利差“变化方向”而不是水平，可关注 做多 {pairs['mom_long']}/{pairs['mom_short']}。")
        st.caption("利差只是汇率的驱动之一。高息货币在避险行情中往往急跌（例如套息交易平仓），"
                   "请结合模块四的汇率强弱和你自己的技术面信号再做决定。")
    else:
        st.warning("可用数据不足两个经济体，无法给出配对。")

    # ---------------------------------------------------------------------------
    # 模块三：利差走势对比
    # ---------------------------------------------------------------------------
    st.header("模块三：利差走势对比")
    ok_codes = [c for c, r in yields.items() if r.ok]
    label = {c: f"{COUNTRIES[c]['name']}（{COUNTRIES[c]['ccy']}）" for c in ok_codes}
    d1, d2 = st.columns(2)
    a = d1.selectbox("国家 A", ok_codes, index=ok_codes.index("US") if "US" in ok_codes else 0,
                     format_func=lambda c: label.get(c, c))
    b_default = ok_codes.index("JP") if "JP" in ok_codes else min(1, len(ok_codes) - 1)
    b = d2.selectbox("国家 B", ok_codes, index=b_default, format_func=lambda c: label.get(c, c))
    if a == b:
        st.warning("请选择两个不同的国家。")
    else:
        window_start = pd.Timestamp(date.today() - timedelta(days=int(months * 30.5)))
        sa, sb = yields[a].series, yields[b].series
        sp = an.spread_series(sa[sa.index >= window_start], sb[sb.index >= window_start])
        if sp.empty:
            st.warning("两国数据没有重叠的日期。")
        else:
            pair_label = f"{COUNTRIES[a]['ccy']} − {COUNTRIES[b]['ccy']}"
            st.plotly_chart(ch.yield_lines(sp, label[a], label[b]), width="stretch")
            e1, e2 = st.columns(2)
            e1.plotly_chart(ch.spread_bars(sp, pair_label), width="stretch")
            e2.plotly_chart(ch.spread_histogram(sp, pair_label), width="stretch")
            q = sp["spread_bp"]
            pct = (q <= q.iloc[-1]).mean() * 100
            st.caption(f"{pair_label} 当前利差 {q.iloc[-1]:.0f} bp，窗口内区间 {q.min():.0f} 至 {q.max():.0f} bp，"
                       f"当前值高于窗口内 {pct:.0f}% 的交易日。")

    # ---------------------------------------------------------------------------
    # 模块四：货币强弱
    # ---------------------------------------------------------------------------
    st.header("模块四：G10 货币强弱")
    strength = an.currency_strength(fx, days)
    if strength.empty:
        st.warning("汇率数据获取失败，无法计算货币强弱。")
    else:
        f1, f2 = st.columns([2, 3])
        with f1:
            st.plotly_chart(ch.strength_bars(strength, days), width="stretch")
        with f2:
            idx = an.strength_index(fx, days)
            default_hl = list(strength.index[:2]) + list(strength.index[-2:])
            hl = st.multiselect("高亮货币", list(idx.columns), default=default_hl)
            st.plotly_chart(ch.strength_lines(idx, hl), width="stretch")
        combo = an.macro_vs_fx(matrix, strength, days)
        if not combo.empty:
            st.subheader("汇率有没有跟着收益率走？")
            st.plotly_chart(ch.macro_fx_scatter(combo, days), width="stretch")
            corr = combo.corr().iloc[0, 1]
            st.caption(f"横轴为收益率动量相对 G10 平均的差值，纵轴为汇率相对一篮子的强弱。"
                       f"右上象限 = 利率上行且货币走强；左下 = 利率下行且货币走弱。当前相关系数 {corr:+.2f}。")

    st.divider()
    st.caption("数据来源：美联储 FRED、美国财政部、德国央行 / 欧洲央行、英国央行、日本财务省、澳洲联储、加拿大央行、"
               "瑞士央行、新西兰联储、瑞典央行、挪威央行；汇率来自 Yahoo Finance（yfinance）。")


# 按设定间隔自动重跑看板主体（关闭时只在打开页面或改参数时刷新）
run_every = timedelta(minutes=refresh_min) if refresh_min else None
st.fragment(run_every=run_every)(render)(days, months, refresh_min)
