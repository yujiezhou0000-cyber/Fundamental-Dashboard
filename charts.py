"""
图表层：全部用 Plotly 绘制，交给 st.plotly_chart 渲染（自动适配 Streamlit 的明暗主题）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE, ORANGE, RED, GRAY = "#2a78d6", "#eb6834", "#e34948", "#9a9892"
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#dcdad4"], [1.0, "#e34948"]]   # 鸽（蓝）-> 中性 -> 鹰（红）


def _layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    opts = dict(height=height, margin=dict(l=10, r=10, t=40, b=10),
                hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0))
    opts.update(kw)
    fig.update_layout(**opts)
    return fig


def hawk_dove_heatmap(matrix: pd.DataFrame, days: int) -> go.Figure:
    """鹰鸽热力图：两列（收益率水平、N 日变化），颜色按列内标准化，文字显示原值。"""
    ok = matrix[matrix["状态"] == "正常"]
    chg = f"{days}日变化(bp)"
    cols = [("最新收益率(%)", "{:.2f}%"), (chg, "{:+.0f} bp")]
    z, text = [], []
    for c, fmt in cols:
        v = ok[c].astype(float)
        sd = v.std() or 1.0
        z.append(((v - v.mean()) / sd).clip(-2, 2).tolist())
        text.append([fmt.format(x) if pd.notna(x) else "—" for x in v])
    labels = [f"{r['经济体']} {r['货币']}" for _, r in ok.iterrows()]
    fig = go.Figure(go.Heatmap(
        z=np.array(z).T, x=["2年期收益率", f"近{days}日变化"], y=labels,
        text=np.array(text).T, texttemplate="%{text}", colorscale=DIVERGING, zmid=0,
        showscale=False, hovertemplate="%{y}<br>%{x}：%{text}<extra></extra>", xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, height=60 + 36 * len(ok), hovermode="closest")


def yield_lines(df: pd.DataFrame, name_a: str, name_b: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=df.index, y=df["a"], name=name_a, line=dict(color=BLUE, width=2))
    fig.add_scatter(x=df.index, y=df["b"], name=name_b, line=dict(color=ORANGE, width=2))
    fig.update_yaxes(title_text="2 年期收益率（%）", ticksuffix="%")
    return _layout(fig, title="收益率走势对比")


def spread_bars(df: pd.DataFrame, label: str) -> go.Figure:
    colors = np.where(df["spread_bp"] >= 0, BLUE, ORANGE)
    fig = go.Figure(go.Bar(x=df.index, y=df["spread_bp"], marker_color=colors, name=label,
                           hovertemplate="%{x|%Y-%m-%d}<br>利差 %{y:.0f} bp<extra></extra>"))
    fig.update_yaxes(title_text="利差（bp）")
    return _layout(fig, height=300, title=f"利差逐日变化：{label}", hovermode="closest")


def spread_histogram(df: pd.DataFrame, label: str) -> go.Figure:
    last = df["spread_bp"].iloc[-1]
    fig = go.Figure(go.Histogram(x=df["spread_bp"], nbinsx=30, marker_color=BLUE, opacity=0.85,
                                 hovertemplate="%{x} bp：%{y} 天<extra></extra>"))
    fig.add_vline(x=last, line_color=ORANGE, line_width=2,
                  annotation_text=f"最新 {last:.0f} bp", annotation_position="top")
    fig.update_xaxes(title_text="利差（bp）")
    fig.update_yaxes(title_text="天数")
    return _layout(fig, height=300, title=f"利差分布直方图：{label}", hovermode="closest")


def strength_bars(strength: pd.DataFrame, days: int) -> go.Figure:
    s = strength["相对一篮子强弱(%)"].sort_values()
    colors = np.where(s >= 0, BLUE, RED)
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=colors,
                           text=[f"{v:+.2f}%" for v in s.values], textposition="outside",
                           hovertemplate="%{y}：%{x:+.2f}%<extra></extra>"))
    span = max(abs(s.min()), abs(s.max()), 0.1)
    fig.update_xaxes(title_text=f"近 {days} 日相对 G10 一篮子（%）", ticksuffix="%",
                     range=[-span * 1.45, span * 1.45])   # 两侧留白，避免数字标签被截断
    fig.update_traces(cliponaxis=False)
    return _layout(fig, height=60 + 32 * len(s), hovermode="closest")


def strength_lines(idx: pd.DataFrame, highlight: list[str]) -> go.Figure:
    palette = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    fig = go.Figure()
    for c in idx.columns:
        if c not in highlight:
            fig.add_scatter(x=idx.index, y=idx[c], name=c, line=dict(color=GRAY, width=1),
                            opacity=0.45, showlegend=False, hovertemplate=f"{c} %{{y:.2f}}<extra></extra>")
    for i, c in enumerate(highlight):
        if c in idx.columns:
            fig.add_scatter(x=idx.index, y=idx[c], name=c,
                            line=dict(color=palette[i % len(palette)], width=2.5))
    fig.add_hline(y=100, line_color=GRAY, line_width=1)
    fig.update_yaxes(title_text="相对强弱指数（起点 = 100）")
    return _layout(fig, height=380)


def macro_fx_scatter(df: pd.DataFrame, days: int) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=df["收益率相对动量(bp)"], y=df["相对一篮子强弱(%)"], mode="markers+text",
        text=df.index, textposition="top center", marker=dict(size=12, color=BLUE,
                                                              line=dict(width=2, color="white")),
        hovertemplate="%{text}<br>收益率相对动量 %{x:+.0f} bp<br>汇率强弱 %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    fig.add_vline(x=0, line_color=GRAY, line_width=1)
    fig.update_xaxes(title_text=f"近 {days} 日收益率变化 − G10 平均（bp）")
    fig.update_yaxes(title_text=f"近 {days} 日汇率相对一篮子（%）", ticksuffix="%")
    return _layout(fig, height=420, hovermode="closest")
