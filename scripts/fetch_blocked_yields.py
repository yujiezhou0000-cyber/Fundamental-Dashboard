#!/usr/bin/env python3
"""
GitHub Actions 定时任务：从一个和 Streamlit Cloud 不同的出口 IP 抓取
瑞士央行 / 新西兰联储 2 年期收益率，写入 cache/blocked_yields.json，
供部署在 Streamlit Cloud 上的 App 在官网直连被拦截时读取兜底。

背景：实测瑞士央行（data.snb.ch）和新西兰联储（rbnz.govt.nz）会拦截
Streamlit Community Cloud 的出口 IP（分别表现为"返回内容正常但没有数据行"
和"403 Forbidden"），但完全相同的抓取代码在别的网络环境下工作正常。
GitHub Actions runner 的出口 IP 不在这两家网站的拦截名单里，所以用它
做一次"中转"：定时抓取 -> 提交到仓库 -> App 读取仓库里的静态 JSON 文件
（走 raw.githubusercontent.com，不再直连官网）。

用法（本地调试）：
    pip install pandas numpy requests openpyxl
    python scripts/fetch_blocked_yields.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import data_sources as ds  # noqa: E402

CACHE_PATH = REPO_ROOT / "cache" / "blocked_yields.json"
START = date.today() - timedelta(days=400)  # 留够长的历史给图表用

# code -> (显示名称, 抓取函数)。都是 data_sources.py 里现成的官方接口抓取函数，
# 中转脚本不重复实现解析逻辑，只是换一个网络环境去调用同一份代码。
TARGETS = {
    "CH": ("瑞士央行数据门户（GitHub Actions 中转缓存）", ds.fetch_ch_snb),
    "NZ": ("新西兰联储 B2（GitHub Actions 中转缓存）", ds.fetch_nz_rbnz),
}


def load_existing() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def main() -> int:
    data = load_existing()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    any_ok = False
    for code, (label, fn) in TARGETS.items():
        try:
            s = fn(START).dropna()
            if s.empty:
                raise ValueError("抓到的序列是空的")
            data[code] = {
                "dates": [d.strftime("%Y-%m-%d") for d in s.index],
                "values": [round(float(v), 4) for v in s.values],
                "source_label": label,
                "updated_at": now,
            }
            any_ok = True
            print(f"OK {code}: {len(s)} rows, last={s.index[-1].date()} {s.iloc[-1]:.3f}")
        except Exception as e:
            # 这次失败就保留上一次成功写入的缓存内容，不要用空数据覆盖掉
            print(f"FAIL {code}: {type(e).__name__}: {e}")

    data["_generated_at"] = now
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not any_ok and not any(code in data for code in TARGETS):
        # 缓存文件里从来没成功过任何一个国家，说明中转本身也被挡住了，用非零退出码
        # 让 workflow 显示成失败，方便及时发现（但仍然会把这次的失败原因写进日志）
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
