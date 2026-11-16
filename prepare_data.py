"""
WebPlotDigitizer 원본(raw) -> 분석용 ε 테이블(1 nm 간격) 생성 + 디지타이징 검증

입력 : data/raw/avobenzone_wpd.csv, data/raw/octocrylene_wpd.csv  (헤더 없음: 파장, ε)
출력 : data/epsilon_digitized.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
WL = np.arange(290, 401)
NAMES = ["avobenzone", "octocrylene"]
MW_AVO = 310.39


def load_raw(name):
    df = pd.read_csv(os.path.join(RAW, f"{name}_wpd.csv"), header=None, names=["wl", "eps"]).dropna()
    return df.sort_values("wl")


def main():
    out = {"wavelength_nm": WL}
    ref = pd.read_csv(os.path.join(HERE, "data", "eps_herzog2002_fig2.csv"), comment="#")
    grid5 = np.arange(290, 401, 5)

    for name in NAMES:
        raw = load_raw(name)
        neg = (raw.eps < 0).sum()
        # 0 근처에서 클릭 위치 때문에 생긴 음수 -> 물리적으로 ε >= 0 이므로 0으로 처리
        eps = np.clip(np.interp(WL, raw.wl, raw.eps), 0, None)
        out[name] = eps.round(1)

        d = np.interp(grid5, raw.wl, raw.eps) - ref[name].values
        print(f"[{name}] 점 {len(raw)}개 | x 오차 평균 {np.abs(raw.wl.values - grid5).mean():.2f} nm"
              f" | 육안 판독과 차이: 평균 {np.abs(d).mean():.0f}, 피크 대비 {np.abs(d).mean() / eps.max() * 100:.2f}%"
              f" | 음수 {neg}개 -> 0 처리 | 피크 {eps.max():.0f} @ {WL[eps.argmax()]} nm")

    lit = np.array([1100, 1160]) * MW_AVO / 10
    print(f"[문헌 검증] 아보벤존 피크 {out['avobenzone'].max():.0f} vs 문헌 환산 {lit[0]:.0f}~{lit[1]:.0f}")

    path = os.path.join(HERE, "data", "epsilon_digitized.csv")
    pd.DataFrame(out).to_csv(path, index=False)
    print("저장:", path)


if __name__ == "__main__":
    main()
