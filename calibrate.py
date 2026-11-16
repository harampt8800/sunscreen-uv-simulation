"""
막 불균일 모델 파라미터 보정

아이디어: SPF 절대값이 아니라 "도포량을 줄였을 때 SPF가 몇 배로 떨어지나(비율)"를
문헌 in vivo 결과에 맞춘다.
  - 각 문헌 제품의 표시 SPF(2 mg/cm²)와 같아지도록 처방 농도를 스케일링한 뒤
  - 0.5, 1.0 mg/cm²에서의 SPF / SPF(2) 비율을 문헌값과 비교 -> 오차 최소 파라미터 선택

문헌 목표값 (SPF_2mg, 도포량, SPF비율)
  Beyer et al. 2010 (BJD)          : SPF 11.7 -> 1.0mg 5.7, 0.5mg 3.3
  Ou-Yang et al. 2012 (JAAD 67:1220): SPF 70 -> 0.5mg 19,  SPF 100 -> 0.5mg 27
  (독일 링테스트 값은 1차 출처가 아닌 기업 블로그 인용이라 제외)
"""
import numpy as np
from physics import load_epsilon, solar_spectrum, erythema, absorbance, transmittance, spf

TARGETS = [  # (출처, SPF@2mg, 도포량, 목표 비율)
    ("Beyer2010", 11.7, 1.0, 5.7 / 11.7), ("Beyer2010", 11.7, 0.5, 3.3 / 11.7),
    ("OuYang2012", 70, 0.5, 19 / 70), ("OuYang2012", 100, 0.5, 27 / 100),
]
BASE = {"avobenzone": 3.0, "octocrylene": 8.0}   # 성분 비율만 고정, 전체 농도는 스케일링

eps, S, E = load_epsilon(), solar_spectrum(), erythema()


def spf_at(scale, dose, model, p):
    conc = {k: v * scale for k, v in BASE.items()}
    return spf(transmittance(absorbance(eps, conc, dose), model, **p), S, E)


def scale_for(spf2, model, p):
    """SPF(2 mg/cm²) = spf2 가 되는 농도 배율 (로그 이분법)"""
    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if spf_at(np.exp(mid), 2.0, model, p) < spf2: lo = mid
        else: hi = mid
    return np.exp((lo + hi) / 2)


def ratios(model, p):
    out = []
    for _, spf2, dose, _ in TARGETS:
        s = scale_for(spf2, model, p)
        out.append(spf_at(s, dose, model, p) / spf2)
    return np.array(out)


def loss(model, p):
    target = np.array([t[3] for t in TARGETS])
    return np.sqrt(np.mean((np.log(ratios(model, p)) - np.log(target)) ** 2))  # 로그 오차 RMS


if __name__ == "__main__":
    target = np.array([t[3] for t in TARGETS])
    print("균일 막          loss =", round(loss("uniform", {}), 3))

    ks = np.round(np.arange(0.3, 4.01, 0.1), 2)
    lk = [loss("gamma", {"k": k}) for k in ks]
    k_best = ks[int(np.argmin(lk))]
    print(f"감마 막 최적 k = {k_best}   loss = {min(lk):.3f}")

    best = (9, None)
    for f1 in np.arange(0.1, 0.91, 0.05):
        for h1 in np.arange(0.0, 0.96, 0.05):
            if f1 * h1 >= 1: continue
            l = loss("step", {"f1": f1, "h1": h1})
            if l < best[0]: best = (l, (round(float(f1), 2), round(float(h1), 2)))
    print(f"2단 막 최적 f1, h1 = {best[1]}   loss = {best[0]:.3f}")

    print("\n출처        SPF2  도포량   문헌비율  균일   감마   2단")
    r_u = ratios("uniform", {}); r_g = ratios("gamma", {"k": k_best})
    r_s = ratios("step", {"f1": best[1][0], "h1": best[1][1]})
    for i, (src, spf2, dose, t) in enumerate(TARGETS):
        print(f"{src:<10} {spf2:>5} {dose:>5}  {t:>8.2f} {r_u[i]:>6.2f} {r_g[i]:>6.2f} {r_s[i]:>6.2f}")
