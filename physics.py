"""
자외선 차단제 투과율 / SPF 계산 (Beer-Lambert + 막 불균일 모델)

A(λ) = Σ ε_i(λ) · c_i · l
  ε_i : 몰흡광계수 [L/(mol·cm)]  (Herzog 2002 Fig.2)
  c_i : 몰농도 [mol/L]  = (w% × 10 × 밀도[g/mL]) / 분자량
  l   : 막 두께 [cm]    = (도포량[mg/cm²] / 1000) / 밀도[g/cm³]
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WL = np.arange(290, 401)  # 290~400 nm, 1 nm 간격

# ---- 성분 정보 (출처를 보고서 표에 같이 기록) ----
MW = {"avobenzone": 310.39, "octocrylene": 361.48}   # g/mol (PubChem)
MAX_PCT = {"avobenzone": 5.0, "octocrylene": 10.0}    # 식약처 사용한도 (%)
DENSITY = 1.0                                          # g/cm³ 가정
K_CAL = 1.1   # 감마 막 불균일도: calibrate.py 로 문헌 in vivo 데이터(Beyer 2010, Ou-Yang 2012)에 보정한 값


# ---------- 데이터 로딩 ----------
def load_epsilon(path=os.path.join(HERE, "data", "epsilon_digitized.csv")):
    """WebPlotDigitizer 추출값 (prepare_data.py 로 1 nm 보간 완료)"""
    df = pd.read_csv(path, comment="#")
    return {name: np.interp(WL, df["wavelength_nm"], df[name])
            for name in df.columns if name != "wavelength_nm"}


def solar_spectrum():
    """ASTM G173-03 global tilted (pvlib 내장), 290~400 nm"""
    from pvlib.spectrum import get_reference_spectra
    return get_reference_spectra(wavelengths=WL)["global"].to_numpy()


def erythema(wl=WL, version="CIE98"):
    """CIE 홍반 작용 스펙트럼"""
    k = 140 if version == "CIE98" else 139
    return np.where(wl <= 298, 1.0,
           np.where(wl <= 328, 10 ** (0.094 * (298 - wl)),
                               10 ** (0.015 * (k - wl))))


# ---------- 단위 변환 ----------
def pct_to_molar(pct, mw, density=DENSITY):
    return pct * 10 * density / mw


def dose_to_cm(dose_mg_cm2, density=DENSITY):
    return dose_mg_cm2 / 1000 / density


# ---------- 흡광도 / 투과율 ----------
def absorbance(eps, conc_pct, dose_mg_cm2):
    """conc_pct = {"avobenzone": 3, "octocrylene": 8}"""
    l = dose_to_cm(dose_mg_cm2)
    A = np.zeros_like(WL, dtype=float)
    for name, pct in conc_pct.items():
        A += eps[name] * pct_to_molar(pct, MW[name]) * l
    return A


def transmittance(A, model="uniform", **p):
    """
    uniform : 균일한 막          T = 10^-A
    step    : O'Neill 2단 막     T = f1·10^(-h1·A) + f2·10^(-h2·A),  f1·h1 + f2·h2 = 1
    gamma   : 감마 분포 막 (Ferrero 아이디어, 평균 두께 1로 정규화)
              h ~ Gamma(k, 1/k)  ->  T = E[10^(-h·A)] = (1 + A·ln10/k)^(-k)
              k가 작을수록 불균일 (k→∞ 이면 uniform 과 같아짐)
    """
    if model == "uniform":
        return 10 ** (-A)
    if model == "step":
        f1, h1 = p.get("f1", 0.5), p.get("h1", 0.2)
        f2 = 1 - f1
        h2 = (1 - f1 * h1) / f2
        return f1 * 10 ** (-h1 * A) + f2 * 10 ** (-h2 * A)
    if model == "gamma":
        k = p.get("k", K_CAL)
        return (1 + A * np.log(10) / k) ** (-k)
    raise ValueError(model)


# ---------- 지표 ----------
def spf(T, S, E):
    return np.trapezoid(E * S, WL) / np.trapezoid(E * S * T, WL)


def uva_uvb_ratio(A):
    """UVA(320-400) 평균 흡광도 / UVB(290-320) 평균 흡광도 (Boots star 방식)"""
    uvb, uva = WL <= 320, WL >= 320
    return np.trapezoid(A[uva], WL[uva]) / 80 / (np.trapezoid(A[uvb], WL[uvb]) / 30)


def critical_wavelength(A):
    """흡광도 적분의 90%에 도달하는 파장"""
    cum = np.concatenate([[0], np.cumsum((A[1:] + A[:-1]) / 2)])
    return WL[np.searchsorted(cum, 0.9 * cum[-1])]


# ---------- 데모 ----------
if __name__ == "__main__":
    eps = load_epsilon()
    S, E = solar_spectrum(), erythema()

    # 1) 판독값 검증: 아보벤존 피크 vs 문헌값(1%/1cm 1100~1160 -> ε ≈ 34,100~36,000)
    lit = np.array([1100, 1160]) * MW["avobenzone"] / 10
    print(f"[검증] 아보벤존 피크 ε = {eps['avobenzone'].max():.0f} "
          f"@ {WL[eps['avobenzone'].argmax()]} nm  (문헌 환산 {lit[0]:.0f}~{lit[1]:.0f})")

    # 2) 도포량별 SPF : 막 모델 비교
    conc = {"avobenzone": 3.0, "octocrylene": 8.0}
    print(f"\n처방 {conc}")
    print(f"{'도포량':>8} | {'uniform':>12} {'step':>8} {f'gamma(k={K_CAL})':>11}")
    for dose in [0.5, 1.0, 1.5, 2.0]:
        A = absorbance(eps, conc, dose)
        row = [spf(transmittance(A, m), S, E) for m in ("uniform", "step", "gamma")]
        print(f"{dose:>5} mg | {row[0]:>12.3g} {row[1]:>8.1f} {row[2]:>11.1f}")

    A2 = absorbance(eps, conc, 2.0)
    print(f"\nUVA/UVB 비 = {uva_uvb_ratio(A2):.2f},  임계파장 = {critical_wavelength(A2)} nm")
