"""
5단계: 보고서용 그래프 6종  (먼저 ai_surrogate.py 를 실행해야 3, 4번 그래프 데이터가 생김)

fig0_spectra.png       : 디지타이징한 흡수 스펙트럼
fig0_calibration.png   : 막 모델 보정 결과 vs 문헌

fig1_transmittance.png : 도포량별 투과율 스펙트럼
fig2_spf_heatmap.png   : 도포량 × 농도 SPF 히트맵
fig3_ai_parity.png     : AI 예측 vs 물리 모델 (테스트 / 외삽)
fig4_model_compare.png : 모델별 오차 비교
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
try:                                   # 코랩/주피터에서 %run 하면 그래프가 화면에 바로 나옴
    get_ipython  # noqa
    IN_NOTEBOOK = True
except NameError:
    IN_NOTEBOOK = False
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import physics as ph
from ai_surrogate import physics_batch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)

# ---------- 한글 폰트 (코랩: !apt-get -qq install fonts-nanum) ----------
for f in glob.glob("/usr/share/fonts/**/*", recursive=True):
    if any(n in f for n in ("NanumGothic.ttf", "NotoSansCJK-Regular")):
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
plt.rcParams["axes.unicode_minus"] = False

# ---------- 스타일 (단일 축, 얇은 선, 은은한 그리드) ----------
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "figure.dpi": 150,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK2,
})
BASE = {"avobenzone": 3.0, "octocrylene": 8.0}
K = ph.K_CAL


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name)); print("저장:", name)
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


def fig_spectra():
    d = pd.read_csv(os.path.join(HERE, "data", "epsilon_digitized.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(290, 320, color="#f1f0ec", lw=0, zorder=0)
    ax.text(305, 36500, "UVB", ha="center", color=MUTED, fontsize=9); ax.text(360, 36500, "UVA", ha="center", color=MUTED, fontsize=9)
    ax.plot(d.wavelength_nm, d.avobenzone, color=ORANGE, lw=2, label="아보벤존 (UVA 담당)")
    ax.plot(d.wavelength_nm, d.octocrylene, color=BLUE, lw=2, label="옥토크릴렌 (UVB 담당)")
    ax.set_xlim(290, 400); ax.set_ylim(0, 39000)
    ax.set_xlabel("파장 (nm)"); ax.set_ylabel("몰흡광계수 ε  [L/(mol·cm)]")
    ax.set_title("디지타이징한 흡수 스펙트럼 (Herzog 2002, Fig.2)")
    ax.legend(fontsize=8, loc="center left")
    save(fig, "fig0_spectra.png")


def fig_calibration():
    from calibrate import scale_for, spf_at, TARGETS
    doses = np.linspace(0.25, 2, 30)
    cm = {11.7: "#1baf7a", 70: ORANGE, 100: BLUE}
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for spf2, c in cm.items():
        s = scale_for(spf2, "gamma", {"k": K})
        ax.plot(doses, [spf_at(s, d, "gamma", {"k": K}) / spf2 for d in doses], color=c, lw=2, label=f"모델 (SPF {spf2:g})")
    su = scale_for(70, "uniform", {})
    ax.plot(doses, [spf_at(su, d, "uniform", {}) / 70 for d in doses], color=INK2, lw=1.4, ls="--", label="균일 막 가정 (SPF 70)")
    mk = {"Beyer2010": "o", "OuYang2012": "^"}; seen = set()
    for src, spf2, d, t in TARGETS:
        ax.scatter(d, t, marker=mk[src], s=70, color=cm[spf2], edgecolor=INK, zorder=5,
                   label=("문헌: " + src) if src not in seen else None); seen.add(src)
    ax.set_xlim(0.2, 2.05); ax.set_ylim(0, 1.05)
    ax.set_xlabel("도포량 (mg/cm²)"); ax.set_ylabel("SPF / SPF(2 mg/cm²)")
    ax.set_title(f"감마 막 모델(k={K}) vs 문헌 in vivo 측정값")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    save(fig, "fig0_calibration.png")


def fig1():
    eps = ph.load_epsilon()
    doses = [0.5, 1.0, 1.5, 2.0]
    cols = [RAMP[2], RAMP[4], RAMP[6], RAMP[8]]            # 단일 색상, 진할수록 많이 바름
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(290, 320, color="#f1f0ec", lw=0, zorder=0)
    ax.text(305, 80, "UVB", ha="center", color=MUTED, fontsize=9)
    ax.text(360, 80, "UVA", ha="center", color=MUTED, fontsize=9)
    for d, c in zip(doses, cols):
        T = ph.transmittance(ph.absorbance(eps, BASE, d), "gamma", k=K) * 100
        ax.plot(ph.WL, T, color=c, lw=2, label=f"{d} mg/cm²")
        i = np.searchsorted(ph.WL, 340)
        ax.annotate(f"{d} mg/cm²", (340, T[i]), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    color=INK2, fontsize=8)
    ax.set_xlim(290, 400); ax.set_yscale("log"); ax.set_ylim(2, 100)
    ax.set_yticks([2, 5, 10, 20, 50, 100]); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("파장 (nm)"); ax.set_ylabel("투과율 (%, 로그 눈금)")
    ax.set_title("도포량이 줄면 자외선 투과가 급격히 늘어난다")
    ax.text(0.5, 0.80, f"아보벤존 {BASE['avobenzone']:g}% + 옥토크릴렌 {BASE['octocrylene']:g}%  ·  감마 막 모델 (k={K})",
            transform=ax.transAxes, color=INK2, fontsize=8.5, ha="center")
    ax.legend(title="도포량", title_fontsize=8, fontsize=8, loc="center left", bbox_to_anchor=(0.01, 0.72))
    save(fig, "fig1_transmittance.png")


def fig2():
    doses = np.linspace(0.25, 3.0, 120)
    scale = np.linspace(0, 1.25, 100)                      # 1.25배 = 옥토크릴렌 10% (사용한도)
    D, Sc = np.meshgrid(doses, scale)
    X = np.c_[D.ravel(), BASE["avobenzone"] * Sc.ravel(), BASE["octocrylene"] * Sc.ravel(), np.full(D.size, K)]
    spf = physics_batch(X)[0].reshape(D.shape)
    total = (BASE["avobenzone"] + BASE["octocrylene"]) * scale
    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    im = ax.pcolormesh(doses, total, spf, cmap=cmap, vmin=1, vmax=45, shading="auto")
    cs = ax.contour(doses, total, spf, levels=[5, 10, 15, 20], colors="white", linewidths=1.2)
    ax.clabel(cs, fmt=lambda v: f"SPF {v:.0f}", fontsize=8, colors="white")
    ax.axvline(2.0, color="white", lw=1, ls="--"); ax.text(2.03, 0.5, "권장량 2 mg/cm²", color="white", fontsize=8, rotation=90)
    ax.axvspan(0.5, 1.0, color="white", alpha=0.12, lw=0)
    ax.text(0.75, total.max() * 0.95, "실제 사용량\n0.5~1", color="white", fontsize=8, ha="center", va="top")
    ax.grid(False)
    ax.set_xlabel("도포량 (mg/cm²)"); ax.set_ylabel("UV 필터 총 농도 (%)  [아보벤존:옥토크릴렌 = 3:8]")
    ax.set_title("SPF는 농도보다 도포량에 더 민감하다")
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("SPF", color=INK2); cb.outline.set_visible(False)
    save(fig, "fig2_spf_heatmap.png")


def fig3():
    p = pd.read_csv(os.path.join(HERE, "data", "parity_best.csv"))
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for g, c, lab in [("테스트", BLUE, "테스트 (도포량 ≤ 2)"), ("외삽", ORANGE, "외삽 (도포량 2~3, 학습에 없음)")]:
        d = p[p["구간"] == g].sample(min(1500, (p["구간"] == g).sum()), random_state=0)
        ax.scatter(d["실제"], d["예측"], s=9, color=c, alpha=0.5, lw=0, label=lab)
    lim = [1, p[["실제", "예측"]].max().max() * 1.1]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--"); ax.text(lim[1] * 0.55, lim[1] * 0.75, "완벽한 예측", color=INK2, fontsize=8, rotation=45)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("물리 모델 SPF (정답)"); ax.set_ylabel("AI 예측 SPF")
    ax.set_title("AI(신경망) 예측 vs 물리 모델")
    ax.legend(fontsize=8, loc="upper left", markerscale=2)
    save(fig, "fig3_ai_parity.png")


def fig4():
    r = pd.read_csv(os.path.join(HERE, "data", "model_comparison.csv"))
    x = np.arange(len(r)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for off, col, c, lab in [(-w / 2 - 0.01, "테스트 오차%", BLUE, "테스트 (학습 범위 안)"),
                             (w / 2 + 0.01, "외삽 오차%", ORANGE, "외삽 (학습 범위 밖)")]:
        b = ax.bar(x + off, r[col], w, color=c, label=lab, zorder=3)
        ax.bar_label(b, fmt="%.1f%%", padding=3, fontsize=8, color=INK2)
    ax.set_xticks(x, r["모델"]); ax.tick_params(axis="x", colors=INK2)
    ax.set_ylabel("평균 SPF 오차 (%)"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, r["외삽 오차%"].max() * 1.2)
    ax.set_title("학습 범위 밖에서는 신경망만 버틴다")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "fig4_model_compare.png")


if __name__ == "__main__":
    fig_spectra(); fig_calibration(); fig1(); fig2(); fig3(); fig4()
