"""
4단계: AI 대리 모델 (surrogate) + 역설계

① 물리 모델(physics.py)로 학습 데이터 생성  (라틴 하이퍼큐브 샘플링 + 측정 노이즈)
② 여러 모델 학습 -> 테스트 오차로 비교 -> 최적 모델 선택
③ 외삽 테스트 (학습 범위 밖 도포량)  +  변수 중요도
④ 역설계: "목표 SPF를 최소 농도로" -> 최적 조합 찾고 물리 모델로 재검증

입력 변수 : 도포량[mg/cm²], 아보벤존[%], 옥토크릴렌[%], 막 불균일도 k
출력 변수 : log(SPF), UVA/UVB 비
"""
import os, time
import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.optimize import differential_evolution
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
import physics as ph

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)
FEATURES = ["dose", "avobenzone", "octocrylene", "k"]
BOUNDS = {"dose": (0.25, 3.0), "avobenzone": (0.0, 5.0), "octocrylene": (0.0, 10.0), "k": (0.6, 3.0)}

# ---------- 물리 모델 (여러 샘플을 한 번에 계산하도록 벡터화) ----------
EPS = ph.load_epsilon()
S, E = ph.solar_spectrum(), ph.erythema()
W = E * S
UVB, UVA = ph.WL <= 320, ph.WL >= 320


def physics_batch(X):
    """X: (N,4) [dose, avo%, oc%, k] -> SPF, UVA/UVB비"""
    dose, avo, oc, k = X.T
    l = ph.dose_to_cm(dose)[:, None]
    A = (EPS["avobenzone"][None] * ph.pct_to_molar(avo, ph.MW["avobenzone"])[:, None]
         + EPS["octocrylene"][None] * ph.pct_to_molar(oc, ph.MW["octocrylene"])[:, None]) * l
    T = (1 + A * np.log(10) / k[:, None]) ** (-k[:, None])          # 감마 막 모델
    spf = np.trapezoid(W, ph.WL) / np.trapezoid(W * T, ph.WL, axis=1)
    a_uva = np.trapezoid(A[:, UVA], ph.WL[UVA], axis=1) / 80
    a_uvb = np.trapezoid(A[:, UVB], ph.WL[UVB], axis=1) / 30
    ratio = np.divide(a_uva, a_uvb, out=np.zeros_like(a_uva), where=a_uvb > 0)
    return spf, ratio


# ---------- ① 데이터 생성 ----------
def make_dataset(n=20000, noise=0.03):
    lo = np.array([BOUNDS[f][0] for f in FEATURES]); hi = np.array([BOUNDS[f][1] for f in FEATURES])
    X = qmc.scale(qmc.LatinHypercube(d=4, seed=0).random(n), lo, hi)
    spf, ratio = physics_batch(X)
    spf_noisy = spf * np.exp(RNG.normal(0, noise, n))              # 측정 오차 ±3% 수준
    df = pd.DataFrame(X, columns=FEATURES)
    df["spf_true"], df["spf"], df["uva_uvb"] = spf, spf_noisy, ratio
    df["log_spf"] = np.log(df["spf"])
    return df


def main():
    t0 = time.time()
    df = make_dataset()
    df.to_csv(os.path.join(HERE, "data", "train_dataset.csv"), index=False)
    print(f"① 데이터 {len(df)}개 생성 ({time.time() - t0:.1f}s) | SPF 범위 {df.spf.min():.1f}~{df.spf.max():.0f}")

    # ---------- ② 모델 비교 ----------
    inside = df[df.dose <= 2.0]                      # 외삽 테스트용으로 2 mg/cm² 초과는 따로 떼어둠
    outside = df[df.dose > 2.0]
    tr, te = train_test_split(inside, test_size=0.2, random_state=0)
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "GradientBoosting": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=0),
        "MLP(신경망)": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=800,
                                                               early_stopping=True, random_state=0)),
    }
    rows, preds = [], {}
    for name, m in models.items():
        t = time.time(); m.fit(tr[FEATURES], tr.log_spf)
        p_in, p_out = np.exp(m.predict(te[FEATURES])), np.exp(m.predict(outside[FEATURES]))
        preds[name] = (p_in, p_out)
        rows.append([name, r2_score(np.log(te.spf_true), np.log(p_in)),
                     np.mean(np.abs(p_in / te.spf_true - 1)) * 100,
                     np.mean(np.abs(p_out / outside.spf_true - 1)) * 100, time.time() - t])
    res = pd.DataFrame(rows, columns=["모델", "R²(log SPF)", "테스트 오차%", "외삽 오차%", "학습시간s"]).round(3)
    print("\n② 모델 비교 (학습: 도포량 ≤ 2.0 / 외삽: 2.0~3.0)\n", res.to_string(index=False))
    best_name = res.sort_values("테스트 오차%").iloc[0]["모델"]
    best = models[best_name]
    res.to_csv(os.path.join(HERE, "data", "model_comparison.csv"), index=False)
    pd.concat([pd.DataFrame({"구간": "테스트", "실제": te.spf_true.values, "예측": preds[best_name][0]}),
               pd.DataFrame({"구간": "외삽", "실제": outside.spf_true.values, "예측": preds[best_name][1]})]
              ).to_csv(os.path.join(HERE, "data", "parity_best.csv"), index=False)
    print(f"→ 선택: {best_name}")

    # ---------- ③ 변수 중요도 ----------
    imp = permutation_importance(best, te[FEATURES], te.log_spf, n_repeats=5, random_state=0)
    print("\n③ 변수 중요도 (permutation importance)")
    for f, v in sorted(zip(FEATURES, imp.importances_mean), key=lambda x: -x[1]):
        print(f"   {f:<12} {v:.3f}")

    # 최종 모델은 전체 범위로 다시 학습
    best.fit(df[FEATURES], df.log_spf)

    # ---------- ④ 역설계 ----------
    print(f"\n④ 역설계: 목표 SPF를 '총 농도 최소'로 달성 (k={ph.K_CAL}, 국내 사용한도 내)")
    def predict(dose, avo, oc):
        return np.exp(best.predict(pd.DataFrame([[dose, avo, oc, ph.K_CAL]], columns=FEATURES))[0])

    for dose in [2.0, 1.0]:
        spf_max = physics_batch(np.array([[dose, 5, 10, ph.K_CAL]]))[0][0]
        for target in [15, 30, 50]:
            if predict(dose, 5, 10) < target:
                print(f"   {dose} mg/cm², 목표 SPF {target:>2}: ❌ 불가능 — 두 성분을 한도(5%+10%)까지 넣어도 최대 SPF {spf_max:.1f}")
                continue
            def obj(c):
                spf_hat = np.exp(best.predict(pd.DataFrame([[dose, c[0], c[1], ph.K_CAL]], columns=FEATURES))[0])
                return c[0] + c[1] + 50 * max(0, np.log(target) - np.log(spf_hat))   # 미달 시 벌점
            r = differential_evolution(obj, [BOUNDS["avobenzone"], BOUNDS["octocrylene"]], seed=0, tol=1e-6, maxiter=200)
            avo, oc = r.x
            spf_ai = predict(dose, avo, oc)
            spf_phys, ratio = physics_batch(np.array([[dose, avo, oc, ph.K_CAL]]))
            err = (spf_ai / spf_phys[0] - 1) * 100
            print(f"   {dose} mg/cm², 목표 SPF {target:>2}: 아보벤존 {avo:4.2f}% + 옥토크릴렌 {oc:5.2f}%"
                  f" | AI 예측 {spf_ai:5.1f} / 물리 재검증 {spf_phys[0]:5.1f} (AI 오차 {err:+.1f}%)"
                  f" | UVA/UVB {ratio[0]:.2f} ✅")
    print(f"\n총 실행 시간 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
