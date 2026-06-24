import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Predictor Mundial 2026", page_icon="⚽", layout="wide")
st.title("⚽ Predictor Mundial 2026")
st.markdown("### Modelo Dixon-Coles + Forma Reciente")

@st.cache_data
def cargar_y_entrenar():
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score","away_score"])
    DESDE = pd.Timestamp("2018-01-01")
    HALF_LIFE = 547
    MATCH_DATE = pd.Timestamp("2026-06-25")
    df_mod = df[df.date >= DESDE].copy().reset_index(drop=True)
    df_mod["w"] = 0.5 ** ((MATCH_DATE - df_mod["date"]).dt.days / HALF_LIFE)
    teams = sorted(set(df_mod["home_team"]) | set(df_mod["away_team"]))
    idx = {t:i for i,t in enumerate(teams)}
    n = len(teams)

    def dc_ll_fast(params):
        att = params[:n]; dfn = params[n:2*n]
        home = params[2*n]; rho = params[2*n+1]
        hi = df_mod["home_team"].map(idx).values
        ai = df_mod["away_team"].map(idx).values
        lam = np.exp(att[hi] - dfn[ai] + home)
        mu  = np.exp(att[ai] - dfn[hi])
        x = df_mod["home_score"].values.astype(int)
        y = df_mod["away_score"].values.astype(int)
        w = df_mod["w"].values
        tau = np.ones(len(df_mod))
        tau[(x==0)&(y==0)] *= 1 - lam[(x==0)&(y==0)]*mu[(x==0)&(y==0)]*rho
        tau[(x==1)&(y==0)] *= 1 + mu[(x==1)&(y==0)]*rho
        tau[(x==0)&(y==1)] *= 1 + lam[(x==0)&(y==1)]*rho
        tau[(x==1)&(y==1)] *= 1 - rho
        ll = w*(np.log(np.maximum(tau,1e-10))+poisson.logpmf(x,lam)+poisson.logpmf(y,mu))
        return -ll.sum()

    x0 = np.zeros(2*n+2); x0[2*n] = 0.3
    bounds = [(None,None)]*2*n + [(0,2),(-0.5,0)]
    res = minimize(dc_ll_fast, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter":300})
    ATT = res.x[:n]; DFN = res.x[n:2*n]
    HOME = res.x[2*n]; RHO = res.x[2*n+1]
    return df, teams, idx, n, ATT, DFN, HOME, RHO

with st.spinner("Cargando datos y entrenando modelo (30 seg)..."):
    df, teams, idx, n, ATT, DFN, HOME, RHO = cargar_y_entrenar()
st.success("✅ Modelo listo")

def tau_dc(x, y, lam, mu, rho):
    if x==0 and y==0: return 1 - lam*mu*rho
    elif x==1 and y==0: return 1 + mu*rho
    elif x==0 and y==1: return 1 + lam*rho
    elif x==1 and y==1: return 1 - rho
    else: return 1

def forma_reciente(equipo, n=6):
    p = df[(df.home_team==equipo)|(df.away_team==equipo)]
    p = p[p.date >= pd.Timestamp("2025-12-01")].tail(n)
    res = []
    for _, r in p.iterrows():
        if r.home_team==equipo:
            res.append("W" if r.home_score>r.away_score else "D" if r.home_score==r.away_score else "L")
        else:
            res.append("W" if r.away_score>r.home_score else "D" if r.away_score==r.home_score else "L")
    pts = sum(3 if r=="W" else 1 if r=="D" else 0 for r in res)
    factor = 0.85 + 0.3*(pts/max(len(res)*3,1))
    return res, factor

# ── SIDEBAR ──
st.sidebar.header("⚽ Selecciona Partido")
team_list = sorted(teams)
home = st.sidebar.selectbox("Equipo LOCAL", team_list, index=team_list.index("Brazil") if "Brazil" in team_list else 0)
away = st.sidebar.selectbox("Equipo VISITANTE", team_list, index=team_list.index("Argentina") if "Argentina" in team_list else 1)

st.sidebar.markdown("---")
st.sidebar.header("💰 Cuotas (opcional)")
usar_cuotas = st.sidebar.checkbox("Analizar valor de apuesta")
cuota_h = cuota_e = cuota_a = None
if usar_cuotas:
    cuota_h = st.sidebar.number_input("Cuota Gana Local", min_value=1.01, value=2.00, step=0.05)
    cuota_e = st.sidebar.number_input("Cuota Empate", min_value=1.01, value=3.50, step=0.05)
    cuota_a = st.sidebar.number_input("Cuota Gana Visitante", min_value=1.01, value=3.00, step=0.05)

if st.sidebar.button("🔍 Analizar", use_container_width=True):
    if home == away:
        st.error("Selecciona equipos diferentes")
    else:
        h, a = idx[home], idx[away]
        forma_h, factor_h = forma_reciente(home)
        forma_a, factor_a = forma_reciente(away)
        lam = np.exp(ATT[h] - DFN[a]) * factor_h
        mu  = np.exp(ATT[a] - DFN[h]) * factor_a
        MAXG = 7
        matriz = np.zeros((MAXG,MAXG))
        for i in range(MAXG):
            for j in range(MAXG):
                matriz[i,j] = tau_dc(i,j,lam,mu,RHO)*poisson.pmf(i,lam)*poisson.pmf(j,mu)
        prob_h = np.sum(np.tril(matriz,-1))
        prob_d = np.sum(np.diag(matriz))
        prob_a = np.sum(np.triu(matriz,1))
        marcadores = sorted([(i,j,matriz[i,j]) for i in range(MAXG) for j in range(MAXG)],key=lambda x:-x[2])
        mejor = marcadores[0]
        pred = f"Gana {home}" if prob_h>prob_a and prob_h>prob_d else ("Empate" if prob_d>prob_h and prob_d>prob_a else f"Gana {away}")

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Gana {home}", f"{prob_h*100:.1f}%")
        col2.metric("Empate", f"{prob_d*100:.1f}%")
        col3.metric(f"Gana {away}", f"{prob_a*100:.1f}%")
        col4.metric("Marcador probable", f"{mejor[0]}-{mejor[1]}")

        st.markdown(f"### 🏆 Predicción: **{pred}**")

        # Gráficos
        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots(figsize=(5,5))
            fig1.patch.set_facecolor("#0d1117")
            ax1.set_facecolor("#0d1117")
            ax1.pie([prob_h,prob_d,prob_a],
                    labels=[f"Gana {home}","Empate",f"Gana {away}"],
                    autopct="%1.1f%%", colors=["#238636","#e3b341","#da3633"],
                    textprops={"color":"white"})
            ax1.set_title("Probabilidades", color="white")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(5,5))
            fig2.patch.set_facecolor("#0d1117")
            ax2.set_facecolor("#161b22")
            top = marcadores[:7]
            ax2.barh([f"{i}-{j}" for i,j,p in top][::-1],
                     [p*100 for i,j,p in top][::-1],
                     color=["#238636" if i>j else "#da3633" if i<j else "#e3b341" for i,j,p in top][::-1])
            ax2.set_title("Top Marcadores", color="white")
            ax2.tick_params(colors="white")
            ax2.spines[:].set_color("#30363d")
            st.pyplot(fig2)

        # Mapa de calor
        st.markdown("### 🗺️ Mapa de Calor de Marcadores")
        fig3, ax3 = plt.subplots(figsize=(8,5))
        fig3.patch.set_facecolor("#0d1117")
        ax3.set_facecolor("#161b22")
        im = ax3.imshow(matriz[:6,:6]*100, cmap="YlOrRd", aspect="auto")
        ax3.set_xticks(range(6)); ax3.set_yticks(range(6))
        ax3.set_xticklabels(range(6), color="white")
        ax3.set_yticklabels(range(6), color="white")
        ax3.set_xlabel(f"Goles {away}", color="white")
        ax3.set_ylabel(f"Goles {home}", color="white")
        for i in range(6):
            for j in range(6):
                ax3.text(j,i,f"{matriz[i,j]*100:.1f}%",ha="center",va="center",
                         color="black",fontsize=8,fontweight="bold")
        st.pyplot(fig3)

        # Forma reciente
        st.markdown("### 📊 Forma Reciente")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{home}**")
            emoji = "".join(["🟢" if r=="W" else "🟡" if r=="D" else "🔴" for r in forma_h])
            st.write(emoji if emoji else "Sin datos recientes")
        with col2:
            st.write(f"**{away}**")
            emoji = "".join(["🟢" if r=="W" else "🟡" if r=="D" else "🔴" for r in forma_a])
            st.write(emoji if emoji else "Sin datos recientes")

        # Análisis de valor
        if usar_cuotas and cuota_h and cuota_e and cuota_a:
            st.markdown("### 💰 Análisis de Valor")
            val_h = prob_h * cuota_h - 1
            val_e = prob_d * cuota_e - 1
            val_a = prob_a * cuota_a - 1
            col1, col2, col3 = st.columns(3)
            for col, label, val, prob, cuota in [
                (col1, f"Gana {home}", val_h, prob_h, cuota_h),
                (col2, "Empate", val_e, prob_d, cuota_e),
                (col3, f"Gana {away}", val_a, prob_a, cuota_a)
            ]:
                with col:
                    if val > 0.05:
                        st.success(f"✅ **{label}**\nValor: +{val:.3f}\nCuota: {cuota} | Modelo: {prob*100:.1f}%")
                    else:
                        st.error(f"❌ **{label}**\nValor: {val:.3f}\nCuota: {cuota} | Modelo: {prob*100:.1f}%")
