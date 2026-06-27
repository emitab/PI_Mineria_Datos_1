import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="PCA", page_icon="🔷", layout="wide")

st.title("🔷 Escalamiento y PCA")
st.markdown("Reducción de dimensionalidad e interpretación de componentes principales.")
st.markdown("---")

# ── Carga ─────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar():
    return pd.read_json("../data/processed/streaming_users_processed.json")

try:
    df = cargar()
except Exception:
    st.error("No se encontró el dataset procesado en `data/processed/`.")
    st.stop()

sns.set_theme(style="darkgrid")

variables = ["age", "monthly_watch_time_mins", "customer_support_tickets", "days_since_last_login"]
X = df[variables]

# ── 1. Variables utilizadas ───────────────────────────────────────────────────
st.markdown("## 1. Variables utilizadas")

st.markdown("""
Se seleccionaron las cuatro variables numéricas continuas del dataset:
`age`, `monthly_watch_time_mins`, `customer_support_tickets` y `days_since_last_login`.

Estas variables describen dimensiones distintas del comportamiento del usuario:
perfil demográfico, consumo, experiencia técnica y actividad reciente.
Las variables categóricas (`subscription_plan`, `country`, `favorite_genre`)
fueron excluidas porque PCA opera exclusivamente sobre variables numéricas.
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Variables incluidas**")
    vars_df = pd.DataFrame({
        "Variable": variables,
        "Descripción": ["Edad del usuario", "Minutos vistos en el mes",
                        "Tickets de atención al cliente", "Días desde el último login"]
    })
    st.dataframe(vars_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**Estadísticas antes de escalar**")
    st.dataframe(X.describe().round(2), use_container_width=True)

# ── 2. Escalamiento ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 2. Escalamiento — StandardScaler")

st.markdown("""
Antes de aplicar PCA es obligatorio escalar las variables.
PCA es sensible a la magnitud de cada variable: sin escalar, `monthly_watch_time_mins`
(que puede superar los 4.000 minutos) dominaría completamente el análisis sobre
`customer_support_tickets` (que rara vez supera 5), simplemente por diferencia de escala
y no por importancia real.

Se aplica `StandardScaler`, que transforma cada variable para que tenga
media 0 y desvío estándar 1, haciendo que todas contribuyan en igualdad de condiciones.
""")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

verificacion = pd.DataFrame({
    "Variable": variables,
    "Media post-escalado": X_scaled.mean(axis=0).round(6),
    "Desvío post-escalado": X_scaled.std(axis=0).round(4)
})
st.dataframe(verificacion, use_container_width=True, hide_index=True)
st.caption("Las medias son aproximadamente 0 y los desvíos aproximadamente 1, confirmando el escalamiento correcto.")

# ── 3. PCA y varianza explicada ───────────────────────────────────────────────
st.markdown("---")
st.markdown("## 3. Varianza explicada")

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

varianza = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza)

resumen_pca = pd.DataFrame({
    "Componente": [f"PC{i+1}" for i in range(4)],
    "Varianza explicada (%)": (varianza * 100).round(2),
    "Varianza acumulada (%)": (varianza_acumulada * 100).round(2)
})

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("PC1", f"{varianza[0]*100:.1f}%")
col_m2.metric("PC2", f"{varianza[1]*100:.1f}%")
col_m3.metric("PC3", f"{varianza[2]*100:.1f}%")
col_m4.metric("PC4", f"{varianza[3]*100:.1f}%")

st.dataframe(resumen_pca, use_container_width=True, hide_index=True)

# ── VIZ 1 — Scree plot ────────────────────────────────────────────────────────
st.markdown("### Viz 1 · Varianza explicada por componente principal")

componentes = [f"PC{i+1}" for i in range(4)]

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.bar(
    componentes, varianza * 100,
    color=sns.color_palette("Blues_d", 4),
    edgecolor="white", label="Varianza individual"
)
ax2 = ax1.twinx()
ax2.plot(
    componentes, varianza_acumulada * 100,
    color="tomato", marker="o", linewidth=1.5, label="Varianza acumulada"
)
ax2.set_ylabel("Varianza acumulada (%)")
ax2.set_ylim(0, 110)

for i, v in enumerate(varianza * 100):
    ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

ax1.set_title("Varianza explicada por componente principal")
ax1.set_xlabel("Componente")
ax1.set_ylabel("Varianza explicada (%)")
fig1.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

st.markdown("""
**Interpretación:** Las cuatro componentes explican proporciones casi idénticas de varianza
(≈25% cada una), con una curva acumulada perfectamente lineal y sin ningún codo visible.
Este resultado confirma lo observado en la matriz de correlación del EDA: las variables
no están correlacionadas entre sí, por lo que cada una aporta información independiente.
PCA no encuentra ninguna dirección que concentre más información que las demás.
Retener 2 componentes implicaría conservar solo el 50.6% de la información original,
lo cual no representa una reducción útil en términos prácticos.
""")

# ── VIZ 2 — Loadings ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Viz 2 · Contribución de variables a PC1 y PC2 (loadings)")

loadings = pd.DataFrame(
    pca.components_.T,
    index=variables,
    columns=[f"PC{i+1}" for i in range(4)]
)

x = np.arange(len(variables))
width = 0.35

fig2, ax3 = plt.subplots(figsize=(7, 4))
ax3.bar(x - width/2, loadings["PC1"], width,
        color=sns.color_palette("Blues_d", 4)[2], label="PC1", edgecolor="white")
ax3.bar(x + width/2, loadings["PC2"], width,
        color=sns.color_palette("Blues_d", 4)[0], label="PC2", edgecolor="white")
ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax3.set_xticks(x)
ax3.set_xticklabels(["age", "watch_mins", "tickets", "days_login"], fontsize=9)
ax3.set_title("Contribución de variables a PC1 y PC2 (loadings)")
ax3.set_ylabel("Loading")
ax3.legend()
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

st.markdown("**Tabla de loadings completa**")
st.dataframe(loadings.round(4), use_container_width=True)

st.markdown("""
**Interpretación:**

**PC1** tiene cargas negativas en `age` y `monthly_watch_time_mins`, y cargas positivas
en `customer_support_tickets` y `days_since_last_login`. PC1 puede interpretarse como
una dimensión de actividad e involucramiento: valores bajos corresponden a usuarios
jóvenes con alto consumo; valores altos a usuarios con más tickets y mayor tiempo
sin ingresar.

**PC2** muestra cargas positivas en `age` y `customer_support_tickets`, capturando
una dimensión demográfica y de fricción técnica: usuarios mayores con más tickets
reportados, independientemente de su actividad reciente.

En ambos casos `monthly_watch_time_mins` tiene el loading más bajo en PC2,
indicando que el tiempo de visualización es la variable que menos estructura aporta
a las componentes principales.
""")

# ── 4. Interpretación general ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 4. Interpretación general")

st.info("""
Las cuatro variables describen aspectos distintos e independientes del comportamiento
del usuario. PCA no puede comprimir el dataset sin pérdida significativa de información.

Para capturar el 75% de la varianza se necesitan 3 componentes; el 100% requiere
las 4 originales. En este caso, el resultado relevante no es la reducción de
dimensionalidad en sí, sino entender qué estructura captura cada componente:
actividad/involucramiento (PC1) y perfil demográfico con fricción técnica (PC2).
""")
