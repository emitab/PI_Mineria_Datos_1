import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")

st.title("📊 Análisis Exploratorio de Datos")
st.markdown("Cinco preguntas concretas respondidas con evidencia del dataset.")
st.markdown("---")

# ── Carga ─────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "streaming_users_processed.json")
    return pd.read_json(path)

try:
    df = cargar()
    datos_ok = True
except Exception:
    datos_ok = False
    st.error("No se encontró el dataset procesado en `data/processed/`.")
    st.stop()

# Configuración visual coherente con los notebooks
sns.set_theme(style="darkgrid")

# Derivar columna grupo_edad si no existe
if "grupo_edad" not in df.columns:
    bins = [0, 17, 25, 35, 45, 60, 100]
    labels = ["<18", "18-25", "26-35", "36-45", "46-60", ">60"]
    df["grupo_edad"] = pd.cut(df["age"], bins=bins, labels=labels)

# Derivar grupo_tickets si no existe
if "grupo_tickets" not in df.columns:
    df["grupo_tickets"] = pd.cut(
        df["customer_support_tickets"],
        bins=[-1, 0, 2, 5, 150],
        labels=["Sin tickets", "1-2", "3-5", "Más de 5"]
    )

# ── Preguntas de análisis ──────────────────────────────────────────────────────
st.markdown("## Preguntas de análisis")
st.markdown("""
Este EDA busca responder cinco preguntas concretas sobre el comportamiento de los usuarios:

1. **¿Cómo se distribuye la edad de los usuarios?**
2. **¿Qué plan de suscripción predomina en la plataforma?**
3. **¿El plan de suscripción se refleja en el consumo real?**
4. **¿Los usuarios con más tickets llevan más días sin ingresar?**
5. **¿Qué relaciones existen entre las variables numéricas?**
""")
st.markdown("---")

# ── VIZ 1 — Distribución de edad (univariada) ─────────────────────────────────
st.markdown("## Viz 1 · Distribución de usuarios por grupo etario")
st.caption("Univariada · Pregunta 1")

conteo = df["grupo_edad"].value_counts().sort_index()

fig1, ax1 = plt.subplots(figsize=(8, 4))
bars = ax1.bar(
    conteo.index, conteo.values,
    color=sns.color_palette("Blues_d", len(conteo)),
    edgecolor="white"
)
for bar in bars:
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        str(int(bar.get_height())),
        ha="center", va="bottom", fontsize=9
    )
ax1.set_title("Distribución de usuarios por grupo etario")
ax1.set_xlabel("Grupo de edad")
ax1.set_ylabel("Cantidad de usuarios")
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

col_t1, _ = st.columns([2, 1])
with col_t1:
    tabla_edades = pd.DataFrame({
        "Cantidad": df["grupo_edad"].value_counts().sort_index(),
        "Porcentaje (%)": round(df["grupo_edad"].value_counts(normalize=True).sort_index() * 100, 2)
    })
    st.dataframe(tabla_edades, use_container_width=True)

st.markdown("""
**Interpretación:** La distribución muestra que la base de usuarios es predominantemente adulta joven.
La mayor concentración se encuentra en el rango de 26 a 35 años, con presencia reducida
en los extremos. Esto condiciona cualquier decisión de segmentación posterior:
las estrategias orientadas al segmento 26-35 tienen mayor cobertura sobre la base de clientes.
""")
st.markdown("---")

# ── VIZ 2 — Distribución de planes (univariada) ───────────────────────────────
st.markdown("## Viz 2 · Distribución de planes de suscripción")
st.caption("Univariada · Pregunta 2")

conteo_planes = df["subscription_plan"].value_counts()

fig2, ax2 = plt.subplots(figsize=(6, 4))
bars2 = ax2.barh(
    conteo_planes.index, conteo_planes.values,
    color=sns.color_palette("Blues_d", len(conteo_planes)),
    edgecolor="white"
)
total = conteo_planes.sum()
for bar, val in zip(bars2, conteo_planes.values):
    ax2.text(
        bar.get_width() + 15,
        bar.get_y() + bar.get_height() / 2,
        f"{val / total * 100:.1f}%",
        va="center", fontsize=9
    )
ax2.set_title("Distribución de planes de suscripción")
ax2.set_xlabel("Cantidad de usuarios")
ax2.set_ylabel("Plan")
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

col_t2, _ = st.columns([2, 1])
with col_t2:
    tabla_planes = pd.DataFrame({
        "Cantidad": df["subscription_plan"].value_counts(),
        "Porcentaje (%)": round(df["subscription_plan"].value_counts(normalize=True) * 100, 2)
    })
    tabla_planes.index.name = "Plan de suscripción"
    st.dataframe(tabla_planes, use_container_width=True)

st.markdown("""
**Interpretación:** El plan básico es el más utilizado, representando aproximadamente el 44%
de la base de usuarios. Este resultado establece al segmento básico como la referencia
principal para comparaciones entre planes en el análisis bivariado.
""")
st.markdown("---")

# ── VIZ 3 — Consumo por plan (bivariada) ──────────────────────────────────────
st.markdown("## Viz 3 · Consumo mensual de minutos por plan de suscripción")
st.caption("Bivariada · Pregunta 3")

orden = ["basico", "estandar", "premium"]

fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.boxplot(
    data=df, x="subscription_plan", y="monthly_watch_time_mins",
    order=orden, palette="Blues", hue="subscription_plan", legend=False,
    width=0.5, linewidth=1.2,
    flierprops=dict(marker="o", markersize=3, alpha=0.3),
    ax=ax3
)
ax3.set_title("Consumo mensual de minutos por plan de suscripción")
ax3.set_xlabel("Plan de suscripción")
ax3.set_ylabel("Minutos mensuales")
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

col_t3, _ = st.columns([2, 1])
with col_t3:
    resumen = (
        df.groupby("subscription_plan")["monthly_watch_time_mins"]
        .agg(Cantidad="count", Promedio="mean", Mediana="median", Minimo="min", Maximo="max")
        .round(2)
    )
    st.dataframe(resumen, use_container_width=True)

st.markdown("""
**Interpretación:** El consumo promedio de minutos aumenta progresivamente del plan básico
al estándar y al premium. Los usuarios de planes superiores utilizan más la plataforma,
lo que es consistente con la expectativa de que quienes pagan por un servicio más completo
hacen un uso más intensivo del mismo. La dispersión también aumenta en los planes superiores,
indicando mayor heterogeneidad en el comportamiento de esos segmentos.
""")
st.markdown("---")

# ── VIZ 4 — Tickets y días sin login (bivariada) ──────────────────────────────
st.markdown("## Viz 4 · Días sin ingresar según nivel de tickets de soporte")
st.caption("Bivariada · Pregunta 4")

fig4, ax4 = plt.subplots(figsize=(8, 4))
sns.boxplot(
    data=df, x="grupo_tickets", y="days_since_last_login",
    palette="Blues", hue="grupo_tickets", legend=False,
    width=0.5, linewidth=1.2,
    flierprops=dict(marker="o", markersize=3, alpha=0.3),
    ax=ax4
)
ax4.set_title("Días sin ingresar según nivel de tickets de soporte")
ax4.set_xlabel("Grupo de tickets de soporte")
ax4.set_ylabel("Días desde último login")
plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

correlacion = df["customer_support_tickets"].corr(df["days_since_last_login"])
st.metric("Correlación de Pearson (tickets vs días sin login)", f"{correlacion:.3f}")

col_t4, _ = st.columns([2, 1])
with col_t4:
    tabla_tickets = (
        df.groupby("grupo_tickets", observed=False)["days_since_last_login"]
        .agg(Cantidad_Usuarios="count", Promedio_Dias="mean", Mediana="median")
        .round(2)
    )
    st.dataframe(tabla_tickets, use_container_width=True)

st.markdown("""
**Interpretación:** El coeficiente de correlación obtenido fue 0.004, indicando una relación
prácticamente nula entre la cantidad de tickets de soporte y los días sin ingresar.
Los boxplots por grupo confirman que las distribuciones de días sin login son similares
entre usuarios con distintos niveles de tickets. Esto sugiere que una experiencia técnica
negativa no se traduce necesariamente en alejamiento de la plataforma en este dataset.
""")
st.markdown("---")

# ── VIZ 5 — Heatmap de correlación (multivariada) ─────────────────────────────
st.markdown("## Viz 5 · Matriz de correlación entre variables numéricas")
st.caption("Multivariada · Pregunta 5")

variables_num = df[["age", "monthly_watch_time_mins", "customer_support_tickets", "days_since_last_login"]]
matriz_corr = variables_num.corr()

fig5, ax5 = plt.subplots(figsize=(6, 5))
sns.heatmap(
    matriz_corr, annot=True, fmt=".2f", cmap="Blues",
    vmin=-1, vmax=1, square=True, linewidths=0.5, linecolor="white",
    ax=ax5
)
ax5.set_title("Matriz de correlación entre variables numéricas")
plt.tight_layout()
st.pyplot(fig5)
plt.close(fig5)

col_t5, _ = st.columns([2, 1])
with col_t5:
    st.dataframe(matriz_corr.round(4), use_container_width=True)

st.markdown("""
**Interpretación:** La matriz muestra coeficientes muy cercanos a cero entre todas las
variables numéricas, indicando ausencia de relaciones lineales significativas.
La edad, el tiempo de visualización, los tickets de soporte y los días desde el último
acceso describen aspectos independientes del comportamiento del usuario.
Este resultado anticipa que PCA no podrá reducir la dimensionalidad de forma efectiva,
ya que no existe redundancia entre las variables.
""")
