import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset", page_icon="📁", layout="wide")

st.title("📁 Dataset")
st.markdown("Inspección inicial, calidad de datos y pipeline de limpieza.")
st.markdown("---")

# ── Carga ─────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_raw():
    return pd.read_json("../data/raw/streaming_users_dirty.json")

@st.cache_data
def cargar_procesado():
    return pd.read_json("../data/processed/streaming_users_processed.json")

try:
    df_raw = cargar_raw()
    raw_ok = True
except Exception:
    raw_ok = False

try:
    df_proc = cargar_procesado()
    proc_ok = True
except Exception:
    proc_ok = False

# ── 1. Descripción general ────────────────────────────────────────────────────
st.markdown("## 1. Descripción general")

st.markdown("""
El dataset contiene registros de usuarios de una plataforma de streaming.
Cada fila representa un usuario único con información demográfica, de suscripción
y de comportamiento dentro de la plataforma.
""")

st.markdown("### Diccionario de variables")

diccionario = pd.DataFrame({
    "Variable": [
        "user_id", "age", "subscription_plan", "monthly_watch_time_mins",
        "country", "favorite_genre", "last_login_date", "customer_support_tickets"
    ],
    "Tipo original": [
        "int", "float", "object", "float",
        "object", "object", "object", "int"
    ],
    "Descripción": [
        "Identificador único de usuario",
        "Edad del usuario",
        "Plan de suscripción (básico / estándar / premium)",
        "Minutos vistos en el mes",
        "País de origen del usuario",
        "Género de contenido preferido",
        "Fecha del último inicio de sesión",
        "Cantidad de tickets de atención al cliente"
    ]
})

st.dataframe(diccionario, use_container_width=True, hide_index=True)

# ── 2. Calidad inicial ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 2. Calidad inicial")

if raw_ok:
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", f"{df_raw.shape[0]:,}")
    col2.metric("Columnas", df_raw.shape[1])
    col3.metric("Nulos totales", int(df_raw.isnull().sum().sum()))

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown("**Vista previa del dataset original**")
        st.dataframe(df_raw.head(8), use_container_width=True)

    with col_der:
        st.markdown("**Nulos por columna**")
        nulos = df_raw.isnull().sum().reset_index()
        nulos.columns = ["Variable", "Nulos"]
        nulos["% sobre total"] = (nulos["Nulos"] / len(df_raw) * 100).round(2)
        st.dataframe(nulos, use_container_width=True, hide_index=True)

    st.info("""
    **Observaciones iniciales:**
    - Nulos en `monthly_watch_time_mins` (193), `favorite_genre` (240) y
      `last_login_date` (384). Total: 817 valores faltantes.
    - `last_login_date` está cargada como `object`, requiere conversión a `datetime`.
    - Valores mínimos negativos en `age` y `monthly_watch_time_mins` (imposibles).
    - Valores máximos fuera de rango en `age` y `monthly_watch_time_mins`.
    """)
else:
    st.warning("No se encontró el dataset original en `data/raw/`.")

# ── 3. Pipeline de limpieza ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 3. Principales transformaciones")

with st.expander("📌 Duplicados — eliminación directa"):
    st.markdown("""
    Un registro idéntico en todas sus columnas no aporta información nueva.
    Se eliminaron directamente sin necesidad de criterio adicional.
    """)

with st.expander("📌 Variables categóricas (subscription_plan, country, favorite_genre)"):
    st.markdown("""
    Se detectaron variantes inconsistentes: `Estándar / estandar / Std`, `Brasil / Brazil`, entre otros.
    Se unificaron bajo una forma canónica en minúsculas sin tildes.

    Los nulos en `favorite_genre` se etiquetaron como `sin_definir` en lugar de imputar o eliminar,
    porque no es posible distinguir si representan un dato no registrado o una preferencia indefinida.
    """)

with st.expander("📌 Fechas (last_login_date)"):
    st.markdown("""
    Se identificaron tres formatos mezclados (`YYYY-MM-DD`, `MM-DD-YYYY`, `YYYY/MM/DD`).
    Se forzó conversión con `errors='coerce'` → fechas inválidas quedan como `NaT`.

    - **Fechas futuras** → imposibles en el contexto real → eliminadas.
    - **Nulos restantes** → sin criterio válido de imputación → eliminados.
    - Se derivó `days_since_last_login` como variable numérica con mayor utilidad analítica.
    """)

with st.expander("📌 Edad (age)"):
    st.markdown("""
    Valores ≤ 0 y ≥ 120 son imposibles para usuarios reales.
    Sin criterio válido de imputación → eliminados directamente.
    """)

with st.expander("📌 Minutos de visualización (monthly_watch_time_mins)"):
    st.markdown("""
    - **Valores negativos** → físicamente imposibles → eliminados.
    - **Valores > 43.200 min** (límite absoluto de un mes de 30 días × 24 hs × 60 min) → eliminados.
    - **Outliers estadísticos válidos** (cercanos a 4.000 min) → conservados.
    - **Nulos** → imputados con mediana agrupada por `subscription_plan`,
      con mediana global como criterio de respaldo.
    """)

with st.expander("📌 Tickets de soporte (customer_support_tickets)"):
    st.markdown("""
    - **Valores negativos** → imposibles → eliminados.
    - **Outlier de 99 tickets** → técnicamente posible para usuario con problemas recurrentes → conservado.
    """)

# ── 4. Log ETL ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 4. Log ETL")

try:
    log = pd.read_csv("../logs/pipeline_log.csv")
    st.dataframe(log, use_container_width=True, hide_index=True)
except Exception:
    log_data = {
        "Paso": ["01", "02", "03", "04", "05", "06", "07"],
        "Descripción": [
            "Dataset original cargado",
            "Eliminación de duplicados",
            "Estandarización de categóricas",
            "Limpieza de fechas",
            "Limpieza de edades",
            "Limpieza de minutos de visualización",
            "Limpieza de tickets de soporte",
        ],
        "Filas": [8160, 8135, 8135, 7741, 7662, 7598, 7561],
        "Nulos": [817, 812, 428, 196, 193, 0, 0],
        "Retención (%)": [100.0, 99.69, 99.69, 94.87, 93.87, 93.09, 92.66],
    }
    st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)

# ── 5. Dataset procesado ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 5. Dataset procesado")

if proc_ok:
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas finales", f"{df_proc.shape[0]:,}")
    col2.metric("Columnas finales", df_proc.shape[1])
    col3.metric("Nulos restantes", int(df_proc.isnull().sum().sum()))
    st.dataframe(df_proc.head(8), use_container_width=True)
else:
    st.warning("No se encontró el dataset procesado en `data/processed/`.")
