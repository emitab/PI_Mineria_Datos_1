import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Dataset", page_icon="📁", layout="wide")

st.title("📁 Dataset")
st.markdown("Inspección inicial, calidad de datos y pipeline de limpieza.")
st.markdown("---")

# ── Carga ─────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_raw():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "streaming_users_dirty.json")
    return pd.read_json(path)

@st.cache_data
def cargar_procesado():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "streaming_users_processed.json")
    return pd.read_json(path)

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

st.markdown("### Diccionario de variables")
st.markdown("""
- `user_id` = identificador de usuario
- `age` = edad
- `subscription_plan` = plan de suscripcion
- `monthly_watch_time_mins` = minutos vistos en el mes
- `country` = pais
- `favorite_genre` = genero favorito
- `last_login_date` = fecha de ultimo inicio de sesion
- `customer_support_tickets` = tickets de atencion al cliente
""")

diccionario = pd.DataFrame({
    "Variable": [
        "user_id", "age", "subscription_plan", "monthly_watch_time_mins",
        "country", "favorite_genre", "last_login_date", "customer_support_tickets"
    ],
    "Tipo original": [
        "int64", "int64", "object", "float64",
        "object", "object", "object", "int64"
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
st.markdown("""
Antes de aplicar cualquier transformación, inspeccionamos la estructura
general del dataset: dimensiones, tipos de datos asignados automáticamente
y presencia de valores nulos por columna. Esta revisión es la base de
evidencia para todas las decisiones posteriores.
""")

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
    **Interpretación:**
    - 8160 filas, 8 columnas.
    - En columnas `monthly_watch_time_mins` (193), `favorite_genre` (240) y `last_login_date` (384) hay presencia de datos nulos, sumando un total de 817.
    - La columna `last_login_date` tiene un tipo de dato erróneo (está en object, debe ser date).
    - Valor mínimo de `age` y `monthly_watch_time_mins` son negativos, cuando no es posible.
    - Valor máximo de `age` y `monthly_watch_time_mins` tienen valores imposibles.
    - Análisis de medidas se realizan luego del procesamiento y limpieza de la base de datos.
    """)
else:
    st.warning("No se encontró el dataset original en `data/raw/`.")

# ── 3. Pipeline de limpieza ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 3. Pipeline de limpieza")

with st.expander("📌 1. Duplicados — eliminación directa"):
    st.markdown("""
    Un registro idéntico en todas sus columnas no aporta información nueva.
    Se eliminaron directamente sin necesidad de criterio adicional.
    """)

with st.expander("📌 2. Estandarización de variables categóricas"):
    st.markdown("""
    Se detectaron variantes inconsistentes en `subscription_plan` (Estándar / estandar / Std),
    `country` (Brasil / Brazil / brasil) y `favorite_genre` (Acción / ACCIÓN / accion / Action).
    Estas diferencias impiden cualquier agrupación o análisis por categoría.
    Se unifican bajo una forma canónica en minúsculas sin tildes.
    No se elimina ningún registro: los valores son válidos, solo están mal escritos.

    Los nulos en `favorite_genre` son un caso especial: no podemos determinar si representan
    un dato no registrado o una preferencia genuinamente indefinida del usuario.
    Por eso se introduce la categoría explícita `sin_definir` en lugar de imputar
    un género o eliminar el registro, preservando la información sobre la ausencia del dato.

    Lo mismo aplica para `subscription_plan` y `country`: cualquier valor que no pueda
    mapearse se etiqueta como `sin_definir` para no perder el registro.
    """)

with st.expander("📌 3. Fechas (last_login_date)"):
    st.markdown("""
    La columna `last_login_date` presenta tres formatos mezclados (YYYY-MM-DD,
    MM-DD-YYYY, YYYY/MM/DD) y fue cargada como `object`.
    Se fuerza la conversión con `infer_datetime_format=True` y `errors="coerce"`,
    lo que convierte automáticamente a `NaT` tanto los nulos originales como las
    fechas inválidas (por ejemplo, mes 15).

    Los registros con fecha futura son imposibles en el contexto real de la plataforma
    y se eliminan. Los nulos restantes tampoco son recuperables: no existe criterio
    válido para imputar una fecha de último acceso, por lo que esas filas se eliminan.

    A partir de la fecha limpia se deriva `days_since_last_login`, variable numérica
    con mayor utilidad analítica y candidata para PCA.
    """)

with st.expander("📌 4. Edad (age)"):
    st.markdown("""
    Se detectaron valores de edad iguales a 0 y superiores a 120, ambos imposibles
    para usuarios reales de una plataforma de streaming.
    No existe criterio válido para imputar la edad de un usuario desconocido,
    por lo que estos registros se eliminan directamente.
    """)

with st.expander("📌 5. Minutos de visualización (monthly_watch_time_mins)"):
    st.markdown("""
    Se aplican tres criterios en orden.

    Primero, valores negativos: un tiempo de visualización negativo es físicamente
    imposible y no recuperable, se eliminan.

    Segundo, valores superiores a 43.200 minutos (60 × 24 × 30): superan el límite
    absoluto de un mes y se eliminan. Valores altos pero dentro del rango posible,
    como los cercanos a 4.000 minutos (~68 horas/mes), son outliers estadísticos
    pero no errores de dato y se conservan.

    Tercero, para los nulos restantes se imputa con la mediana agrupada por
    `subscription_plan`. Esta decisión se justifica porque el comportamiento de
    visualización varía razonablemente según el plan contratado, y la mediana
    es robusta frente a los outliers conservados. Si algún registro tiene
    `subscription_plan` sin_definir y no puede agruparse correctamente,
    se aplica la mediana global como segundo criterio.
    """)

with st.expander("📌 6. Tickets de soporte (customer_support_tickets)"):
    st.markdown("""
    Se detectó un registro con 99 tickets de soporte, valor extremadamente alto
    pero técnicamente posible para un usuario con problemas recurrentes.
    No se elimina ni imputa: se conserva y se documenta como observación de
    interés analítico.

    Sin embargo, se detectaron también valores negativos, que son imposibles:
    un ticket no puede tener cantidad negativa. Esos registros se eliminan.
    """)

with st.expander("📌 7. Estado final — guardado y log ETL"):
    st.markdown("""
    El dataset limpio se guarda en `data/processed/` preservando el original
    intacto en `data/raw/`. El log ETL registra cada transformación con su
    impacto en filas y nulos, permitiendo comparar el estado inicial y final.

    Verificamos que el dataset resultante no tenga nulos sin resolver,
    que los tipos de datos sean correctos y que las estadísticas
    descriptivas sean coherentes con los rangos esperados.
    """)

# ── 4. Log ETL ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 4. Log ETL")

try:
    path_log = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "pipeline_log.csv")
    log = pd.read_csv(path_log)
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
            "Limpieza de minutos visualización",
            "Limpieza de tickets de soporte",
        ],
        "Filas": [8160, 8034, 8034, 7250, 7159, 7091, 7067],
        "Nulos": [753, 753, 513, 177, 175, 0, 0],
        "Retención (%)": [100.0, 98.46, 98.46, 88.85, 87.73, 86.90, 86.61],
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
