import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Conclusiones", page_icon="📝", layout="wide")

st.title("📝 Conclusiones")
st.markdown("Hallazgos del análisis, limitaciones del proyecto y próximos pasos.")
st.markdown("---")

# ── 1. Resumen del proceso ────────────────────────────────────────────────────
st.markdown("## 1. Resumen del proceso")

st.markdown("""
Este proyecto analizó un dataset de usuarios de una plataforma de streaming
que contenía 8.160 registros y 8 variables originales. A lo largo de cinco
etapas se construyó un pipeline reproducible que va desde la inspección
inicial hasta la reducción de dimensionalidad.

El proceso siguió el siguiente orden:

- **Inspección inicial**: se identificaron nulos en tres columnas, tipo de dato
  incorrecto en `last_login_date`, valores imposibles en `age` y
  `monthly_watch_time_mins`, y duplicados.
- **Limpieza y preparación**: se tomaron decisiones justificadas por evidencia
  en cada variable, conservando el dataset original intacto y registrando
  cada transformación en el log ETL.
- **EDA**: se respondieron cinco preguntas concretas sobre el comportamiento
  de los usuarios mediante análisis univariado, bivariado y multivariado.
- **PCA**: se aplicó reducción de dimensionalidad sobre las cuatro variables
  numéricas, documentando el escalamiento, la varianza explicada y la
  interpretación de los loadings.
""")

# ── 2. Hallazgos principales ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 2. Hallazgos principales")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Sobre el perfil de los usuarios")
    st.markdown("""
    La base de usuarios es predominantemente adulta joven: el grupo con mayor
    concentración se encuentra entre los 26 y 35 años, con una distribución
    relativamente uniforme entre los 18 y 60 años y muy pocos usuarios en los
    extremos. El plan básico es el más frecuente, representando aproximadamente
    el 44% de la base, lo que indica que la mayor parte de los usuarios accede
    al nivel de servicio más elemental.
    """)

    st.markdown("### Sobre el consumo según plan")
    st.markdown("""
    El análisis bivariado mostró que el consumo mensual de minutos crece
    de forma consistente de básico a estándar y de estándar a premium.
    Los usuarios premium presentan una mediana de minutos claramente superior
    a los otros dos planes, lo que sugiere que el tipo de suscripción contratada
    se refleja en el uso real de la plataforma. Sin embargo, la dispersión
    dentro de cada plan es considerable, lo que indica que el plan no explica
    por sí solo el comportamiento de visualización.
    """)

with col2:
    st.markdown("### Sobre tickets de soporte y actividad")
    st.markdown("""
    La correlación de Pearson entre `customer_support_tickets` y
    `days_since_last_login` fue de 0.004, prácticamente nula. El análisis
    por grupos de tickets confirmó visualmente que las medianas de días sin
    ingresar son similares entre usuarios sin tickets y usuarios con muchos.
    La hipótesis de que una experiencia técnica negativa se asocia con el
    abandono de la plataforma no encuentra respaldo en los datos disponibles.
    """)

    st.markdown("### Sobre la estructura del dataset")
    st.markdown("""
    La matriz de correlación del EDA mostró coeficientes cercanos a cero entre
    todas las variables numéricas. Este resultado fue confirmado por el PCA:
    las cuatro componentes principales explican proporciones casi idénticas
    de varianza (entre 24.6% y 25.4%), sin ningún codo visible en el scree plot.
    Las variables describen dimensiones independientes del comportamiento del
    usuario y no presentan redundancia entre sí.
    """)

# ── 3. Interpretación del PCA ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 3. Interpretación del PCA")

st.markdown("""
PC1 puede interpretarse como una dimensión de actividad e involucramiento:
valores bajos corresponden a usuarios más jóvenes con mayor consumo mensual,
mientras que valores altos corresponden a usuarios con más tickets reportados
y mayor tiempo sin ingresar. PC2 captura principalmente la oposición entre
edad y tickets de soporte contra el consumo mensual.

El resultado más relevante del PCA no es la reducción de dimensionalidad
en sí —que en este caso no es posible sin pérdida significativa—, sino
la confirmación de que las cuatro variables aportan información
independiente y complementaria sobre el usuario.
""")

# Tabla resumen del proceso
@st.cache_data
def cargar_procesado():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "streaming_users_processed.json")
    return pd.read_json(path)

try:
    df = cargar_procesado()
    resumen = pd.DataFrame([
        {"Etapa": "Dataset original",           "Filas": 8160, "Variables": 8},
        {"Etapa": "Tras limpieza y preparación", "Filas": df.shape[0], "Variables": df.shape[1]},
    ])
    st.dataframe(resumen, use_container_width=True, hide_index=True)
    eliminados = 8160 - df.shape[0]
    st.caption(f"Registros eliminados durante la limpieza: {eliminados} ({eliminados/8160*100:.1f}%) · Variable derivada incorporada: days_since_last_login")
except Exception:
    pass

# ── 4. Limitaciones ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 4. Limitaciones")

st.markdown("""
El alcance de las conclusiones se encuentra condicionado por la información
disponible y por las decisiones documentadas durante el proceso.

- El dataset no incluye información sobre el contenido consumido, lo que
  impide analizar si el género favorito declarado se corresponde con el
  consumo real.
- La variable `favorite_genre` presenta un 3% de registros etiquetados
  como `sin_definir` por ambigüedad en el dato original, lo que limita
  el análisis por preferencia de contenido.
- La ausencia de una variable temporal que indique cuándo se registró
  cada usuario impide analizar la evolución del comportamiento a lo largo
  del tiempo.
- Las correlaciones nulas entre variables numéricas limitan la capacidad
  de PCA para resumir el dataset de forma compacta.
""")

# ── 5. Mejoras futuras ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 5. Mejoras futuras")

st.markdown("""
- Una mejora futura podría consistir en incorporar datos de consumo
  por género de contenido, lo que permitiría contrastar la preferencia
  declarada con el comportamiento real de visualización.
- Podría incorporarse una variable de fecha de registro para analizar
  la retención de usuarios a lo largo del tiempo y detectar patrones
  de abandono.
- Un análisis de clustering sobre las componentes principales podría
  complementar el PCA e identificar segmentos de usuarios con
  comportamientos diferenciados, aun cuando las variables no presenten
  correlación lineal entre sí.
""")

# ── 6. Referencias ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 6. Referencias del proyecto")

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown("""
    **Repositorio GitHub**
    🔗 [Ver repositorio](https://github.com/emitab/PI_Mineria_Datos_1)

    **Aplicación Streamlit Cloud**
    🔗 [Ver aplicación pública](https://streamlit.io/)
    """)
with col_r2:
    st.markdown("""
    **Notebooks**
    - `01_inspeccion_inicial.ipynb`
    - `02_calidad_y_limpieza.ipynb`
    - `03_eda.ipynb`
    - `04_pca.ipynb`
    - `05_conclusiones.ipynb`
    """)

st.markdown("---")
st.caption("Taboada, Emiliano Nicolás · Pérez Carletti, Martín · Minería de Datos 1")
