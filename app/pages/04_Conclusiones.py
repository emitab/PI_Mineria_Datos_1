import streamlit as st

st.set_page_config(page_title="Conclusiones", page_icon="📝", layout="wide")

st.title("📝 Conclusiones")
st.markdown("Hallazgos del análisis, limitaciones del proyecto y próximos pasos.")
st.markdown("---")

# ── 1. Hallazgos ──────────────────────────────────────────────────────────────
st.markdown("## 1. Hallazgos principales")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Dataset y calidad")
    st.markdown("""
    El dataset original contenía 8.160 registros con 817 valores nulos distribuidos
    en tres columnas, valores imposibles en variables numéricas (`age` negativa,
    minutos fuera de rango) y formatos inconsistentes en fechas y categóricas.
    Tras el pipeline de limpieza se conservaron 7.561 registros (92.66%),
    con decisiones documentadas y justificadas en evidencia observada para cada transformación.
    """)

    st.markdown("### Perfil de la base de usuarios")
    st.markdown("""
    La plataforma tiene una base de usuarios predominantemente adulta joven,
    con mayor concentración en el rango de 26 a 35 años.
    El plan básico es el más frecuente (≈44% de los usuarios),
    lo que lo establece como referencia natural para comparaciones entre segmentos.
    """)

with col2:
    st.markdown("### Comportamiento por plan")
    st.markdown("""
    El consumo mensual de minutos aumenta progresivamente del plan básico al premium,
    lo que es consistente con la expectativa de que los usuarios que pagan más
    utilizan más intensivamente la plataforma.
    La dispersión también es mayor en los planes superiores,
    indicando heterogeneidad dentro de esos segmentos.
    """)

    st.markdown("### Relación tickets y actividad")
    st.markdown("""
    No se encontró relación entre la cantidad de tickets de soporte y los días
    transcurridos desde el último ingreso (correlación de Pearson: 0.004).
    Una experiencia técnica negativa no se asocia con alejamiento de la plataforma
    en este dataset.
    """)

st.markdown("### Variables numéricas y PCA")
st.markdown("""
La matriz de correlación mostró coeficientes cercanos a cero entre todas las variables
numéricas, indicando ausencia de relaciones lineales significativas.
El análisis PCA confirmó este resultado: las cuatro componentes explican proporciones
casi idénticas de varianza (≈25% cada una), sin ningún codo en la curva acumulada.
La reducción de dimensionalidad no es efectiva en este dataset porque las variables
describen aspectos independientes del comportamiento del usuario.
El resultado relevante del PCA es la interpretación de los componentes: PC1 captura
una dimensión de actividad e involucramiento, y PC2 una dimensión demográfica
con fricción técnica.
""")

# ── 2. Limitaciones ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 2. Limitaciones")

st.markdown("""
- El alcance de las conclusiones se encuentra condicionado por la información disponible
  y por las decisiones documentadas durante el proceso. El dataset no incluye variables
  de contenido consumido, historial de navegación ni indicadores de satisfacción,
  lo que limita la profundidad del análisis de comportamiento.

- La ausencia de correlaciones lineales entre variables numéricas puede deberse a la
  naturaleza del dataset o a la presencia de relaciones no lineales que el análisis
  descriptivo y PCA no capturan.

- Los nulos en `favorite_genre` fueron etiquetados como `sin_definir` por no poder
  distinguir entre dato no registrado y preferencia indefinida. Esto introduce
  una categoría artificial que puede afectar análisis futuros sobre géneros.

- El dataset no incluye información temporal longitudinal: cada registro es una
  instantánea de un usuario en un momento dado, lo que impide analizar evolución
  del comportamiento en el tiempo.
""")

# ── 3. Próximos pasos ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 3. Próximos pasos")

st.markdown("""
- Incorporar variables adicionales que permitan caracterizar mejor el comportamiento
  del usuario, como historial de contenido consumido o métricas de engagement detalladas.

- Explorar técnicas de análisis no lineal (como t-SNE o UMAP) para identificar
  estructuras en los datos que PCA no puede capturar por su linealidad.

- Aplicar algoritmos de clustering (k-means, clustering jerárquico) sobre las
  variables disponibles para identificar segmentos de usuarios con comportamiento similar,
  lo que podría orientar estrategias de retención diferenciadas por perfil.

- Analizar la variable `favorite_genre` con mayor detalle una vez resuelta
  la ambigüedad de los valores faltantes, incorporando análisis de co-ocurrencia
  entre género y plan de suscripción.
""")

# ── 4. Referencias ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 4. Referencias del proyecto")

col_r1, col_r2 = st.columns(2)
with col_r1:
    st.markdown("""
    **Repositorio GitHub**
    🔗 [Ver repositorio](https://github.com/)

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
st.caption("Taboada, Emiliano Nicolás · Pérez Carletti, Martín · Minería de Datos 1 · 2025")
