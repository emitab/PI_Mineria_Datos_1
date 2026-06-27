import streamlit as st

st.set_page_config(
    page_title="PI — Minería de Datos",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 Proyecto Integrador — Minería de Datos 1")
st.markdown("**Análisis de usuarios de plataforma de streaming**")
st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    ### Contexto

    Este proyecto aplica técnicas de Minería de Datos sobre un dataset de usuarios
    de una plataforma de streaming. El objetivo es construir un análisis reproducible
    que cubra inspección, limpieza, exploración, reducción de dimensionalidad
    e interpretación de resultados con decisiones justificadas en evidencia.

    ### Etapas del proyecto

    | Página | Contenido |
    |---|---|
    | 📁 Dataset | Inspección inicial, calidad de datos y pipeline de limpieza |
    | 📊 EDA | Análisis exploratorio con cinco preguntas de análisis concretas |
    | 🔷 PCA | Escalamiento, reducción de dimensionalidad e interpretación |
    | 📝 Conclusiones | Hallazgos, limitaciones y próximos pasos |
    """)

with col2:
    st.markdown("### Integrantes")
    st.markdown("""
    - Taboada, Emiliano Nicolás
    - Pérez Carletti, Martín
    """)
    st.markdown("### Cursado")
    st.markdown("""
    - **Materia:** Minería de Datos 1
    - **Comisión:** Grupo 4
    """)
    st.markdown("### Repositorio")
    st.markdown("🔗 [Ver en GitHub](https://github.com/)")

st.markdown("---")
st.caption("Proyecto Integrador · Minería de Datos 1 · 2025")
