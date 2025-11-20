import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
from typing import Dict, Optional

# --- Configuración del Entorno Académico ---
st.set_page_config(
    page_title="Scientometric Production Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos para figuras con estándar de publicación (APA/IEEE friendly)
PLOTLY_TEMPLATE = "plotly_white"
PUBLICATION_FONT = "Times New Roman, sans-serif"

# --- Backend: Lógica Cienciométrica (Clase Modular) ---
class ScientometricAnalyzer:
    """
    Motor de análisis para métricas de diversidad y paridad.
    Implementa algoritmos estándar de bibliometría.
    """
    
    def __init__(self):
        self.guesser = self._load_gender_guesser()
        self.cc = self._load_country_converter()
        self.df = None

    @staticmethod
    @st.cache_resource
    def _load_gender_guesser():
        """Carga eficiente del modelo de inferencia de género."""
        try:
            import gender_guesser.detector as gender
            return gender.Detector(case_sensitive=False)
        except ImportError:
            return None

    @staticmethod
    @st.cache_resource
    def _load_country_converter():
        """Carga de la base de datos de normalización ISO3."""
        try:
            import country_converter as coco
            return coco.CountryConverter()
        except ImportError:
            return None

    def load_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Carga robusta de datos soportando codificaciones comunes en bases de datos (WoS, Scopus).
        """
        if uploaded_file is None:
            return None
        
        # Decodificación segura
        content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        
        # Heurística para detectar delimitadores comunes en exportaciones científicas
        separators = [',', ';', '\t', '|']
        
        for sep in separators:
            try:
                df = pd.read_csv(StringIO(content), sep=sep, engine='python')
                # Validación mínima: debe tener columnas
                if df.shape[1] > 1:
                    self.df = df
                    return df
            except Exception:
                continue
        return None

    def process_attributes(self, name_col: str, country_col: str) -> pd.DataFrame:
        """
        Procesa la normalización de datos crudos.
        """
        if self.df is None:
            raise ValueError("Dataframe is empty.")
        
        processed_df = self.df.copy()

        # 1. Inferencia de Género
        if self.guesser and name_col in processed_df.columns:
            # Tomar solo el primer nombre para la inferencia
            processed_df['clean_name'] = processed_df[name_col].astype(str).str.split().str[0]
            processed_df['Inferred Gender'] = processed_df['clean_name'].apply(self._infer_gender_wrapper)
        else:
            processed_df['Inferred Gender'] = 'Unknown'

        # 2. Estandarización Geográfica (ISO3)
        if self.cc and country_col in processed_df.columns:
            processed_df['Standard Country'] = processed_df[country_col].apply(
                lambda x: self.cc.convert(names=x, to='ISO3', not_found='Unknown') if pd.notna(x) else 'Unknown'
            )
        else:
            processed_df['Standard Country'] = 'Unknown'
            
        return processed_df

    def _infer_gender_wrapper(self, name: str) -> str:
        if pd.isna(name): return 'Unknown'
        res = self.guesser.get_gender(name.lower())
        if 'male' in res: return 'Male'
        if 'female' in res: return 'Female'
        return 'Unknown'

    def calculate_indices(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula índices fundamentados en literatura cienciométrica.
        
        Returns:
            dict: 
                - Collaboration_Diversity: 1 - Simpson's D (Simpson, 1949).
                - Gender_Parity: 1 - |Pm - Pf| (Adaptado de UNESCO, 2020).
        """
        metrics = {}
        
        # A. Simpson's Diversity Index (1 - D) para Colaboración Internacional
        if 'Standard Country' in df.columns:
            # Filtrar desconocidos para no sesgar la diversidad
            valid_countries = df[df['Standard Country'] != 'Unknown']['Standard Country']
            if not valid_countries.empty:
                counts = valid_countries.value_counts(normalize=True)
                simpson_d = (counts ** 2).sum() # D = sum(pi^2)
                metrics['Collaboration_Diversity'] = 1 - simpson_d
            else:
                metrics['Collaboration_Diversity'] = 0.0

        # B. Gender Parity Convergence (Justicia de Género)
        if 'Inferred Gender' in df.columns:
            gender_subset = df[df['Inferred Gender'].isin(['Male', 'Female'])]
            total = len(gender_subset)
            if total > 0:
                p_male = len(gender_subset[gender_subset['Inferred Gender'] == 'Male']) / total
                p_female = len(gender_subset[gender_subset['Inferred Gender'] == 'Female']) / total
                # Fórmula de convergencia absoluta a la paridad
                metrics['Gender_Parity'] = 1 - abs(p_male - p_female)
            else:
                metrics['Gender_Parity'] = 0.0
                
        return metrics

# --- Frontend: Interfaz de Usuario (Streamlit) ---

def main():
    st.title("🔬 Scientometric Analysis Platform")
    st.markdown("""
    **Herramienta de Análisis de Producción Científica.** Permite calcular índices de diversidad y paridad en conjuntos de datos bibliométricos.
    """)
    
    analyzer = ScientometricAnalyzer()

    # Sidebar para configuración
    with st.sidebar:
        st.header("1. Carga de Datos")
        uploaded_file = st.file_uploader("Subir archivo (CSV/TSV)", type=['csv', 'txt'])
        
        st.divider()
        st.header("2. Metodología")
        st.info("""
        **Índice de Diversidad (1-D):** Basado en el Índice de Simpson (1949). Mide la heterogeneidad de las afiliaciones nacionales.
        
        **Justicia de Género (GPI adj):** Convergencia a la paridad. 1.0 indica equilibrio perfecto (50/50).
        """)

    if uploaded_file:
        # Carga inicial
        raw_df = analyzer.load_data(uploaded_file)
        
        if raw_df is not None:
            # Selección de variables
            with st.expander("⚙️ Configuración de Variables", expanded=True):
                cols = raw_df.columns.tolist()
                c1, c2 = st.columns(2)
                name_col = c1.selectbox("Columna: Nombres (Autores)", cols, index=0)
                country_col = c2.selectbox("Columna: Países (Afiliación)", cols, index=min(1, len(cols)-1))
            
            # Ejecución del pipeline
            if st.button("Procesar Datos", type="primary"):
                with st.spinner("Normalizando datos y calculando métricas..."):
                    processed_df = analyzer.process_attributes(name_col, country_col)
                    metrics = analyzer.calculate_indices(processed_df)
                
                st.success("Análisis completado.")
                
                # Sección de Resultados
                st.divider()
                st.subheader("📊 Indicadores Bibliométricos")
                
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    label="Diversidad Internacional (1-D)", 
                    value=f"{metrics.get('Collaboration_Diversity', 0):.3f}",
                    help="0 = Monocultura (un solo país), 1 = Diversidad infinita."
                )
                m2.metric(
                    label="Paridad de Género (GPI)", 
                    value=f"{metrics.get('Gender_Parity', 0):.3f}",
                    help="1.0 = Paridad perfecta. Valores cercanos a 0 indican disparidad extrema."
                )
                m3.metric(
                    label="N (Autores)", 
                    value=len(processed_df)
                )

                # Visualización
                st.subheader("Visualización de Distribución")
                tab1, tab2 = st.tabs(["Género", "Geografía"])
                
                with tab1:
                    if 'Inferred Gender' in processed_df.columns:
                        counts = processed_df['Inferred Gender'].value_counts().reset_index()
                        counts.columns = ['Gender', 'Count']
                        fig = px.bar(
                            counts, x='Gender', y='Count', color='Gender',
                            title="Distribución por Género Inferido",
                            template=PLOTLY_TEMPLATE
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    if 'Standard Country' in processed_df.columns:
                        counts_c = processed_df[processed_df['Standard Country']!='Unknown']['Standard Country'].value_counts().reset_index()
                        counts_c.columns = ['ISO3', 'Count']
                        fig = px.choropleth(
                            counts_c, locations="ISO3", color="Count",
                            title="Mapa de Colaboración Global (ISO3)",
                            template=PLOTLY_TEMPLATE
                        )
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()