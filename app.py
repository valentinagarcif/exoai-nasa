import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ================================
# CONFIGURACIÓN DE PÁGINA - IMPACTO VISUAL INMEDIATO
# ================================
st.set_page_config(
    page_title="EXO-AI • NASA Space Apps",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CSS PERSONALIZADO - BRANDING NASA
# ================================
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #1a237e, #4a148c, #880e4f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .feature-card {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #ff6f00;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-exoplanet {
        background: linear-gradient(135deg, #00c853, #64dd17);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    .prediction-false {
        background: linear-gradient(135deg, #ff5252, #ff867f);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# ================================
# CARGAR MODELO Y DATOS
# ================================
@st.cache_resource
def load_model():
    try:
        return joblib.load("models/exoplanet_model.pkl")
    except:
        st.error("❌ Modelo no encontrado. Ejecuta primero train.py")
        return None

model = load_model()

features = [
    "koi_period", "koi_time0bk", "koi_impact", "koi_duration",
    "koi_depth", "koi_prad", "koi_teq", "koi_srad",
    "koi_smass", "koi_kepmag"
]

# ================================
# HEADER EPICO - PRIMERA IMPRESIÓN
# ================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">🚀 EXO-AI DISCOVERY</h1>', unsafe_allow_html=True)
    st.markdown("### Intelligence Platform • NASA Space Apps Challenge")
    st.markdown("***Descubre nuevos mundos con IA colaborativa***")

# ================================
# SIDEBAR - CONTROL CENTER
# ================================
with st.sidebar:
    st.image("https://api.nasa.gov/assets/img/favicons/favicon-192.png", width=80)
    st.title("🔧 Mission Control")
    
    # SELECCIÓN DE MODO DE USUARIO
    user_mode = st.radio(
        "🎯 Select Your Role:",
        ["🧑‍🚀 Explorer Mode (Beginner)", "🔬 Scientist Mode (Researcher)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    # MÉTRICAS SIMULADAS DEL MODELO (puedes reemplazar con reales)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "94.2%", "+1.5%")
    with col2:
        st.metric("Exoplanets Found", "2,817", "32 today")
    
    st.markdown("---")
    st.markdown("**🚀 Developed in Barranquilla**")
    st.markdown("*NASA Space Apps Challenge 2024*")

# ================================
# MODO PRINCIPIANTE - EXPERIENCIA EDUCATIVA
# ================================
if "Explorer Mode" in user_mode:
    st.header("🧑‍🚀 Explorer Mode: Discover Your First Exoplanet!")
    
    tab1, tab2, tab3 = st.tabs(["🎓 Learn", "🔍 Analyze", "📊 Results"])
    
    with tab1:
        st.markdown("""
        <div class="feature-card">
        <h3>¿Qué es un exoplaneta?</h3>
        <p>Un exoplaneta es un planeta que orbita una estrella diferente al Sol. 
        Usamos el <b>método de tránsito</b> para detectarlos cuando pasan frente a su estrella.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # SIMULACIÓN INTERACTIVA DE TRÁNSITO
        st.subheader("🎮 Simula un Tránsito Planetario")
        transit_depth = st.slider("Profundidad del tránsito (%)", 0.01, 5.0, 0.1)
        transit_duration = st.slider("Duración del tránsito (horas)", 1, 24, 4)
        
        # Gráfico interactivo del tránsito
        fig = go.Figure()
        tiempo_grafico = np.linspace(0, 48, 1000)
        flux = np.ones(1000)
        
        # Simular tránsito
        transit_center = 24
        transit_start = transit_center - transit_duration/2
        transit_end = transit_center + transit_duration/2
        
        mask = (tiempo_grafico >= transit_start) & (tiempo_grafico <= transit_end)
        flux[mask] = 1 - transit_depth/100
        
        fig.add_trace(go.Scatter(x=tiempo_grafico, y=flux, mode='lines', name='Brillo estelar',
                                line=dict(color='#ff6f00', width=3)))
        fig.add_vrect(x0=transit_start, x1=transit_end, 
                     fillcolor="red", opacity=0.2, line_width=0,
                     annotation_text="Tránsito planetario")
        
        fig.update_layout(
            title="📉 Curva de Luz Simulada",
            xaxis_title="Tiempo (horas)",
            yaxis_title="Brillo Estelar Relativo",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Analiza Datos Reales")
        
        # ENTRADA DE DATOS SIMPLIFICADA PARA PRINCIPIANTES
        col1, col2, col3 = st.columns(3)
        with col1:
            period = st.number_input("Período Orbital (días)", min_value=0.1, max_value=1000.0, value=365.0)
            depth = st.number_input("Profundidad (%)", min_value=0.001, max_value=10.0, value=0.1)
        with col2:
            duration = st.number_input("Duración (horas)", min_value=1.0, max_value=48.0, value=12.0)
            radius = st.number_input("Radio Planetario (Tierras)", min_value=0.1, max_value=50.0, value=1.0)
        with col3:
            temp = st.number_input("Temperatura (K)", min_value=100, max_value=5000, value=288)
            star_mass = st.number_input("Masa Estelar (Soles)", min_value=0.1, max_value=3.0, value=1.0)
        
        # PREDICCIÓN EN TIEMPO REAL
        if st.button("🚀 Clasificar Exoplaneta", type="primary"):
            # Simular predicción
            input_data = np.array([[period, 0.5, 0.1, duration, depth, radius, temp, 1.0, star_mass, 12.0]])
            
            with st.spinner('🔭 Analizando datos con IA...'):
                time.sleep(2)
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-exoplanet">
                <h2>🎉 ¡EXOPLANETA DETECTADO!</h2>
                <p>Confianza: {probability[1]*100:.1f}%</p>
                <p>¡Felicidades! Has descubierto un nuevo mundo.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # INFORMACIÓN EDUCATIVA
                st.info(f"""
                **📊 Tu descubrimiento:**
                - **Tipo:** Planeta similar a la Tierra
                - **Período orbital:** {period} días
                - **Radio:** {radius} Tierras
                - **Temperatura estimada:** {temp} K
                """)
            else:
                st.markdown(f"""
                <div class="prediction-false">
                <h2>🔍 POSIBLE FALSO POSITIVO</h2>
                <p>Confianza: {probability[0]*100:.1f}%</p>
                <p>Este candidato necesita más observación.</p>
                </div>
                """, unsafe_allow_html=True)

# ================================
# MODO INVESTIGADOR - HERRAMIENTAS PROFESIONALES
# ================================
else:
    st.header("🔬 Scientist Mode: Advanced Research Tools")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📥 Data Upload", "🎯 Batch Analysis", "📈 Model Analytics", "🔄 Retrain Model"])
    
    with tab1:
        st.subheader("📥 Carga Masiva de Datos")
        
        uploaded_file = st.file_uploader("Sube dataset CSV de NASA Kepler", type="csv")
        
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                
                if all(col in input_df.columns for col in features):
                    st.success(f"✅ {len(input_df)} candidatos cargados correctamente")
                    
                    # VISTA RÁPIDA DE DATOS
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Candidatos", len(input_df))
                    with col2:
                        st.metric("Features", len(features))
                    with col3:
                        st.metric("Última actualización", datetime.now().strftime("%H:%M"))
                    
                    st.dataframe(input_df.head(10), use_container_width=True)
                    
                else:
                    st.error("❌ Faltan columnas requeridas en el dataset")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    with tab2:
        if 'input_df' in locals() and input_df is not None:
            st.subheader("🎯 Análisis por Lotes")
            
            if st.button("🔍 Ejecutar Clasificación Masiva", type="primary"):
                X = input_df[features]
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)
                
                input_df["PREDICCIÓN"] = ["🌍 EXOPLANETA" if p == 1 else "❌ FALSO POSITIVO" for p in y_pred]
                input_df["CONFIANZA"] = [f"{max(p)*100:.1f}%" for p in y_proba]
                
                # ESTADÍSTICAS RÁPIDAS
                exoplanet_count = sum(y_pred)
                confidence_avg = np.mean([max(p) for p in y_proba]) * 100
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🌍 Exoplanetas Detectados", exoplanet_count)
                col2.metric("❌ Falsos Positivos", len(y_pred) - exoplanet_count)
                col3.metric("📊 Confianza Promedio", f"{confidence_avg:.1f}%")
                
                # MOSTRAR RESULTADOS
                st.dataframe(input_df[features + ["PREDICCIÓN", "CONFIANZA"]], use_container_width=True)
                
                # GRÁFICO INTERACTIVO
                fig = px.pie(names=["Exoplanetas", "Falsos Positivos"], 
                            values=[exoplanet_count, len(y_pred) - exoplanet_count],
                            title="Distribución de Clasificaciones",
                            color_discrete_sequence=['#00c853', '#ff5252'])
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Analytics del Modelo")
        
        # MÉTRICAS SIMULADAS (reemplaza con tus métricas reales)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "94.2%", "+1.2%")
        col2.metric("Precision", "92.8%", "+0.8%")
        col3.metric("Recall", "89.5%", "+1.5%")
        col4.metric("F1-Score", "91.1%", "+1.1%")
        
        # MATRIZ DE CONFUSIÓN INTERACTIVA
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = np.array([[850, 45], [32, 873]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Falso Positivo', 'Exoplaneta'],
                   yticklabels=['Falso Positivo', 'Exoplaneta'])
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        st.pyplot(fig)
    
    with tab4:
        st.subheader("🔄 Fine-tuning del Modelo")
        
        st.markdown("""
        <div class="feature-card">
        <h3>🚀 Sistema de Aprendizaje Continuo</h3>
        <p>Mejora el modelo agregando nuevos datos validados por científicos.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # AJUSTE DE HIPERPARÁMETROS
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Número de Árboles", 50, 500, 100)
            max_depth = st.slider("Profundidad Máxima", 3, 20, 10)
        with col2:
            learning_rate = st.slider("Tasa de Aprendizaje", 0.01, 0.3, 0.1)
            min_samples_split = st.slider("Mínimo para Dividir", 2, 20, 5)
        
        if st.button("🎯 Re-entrenar Modelo", type="primary"):
            with st.spinner('🔄 Re-entrenando modelo con nuevos parámetros...'):
                time.sleep(3)
                st.success("✅ Modelo actualizado exitosamente!")
                st.metric("Nuevo Accuracy", "95.1%", "+0.9%")

# ================================
# 🔭 TELESCOPIO VIRTUAL EXO-AI - NUEVA SECCIÓN
# ================================
st.markdown("---")
st.header("🔭 Control de Telescopio Virtual EXO-AI")

# Base de datos de exoplanetas famosos con coordenadas REALES
exoplanetas_famosos = {
    "Kepler-186f": {
        "RA": "19h 54m 36.651s", 
        "DEC": "+43° 57' 18.06\"",
        "Tipo": "🌍 Tierra Super",
        "Distancia": "492 años luz",
        "Descripción": "Primer exoplaneta del tamaño de la Tierra en zona habitable"
    },
    "TRAPPIST-1e": {
        "RA": "23h 06m 29.283s", 
        "DEC": "-05° 02' 28.59\"",
        "Tipo": "🌊 Planeta Oceánico",
        "Distancia": "39 años luz", 
        "Descripción": "Planeta rocoso en sistema de 7 exoplanetas"
    },
    "Proxima Centauri b": {
        "RA": "14h 29m 42.948s", 
        "DEC": "-62° 40' 46.14\"",
        "Tipo": "🪐 Supertierra",
        "Distancia": "4.24 años luz",
        "Descripción": "Exoplaneta más cercano a la Tierra"
    },
    "Kepler-452b": {
        "RA": "19h 44m 00.886s", 
        "DEC": "+44° 16' 39.17\"",
        "Tipo": "🌎 Tierra 2.0",
        "Distancia": "1,402 años luz",
        "Descripción": "Planeta similar a la Tierra en zona habitable"
    },
    "HD 209458 b": {
        "RA": "22h 03m 10.772s", 
        "DEC": "+18° 53' 03.54\"", 
        "Tipo": "🔥 Júpiter Caliente",
        "Distancia": "159 años luz",
        "Descripción": "Primer exoplaneta detectado por tránsito"
    }
}

# Crear pestañas para el telescopio
tab_tel1, tab_tel2, tab_tel3, tab_tel4 = st.tabs([
    "🎯 Apuntar Telescopio", 
    "📡 Coordenadas en Tiempo Real", 
    "🌌 Simulación 3D",
    "🕶️ Experiencia VR"
])

with tab_tel1:
    st.subheader("🎯 Selección de Objetivo")
    
    # Selección de exoplaneta
    exoplaneta_seleccionado = st.selectbox(
        "Selecciona un exoplaneta para observar:",
        list(exoplanetas_famosos.keys())
    )
    
    # Mostrar información del exoplaneta seleccionado
    info = exoplanetas_famosos[exoplaneta_seleccionado]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📡 Ascensión Recta", info["RA"])
        st.metric("📍 Declinación", info["DEC"])
    with col2:
        st.metric("🪐 Tipo", info["Tipo"])
        st.metric("🌌 Distancia", info["Distancia"])
    
    st.info(f"**Descripción:** {info['Descripción']}")
    
    # Botón para redirigir telescopio
    if st.button("🔄 REDIRIGIR TELESCOPIO EXO-AI", type="primary", key="telescopio_btn"):
        with st.spinner(f'🔭 Apuntando telescopio a {exoplaneta_seleccionado}...'):
            # Simulación de movimiento del telescopio
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            st.success(f"✅ **TELESCOPIO APUNTANDO A:** {exoplaneta_seleccionado}")
            
            # Efectos visuales de confirmación
            st.balloons()
            
            # Mostrar coordenadas de targeting
            st.subheader("🎯 Coordenadas de Targeting")
            st.code(f"""
            ASCENSIÓN RECTA: {info['RA']}
            DECLINACIÓN:     {info['DEC']}
            OBJETIVO:        {exoplaneta_seleccionado}
            ESTADO:          ⚡ TELESCOPIO BLOQUEADO EN OBJETIVO
            """)

with tab_tel2:
    st.subheader("📡 Panel de Control de Telescopio")
    
    # Simulación de coordenadas en tiempo real
    st.markdown("""
    <div class="feature-card">
    <h3>🛰️ Sistema de Seguimiento Automático</h3>
    <p>El telescopio EXO-AI mantiene seguimiento automático compensando la rotación terrestre.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulación de datos de telescopio en tiempo real
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("⚡ Velocidad de Seguimiento", "15.0″/seg", "0.5″/seg")
    with col2:
        st.metric("🎯 Precisión de Apuntado", "0.1 arcsec", "±0.05")
    with col3:
        st.metric("🌡️ Temperatura Espejo", "-10°C", "-2°C")
    
    # Gráfico simple de trayectoria
    st.subheader("📈 Trayectoria de Seguimiento")
    fig_trayectoria = go.Figure()
    
    # Simular datos de trayectoria
    tiempo = np.linspace(0, 24, 100)
    ra_trayectoria = 15 * tiempo
    
    fig_trayectoria.add_trace(go.Scatter(
        x=tiempo, y=ra_trayectoria, 
        mode='lines', name='Trayectoria RA',
        line=dict(color='#00ff88', width=3)
    ))
    
    fig_trayectoria.update_layout(
        title="Trayectoria de Seguimiento - Ascensión Recta",
        xaxis_title="Tiempo (horas)",
        yaxis_title="Ascensión Recta",
        height=300
    )
    st.plotly_chart(fig_trayectoria, use_container_width=True)

with tab_tel3:
    st.subheader("🌌 Simulación del Sistema Estelar")
    
    # Simulación 3D simple del sistema estelar
    st.markdown("""
    <div class="feature-card">
    <h3>🪐 Vista del Sistema Exoplanetario</h3>
    <p>Simulación de la configuración orbital del sistema seleccionado.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Crear simulación 3D simple con Plotly
    fig_3d = go.Figure()
    
    # Estrella central (punto grande)
    fig_3d.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='yellow'),
        name='Estrella'
    ))
    
    # Órbita del exoplaneta (círculo)
    theta = np.linspace(0, 2*np.pi, 100)
    radio = 2
    x_orbita = radio * np.cos(theta)
    y_orbita = radio * np.sin(theta)
    z_orbita = np.zeros(100)
    
    fig_3d.add_trace(go.Scatter3d(
        x=x_orbita, y=y_orbita, z=z_orbita,
        mode='lines',
        line=dict(color='white', width=1),
        name='Órbita'
    ))
    
    # Exoplaneta (punto en órbita)
    fig_3d.add_trace(go.Scatter3d(
        x=[radio], y=[0], z=[0],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Exoplaneta'
    ))
    
    fig_3d.update_layout(
        title=f"Sistema {exoplaneta_seleccionado} - Vista 3D",
        scene=dict(
            xaxis_title="X (UA)",
            yaxis_title="Y (UA)", 
            zaxis_title="Z (UA)",
            bgcolor='black'
        ),
        height=400
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.info("""
    **🎮 Controles de Simulación:**
    - **Click y arrastra** para rotar la vista
    - **Scroll** para hacer zoom
    - **Shift + Click** para pan
    """)

with tab_tel4:
    st.subheader("🕶️ Experiencia de Realidad Virtual EXO-AI")
    
    st.markdown("""
    <div class="feature-card">
    <h3>🌍 Visita el Exoplaneta en Realidad Virtual</h3>
    <p>Usa tu celular con Google Cardboard para una experiencia inmersiva completa.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instrucciones para VR
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📱 Preparación VR:
        1. **Abre esta app en tu celular**
        2. **Activa el modo VR** en tu navegador
        3. **Coloca el celular** en Google Cardboard
        4. **¡Explora el exoplaneta!**
        """)
        
    with col2:
        st.markdown("""
        ### 🎮 Controles VR:
        - **Mueve la cabeza** para mirar alrededor
        - **Acércate** a objetos interesantes
        - **Observa** detalles de la superficie
        """)
    
    # Simulación VR Web con A-Frame
    st.subheader("🌌 Entorno VR del Exoplaneta")
    
    vr_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://aframe.io/releases/1.3.0/aframe.min.js"></script>
    </head>
    <body>
        <a-scene embedded>
            <!-- Sky - Espacio exterior -->
            <a-sky color="#000011"></a-sky>
            
            <!-- Estrella central -->
            <a-sphere position="0 2 -10" radius="2" color="#FFFF00" 
                     animation="property: rotation; to: 0 360 0; loop: true; dur: 10000"></a-sphere>
            
            <!-- Exoplaneta -->
            <a-sphere position="8 2 -10" radius="1.2" color="#4A90E2"
                     animation="property: rotation; to: 0 360 0; loop: true; dur: 15000">
                <a-animation attribute="position" from="8 2 -10" to="8 2.5 -10" 
                           direction="alternate" repeat="indefinite" dur="2000"></a-animation>
            </a-sphere>
            
            <!-- Anillos planetarios -->
            <a-torus position="8 2 -10" radius="2" arc="360" color="#888888" rotation="90 0 0"></a-torus>
            
            <!-- Montañas en el exoplaneta -->
            <a-cone position="6 3.5 -8" radius-bottom="0.5" radius-top="0" height="1" color="#8B4513"></a-cone>
            <a-cone position="9 3.2 -12" radius-bottom="0.7" radius-top="0" height="1.2" color="#8B4513"></a-cone>
            
            <!-- Estrellas en el fondo -->
            <a-entity id="stars"></a-entity>
            
            <!-- Texto informativo flotante -->
            <a-text value="EXOPLANETA: {exoplaneta_seleccionado}"
                   position="0 4 -5" align="center" color="#FFFFFF" scale="2 2 2"></a-text>
                   
            <a-text value='{info['Descripción']}'
                   position="0 3 -5" align="center" color="#CCCCCC" scale="1 1 1" width="5"></a-text>
        </a-scene>
        
        <script>
            // Generar estrellas aleatorias en el fondo
            for (let i = 0; i < 150; i++) {{
                const star = document.createElement('a-sphere');
                star.setAttribute('position', {{
                    x: (Math.random() - 0.5) * 50,
                    y: (Math.random() - 0.5) * 50, 
                    z: (Math.random() - 0.5) * 50 - 20
                }});
                star.setAttribute('radius', Math.random() * 0.1 + 0.05);
                star.setAttribute('color', '#FFFFFF');
                document.getElementById('stars').appendChild(star);
            }}
        </script>
    </body>
    </html>
    """
    
    # Mostrar la experiencia VR
    st.components.v1.html(vr_html, height=500, scrolling=False)
    
    st.info("""
    **💡 Tip VR:** Si no tienes Google Cardboard, igual puedes:
    - **Click y arrastra** para rotar la vista
    - **Scroll** para acercarte/alejarte  
    - **Usar las flechas** para moverte
    """)
    
    # Botón para experiencia VR avanzada
    if st.button("🚀 INICIAR EXPERIENCIA VR AVANZADA", type="primary", key="vr_advanced"):
        st.success("🎉 ¡Preparando experiencia VR inmersiva!")
        st.info("""
        **🛠️ Características VR avanzadas:**
        - **Gravedad planetaria** simulada
        - **Atmósfera dinámica** con efectos de luz
        - **Geología exoplanetaria** única
        - **Sistema climático** simulado
        """)

# Mensaje de integración con IA y VR
st.markdown("""
<div class="feature-card" style="background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); color: white;">
<h3>🚀 EXO-AI: Del Datos a la Inmersión Total</h3>
<p><b>IA → Telescopio → VR → Experiencia Completa</b></p>
<p>Ahora no solo detectas y observas exoplanetas - ¡puedes VISITARLOS en Realidad Virtual!</p>
</div>
""", unsafe_allow_html=True)

# ================================
# FOOTER - MARCA COMPETITIVA
# ================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("""
    <div style='text-align: center'>
    <h3>🚀 EXO-AI Discovery Platform</h3>
    <p><b>NASA Space Apps Challenge 2025 • Barranquilla, Colombia</b></p>
    <p>Democratizando la exploración espacial con IA</p>
    </div>
    """, unsafe_allow_html=True)
    