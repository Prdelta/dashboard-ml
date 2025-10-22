import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.datasets import make_regression, make_classification
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Dashboard Aprendizaje Supervisado - UNAP",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INFORMACIÓN DEL ESTUDIANTE
# ============================================================================
ESTUDIANTE_INFO = {
    "nombre": "Apaza Bruna Luis Wilson",
    "universidad": "Universidad Nacional del Altiplano",
    "facultad": "Facultad de Ingeniería Estadística e Informática",
    "curso": "Aprendizaje Supervisado",
    "email": "luis.apaza@unap.edu.pe"  # Ajusta si tienes otro email
}

# Estilos CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# TÍTULO Y DOCUMENTACIÓN INICIAL
# ============================================================================
st.markdown('<h1 class="main-header">🤖 Dashboard de Aprendizaje Supervisado</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Implementación, Visualización y Análisis de Algoritmos de Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# README / DOCUMENTACIÓN DEL PROYECTO
# ============================================================================
with st.expander("📚 README - Documentación del Proyecto", expanded=False):
    st.markdown("""
    ## 📋 Descripción del Proyecto
    
    Este dashboard interactivo integra y documenta **6 algoritmos de aprendizaje supervisado** 
    implementados durante la Unidad I del curso de Aprendizaje Supervisado.
    
    ### 🎯 Objetivo
    Consolidar el aprendizaje práctico mediante la implementación, visualización y análisis 
    de algoritmos de Machine Learning, permitiendo:
    - Exploración interactiva de cada algoritmo
    - Visualización de resultados y métricas
    - Pruebas con datos reales y sintéticos
    - Documentación técnica completa
    
    ### 🔧 Algoritmos Implementados
    
    1. **Regresión Lineal** - Modelado de relaciones lineales
    2. **Regresión Logística** - Clasificación binaria y multiclase
    3. **Regresión Ridge y Lasso** - Regularización L1 y L2
    4. **Árboles de Regresión CART** - Modelos basados en decisiones
    5. **K-Nearest Neighbors (KNN)** - Predicción por vecinos cercanos
    6. **Redes Neuronales** - Deep learning para regresión
    
    ### 💻 Tecnologías Utilizadas
    - **Python 3.x** - Lenguaje de programación
    - **Streamlit** - Framework para aplicaciones web
    - **Scikit-learn** - Librería de Machine Learning
    - **Pandas & NumPy** - Procesamiento de datos
    - **Matplotlib & Seaborn** - Visualización
    
    ### 📊 Características
    - ✅ Especificación técnica de entradas y salidas
    - ✅ Visualizaciones interactivas
    - ✅ Casos de prueba con datos reales
    - ✅ Métricas de evaluación
    - ✅ Documentación de cada algoritmo
    
    ### 👤 Información del Trabajo
    - **Modalidad:** Individual
    - **Curso:** Aprendizaje Supervisado - Unidad I
    - **Entrega:** Última semana de la Unidad I
    
    ---
    *Navegue por el menú lateral para explorar cada algoritmo*
    """)

# ============================================================================
# SIDEBAR - NAVEGACIÓN
# ============================================================================
st.sidebar.title("🎯 Navegación")
st.sidebar.markdown("---")

menu_option = st.sidebar.radio(
    "Seleccione un Algoritmo:",
    [
        "🏠 Inicio",
        "📈 1. Regresión Lineal",
        "🎯 2. Regresión Logística", 
        "🔒 3. Ridge & Lasso",
        "🌳 4. Árboles CART",
        "👥 5. KNN Regression",
        "🧠 6. Redes Neuronales"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Cada algoritmo incluye documentación técnica, visualizaciones y ejemplos prácticos.")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def generar_datos_regresion(n_samples=100, n_features=1, noise=10):
    """Genera datos sintéticos para regresión"""
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    return X, y

def generar_datos_clasificacion(n_samples=100, n_features=2, n_classes=2):
    """Genera datos sintéticos para clasificación"""
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_classes=n_classes, n_informative=n_features,
                               n_redundant=0, random_state=42)
    return X, y

def mostrar_metricas_regresion(y_true, y_pred):
    """Muestra métricas de regresión"""
    col1, col2, col3 = st.columns(3)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    with col1:
        st.metric("MSE", f"{mse:.4f}")
        st.metric("RMSE", f"{rmse:.4f}")
    with col2:
        st.metric("R²", f"{r2:.4f}")
        st.metric("MAE", f"{mae:.4f}")
    with col3:
        st.markdown(f"""
        **Interpretación:**
        - R² = {r2:.2%} de varianza explicada
        - Error promedio: {mae:.2f}
        """)

def plot_regression_results(X, y_true, y_pred, title="Resultados de Regresión"):
    """Visualiza resultados de regresión"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if X.shape[1] == 1:
        ax.scatter(X, y_true, alpha=0.6, label='Datos Reales', s=50)
        sorted_idx = X.flatten().argsort()
        ax.plot(X[sorted_idx], y_pred[sorted_idx], 'r-', linewidth=2, label='Predicción')
    else:
        ax.scatter(range(len(y_true)), y_true, alpha=0.6, label='Datos Reales', s=50)
        ax.scatter(range(len(y_pred)), y_pred, alpha=0.6, label='Predicción', s=50)
    
    ax.set_xlabel('Variable X')
    ax.set_ylabel('Variable Y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# ============================================================================
# PÁGINA DE INICIO
# ============================================================================
if menu_option == "🏠 Inicio":
    st.header("🏠 Bienvenido al Dashboard de Aprendizaje Supervisado")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎓 Sobre este Dashboard
        
        Este proyecto es una implementación completa de algoritmos de **aprendizaje supervisado** 
        que cumple con todos los requisitos del trabajo parcial de la Unidad I.
        
        #### ✨ Funcionalidades Principales:
        
        - **Documentación Completa**: Cada algoritmo incluye especificaciones técnicas detalladas
        - **Visualizaciones Interactivas**: Gráficos dinámicos para entender el comportamiento
        - **Pruebas en Tiempo Real**: Experimente con diferentes parámetros y datos
        - **Métricas de Evaluación**: Análisis cuantitativo del rendimiento
        - **Ejemplos Prácticos**: Casos de uso reales y sintéticos
        
        #### 🚀 Cómo Usar:
        
        1. **Seleccione un algoritmo** del menú lateral
        2. **Configure los parámetros** según sus necesidades
        3. **Ejecute el modelo** con el botón correspondiente
        4. **Analice los resultados** con las visualizaciones y métricas
        
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Algoritmos
        
        1. 📈 Regresión Lineal
        2. 🎯 Regresión Logística
        3. 🔒 Ridge & Lasso
        4. 🌳 Árboles CART
        5. 👥 KNN Regression
        6. 🧠 Redes Neuronales
        
        ---
        
        ### 📈 Estadísticas
        
        - **6** Algoritmos
        - **100%** Funcional
        - **Documentado** ✓
        - **Probado** ✓
        """)
    
    st.markdown("---")
    
    # Tabla comparativa
    st.subheader("📊 Comparación Rápida de Algoritmos")
    
    comparison_df = pd.DataFrame({
        'Algoritmo': ['Regresión Lineal', 'Regresión Logística', 'Ridge/Lasso', 
                     'CART', 'KNN', 'Redes Neuronales'],
        'Tipo': ['Regresión', 'Clasificación', 'Regresión', 'Regresión', 'Regresión', 'Regresión'],
        'Complejidad': ['Baja', 'Media', 'Media', 'Media', 'Baja', 'Alta'],
        'Interpretabilidad': ['Alta', 'Alta', 'Alta', 'Alta', 'Media', 'Baja'],
        'Mejor Para': ['Relaciones lineales', 'Clasificación binaria', 'Muchas features', 
                      'Datos no lineales', 'Patrones locales', 'Datos complejos']
    })
    
    st.dataframe(comparison_df, use_container_width=True)

# ============================================================================
# 1. REGRESIÓN LINEAL
# ============================================================================
elif menu_option == "📈 1. Regresión Lineal":
    st.header("📈 Regresión Lineal")
    
    # Documentación técnica
    with st.expander("📖 Documentación Técnica", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📥 Entradas Requeridas
            
            **Tipo de Datos:**
            - **X (Features):** Array NumPy o DataFrame de Pandas
              - Formato: (n_samples, n_features)
              - Tipo: float o int
              - Restricciones: Sin valores nulos
            
            - **y (Target):** Array 1D
              - Formato: (n_samples,)
              - Tipo: float (valores continuos)
              - Restricciones: Variable numérica
            
            **Parámetros del Modelo:**
            - `fit_intercept`: bool (True/False)
            - `normalize`: bool (deprecated, usar StandardScaler)
            
            **Preprocesamiento:**
            - Escalado de features (opcional pero recomendado)
            - Manejo de outliers
            - División train/test
            """)
        
        with col2:
            st.markdown("""
            ### 📤 Salidas Esperadas
            
            **Parámetros del Modelo:**
            - **Coeficientes (weights):** Array de pesos
            - **Intercepto (bias):** Valor escalar
            
            **Predicciones:**
            - **y_pred:** Valores continuos predichos
            - Mismo formato que y original
            
            **Métricas de Evaluación:**
            - **MSE:** Error cuadrático medio
            - **RMSE:** Raíz del MSE
            - **R²:** Coeficiente de determinación (0-1)
            - **MAE:** Error absoluto medio
            
            **Visualizaciones:**
            - Gráfico de dispersión con línea de regresión
            - Residuos del modelo
            - Comparación valores reales vs predichos
            """)
    
    with st.expander("🎯 Contexto de Aplicación"):
        st.markdown("""
        ### 📊 Casos de Uso
        
        1. **Predicción de Precios**
           - Precios de viviendas según características
           - Valoración de vehículos usados
           - Estimación de costos de proyectos
        
        2. **Análisis Financiero**
           - Predicción de ventas futuras
           - Estimación de ingresos
           - Análisis de tendencias económicas
        
        3. **Ciencias e Ingeniería**
           - Modelado de fenómenos físicos
           - Relaciones entre variables experimentales
           - Calibración de instrumentos
        
        ### ✅ Criterios de Validación
        
        - **R² > 0.7:** Modelo con buen ajuste
        - **RMSE:** Debe ser menor que la desviación estándar de y
        - **Residuos:** Deben distribuirse normalmente
        - **No multicolinealidad:** VIF < 5 para cada feature
        """)
    
    st.markdown("---")
    st.subheader("⚙️ Configuración del Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples = st.slider("Número de muestras", 50, 500, 100)
        noise = st.slider("Nivel de ruido", 0, 50, 10)
    
    with col2:
        n_features = st.selectbox("Número de features", [1, 2, 3, 5])
        test_size = st.slider("Tamaño de test (%)", 10, 40, 20) / 100
    
    with col3:
        fit_intercept = st.checkbox("Incluir intercepto", value=True)
        normalize_data = st.checkbox("Normalizar datos", value=True)
    
    if st.button("🚀 Entrenar Modelo de Regresión Lineal", type="primary"):
        with st.spinner("Entrenando modelo..."):
            # Generar datos
            X, y = generar_datos_regresion(n_samples, n_features, noise)
            
            # División train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Normalización
            if normalize_data:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Entrenar modelo
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            st.success("✅ Modelo entrenado exitosamente!")
            
            # Mostrar parámetros del modelo
            st.markdown("### 🔍 Parámetros del Modelo")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Coeficientes:**")
                coef_df = pd.DataFrame({
                    'Feature': [f'X{i+1}' for i in range(n_features)],
                    'Coeficiente': model.coef_
                })
                st.dataframe(coef_df, use_container_width=True)
            
            with col2:
                st.markdown("**Intercepto:**")
                st.code(f"bias = {model.intercept_:.4f}")
                
                st.markdown("**Ecuación del Modelo:**")
                equation = f"y = {model.intercept_:.2f}"
                for i, coef in enumerate(model.coef_):
                    equation += f" + ({coef:.2f})·X{i+1}"
                st.code(equation)
            
            # Métricas
            st.markdown("### 📊 Métricas de Evaluación")
            
            tab1, tab2 = st.tabs(["🎓 Conjunto de Entrenamiento", "🧪 Conjunto de Prueba"])
            
            with tab1:
                st.markdown("#### Métricas en Entrenamiento")
                mostrar_metricas_regresion(y_train, y_pred_train)
            
            with tab2:
                st.markdown("#### Métricas en Prueba")
                mostrar_metricas_regresion(y_test, y_pred_test)
            
            # Visualizaciones
            st.markdown("### 📈 Visualizaciones")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = plot_regression_results(X_test, y_test, y_pred_test, 
                                              "Predicciones vs Valores Reales")
                st.pyplot(fig1)
            
            with col2:
                # Gráfico de residuos
                residuals = y_test - y_pred_test
                fig2, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_pred_test, residuals, alpha=0.6)
                ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
                ax.set_xlabel('Valores Predichos')
                ax.set_ylabel('Residuos')
                ax.set_title('Gráfico de Residuos')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig2)

# ============================================================================
# 2. REGRESIÓN LOGÍSTICA
# ============================================================================
elif menu_option == "🎯 2. Regresión Logística":
    st.header("🎯 Regresión Logística")
    
    with st.expander("📖 Documentación Técnica", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📥 Entradas Requeridas
            
            **Tipo de Datos:**
            - **X (Features):** Array NumPy o DataFrame
              - Formato: (n_samples, n_features)
              - Tipo: float o int
              - Características numéricas o categóricas codificadas
            
            - **y (Target):** Array 1D de clases
              - Formato: (n_samples,)
              - Tipo: int o string
              - Clases: Binaria (0/1) o Multiclase
            
            **Parámetros del Modelo:**
            - `penalty`: 'l1', 'l2', 'elasticnet', None
            - `C`: float, inverso de la regularización
            - `solver`: 'lbfgs', 'liblinear', 'saga'
            - `max_iter`: int, número máximo de iteraciones
            - `multi_class`: 'ovr', 'multinomial'
            """)
        
        with col2:
            st.markdown("""
            ### 📤 Salidas Esperadas
            
            **Probabilidades:**
            - **Probabilidades de clase:** Array (n_samples, n_classes)
            - Valores entre 0 y 1
            - Suma = 1 para cada muestra
            
            **Predicciones:**
            - **Clase predicha:** Array 1D
            - Clase con mayor probabilidad
            
            **Métricas de Clasificación:**
            - **Accuracy:** Precisión global
            - **Precision:** Precisión por clase
            - **Recall:** Sensibilidad
            - **F1-Score:** Media armónica
            - **Matriz de Confusión**
            - **ROC-AUC:** Área bajo la curva ROC
            """)
    
    with st.expander("🎯 Contexto de Aplicación"):
        st.markdown("""
        ### 📊 Casos de Uso
        
        1. **Detección de Spam**
           - Clasificar correos como spam o no spam
           - Análisis de contenido textual
        
        2. **Diagnóstico Médico**
           - Predecir presencia/ausencia de enfermedad
           - Clasificación de riesgo (bajo/medio/alto)
        
        3. **Aprobación de Créditos**
           - Decidir si aprobar o rechazar solicitud
           - Evaluación de riesgo crediticio
        
        4. **Marketing**
           - Predecir si un cliente comprará
           - Segmentación de clientes
        
        ### ✅ Criterios de Validación
        
        - **Accuracy > 80%:** para problemas balanceados
        - **F1-Score:** Importante en datos desbalanceados
        - **ROC-AUC > 0.8:** Buen poder discriminativo
        - **Matriz de Confusión:** Análisis de errores
        """)
    
    st.markdown("---")
    st.subheader("⚙️ Configuración del Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples_clf = st.slider("Número de muestras", 50, 500, 200, key="clf_samples")
        n_features_clf = st.slider("Número de features", 2, 10, 2, key="clf_features")
    
    with col2:
        n_classes = st.selectbox("Número de clases", [2, 3, 4])
        test_size_clf = st.slider("Tamaño de test (%)", 10, 40, 20, key="clf_test") / 100
    
    with col3:
        penalty = st.selectbox("Regularización", ['l2', 'l1', 'none'])
        C_value = st.slider("Parámetro C", 0.01, 10.0, 1.0)
    
    if st.button("🚀 Entrenar Modelo de Regresión Logística", type="primary"):
        with st.spinner("Entrenando modelo..."):
            # Generar datos
            X, y = generar_datos_clasificacion(n_samples_clf, n_features_clf, n_classes)
            
            # División train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_clf, random_state=42, stratify=y
            )
            
            # Normalización
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Entrenar modelo
            solver = 'saga' if penalty == 'l1' else 'lbfgs'
            penalty_param = None if penalty == 'none' else penalty
            
            model = LogisticRegression(
                penalty=penalty_param,
                C=C_value,
                solver=solver,
                max_iter=1000,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test)
            
            st.success("✅ Modelo entrenado exitosamente!")
            
            # Métricas
            st.markdown("### 📊 Métricas de Evaluación")
            
            col1, col2, col3, col4 = st.columns(4)
            
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            
            with col1:
                st.metric("Accuracy (Train)", f"{acc_train:.2%}")
            with col2:
                st.metric("Accuracy (Test)", f"{acc_test:.2%}")
            with col3:
                st.metric("Número de Clases", n_classes)
            with col4:
                st.metric("Muestras de Test", len(y_test))
            
            # Matriz de Confusión
            st.markdown("### 🎯 Matriz de Confusión")
            
            cm = confusion_matrix(y_test, y_pred_test)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Valor Real')
            ax.set_title('Matriz de Confusión')
            st.pyplot(fig)
            
            # Probabilidades predichas
            if n_classes == 2 and n_features_clf == 2:
                st.markdown("### 📈 Visualización de Decisión")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Crear mesh
                h = 0.02
                x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
                y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))
                
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
                scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                                   cmap='RdYlBu', edgecolors='black', s=100)
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                ax.set_title('Frontera de Decisión')
                plt.colorbar(scatter, ax=ax)
                
                st.pyplot(fig)

# ============================================================================
# 3. RIDGE & LASSO
# ============================================================================
elif menu_option == "🔒 3. Ridge & Lasso":
    st.header("🔒 Regresión Ridge y Lasso")
    
    with st.expander("📖 Documentación Técnica", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📥 Entradas Requeridas
            
            **Tipo de Datos:**
            - **X (Features):** Array o DataFrame
              - Formato: (n_samples, n_features)
              - Puede tener muchas features (alta dimensionalidad)
              - Features pueden estar correlacionadas
            
            - **y (Target):** Array 1D continuo
              - Formato: (n_samples,)
              - Variable numérica a predecir
            
            **Parámetros Principales:**
            - **alpha:** float > 0
              - Fuerza de regularización
              - Mayor alpha = más regularización
            
            **Tipo de Regularización:**
            - **Ridge (L2):** Penaliza suma de cuadrados
            - **Lasso (L1):** Penaliza valor absoluto
            - Lasso puede hacer coeficientes = 0
            """)
        
        with col2:
            st.markdown("""
            ### 📤 Salidas Esperadas
            
            **Coeficientes Regularizados:**
            - Pesos ajustados con penalización
            - Ridge: Coeficientes reducidos pero no cero
            - Lasso: Algunos coeficientes = 0 (selección automática)
            
            **Predicciones:**
            - Valores continuos predichos
            - Mejor generalización que regresión lineal
            
            **Métricas:**
            - MSE, RMSE, R², MAE
            - Comparación con modelo sin regularización
            - Análisis de coeficientes
            
            **Visualizaciones:**
            - Path de coeficientes vs alpha
            - Comparación Ridge vs Lasso
            - Importancia de features (Lasso)
            """)
    
    with st.expander("🎯 Contexto de Aplicación"):
        st.markdown("""
        ### 📊 Casos de Uso
        
        1. **Alta Dimensionalidad**
           - Muchas features vs pocas muestras
           - Datos genómicos o texto
           - Prevención de overfitting
        
        2. **Multicolinealidad**
           - Features altamente correlacionadas
           - Ridge estabiliza los coeficientes
        
        3. **Selección de Variables**
           - Lasso para feature selection automática
           - Identificar variables relevantes
        
        ### ✅ Criterios de Validación
        
        - **Validación Cruzada:** Para seleccionar alpha óptimo
        - **Comparar Alphas:** Probar diferentes valores
        - **Ridge vs Lasso:** Comparar ambos métodos
        - **Análisis de Coeficientes:** Verificar reducción o eliminación
        """)
    
    st.markdown("---")
    st.subheader("⚙️ Configuración del Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples_reg = st.slider("Número de muestras", 50, 500, 100, key="ridge_samples")
        n_features_reg = st.slider("Número de features", 5, 50, 20, key="ridge_features")
    
    with col2:
        noise_reg = st.slider("Nivel de ruido", 0, 50, 20, key="ridge_noise")
        test_size_reg = st.slider("Tamaño de test (%)", 10, 40, 20, key="ridge_test") / 100
    
    with col3:
        alpha_ridge = st.slider("Alpha Ridge", 0.01, 10.0, 1.0, key="alpha_ridge")
        alpha_lasso = st.slider("Alpha Lasso", 0.01, 10.0, 1.0, key="alpha_lasso")
    
    if st.button("🚀 Entrenar Modelos Ridge y Lasso", type="primary"):
        with st.spinner("Entrenando modelos..."):
            # Generar datos con muchas features
            X, y = generar_datos_regresion(n_samples_reg, n_features_reg, noise_reg)
            
            # División train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_reg, random_state=42
            )
            
            # Normalización
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entrenar modelos
            model_linear = LinearRegression()
            model_ridge = Ridge(alpha=alpha_ridge)
            model_lasso = Lasso(alpha=alpha_lasso)
            
            model_linear.fit(X_train_scaled, y_train)
            model_ridge.fit(X_train_scaled, y_train)
            model_lasso.fit(X_train_scaled, y_train)
            
            # Predicciones
            y_pred_linear = model_linear.predict(X_test_scaled)
            y_pred_ridge = model_ridge.predict(X_test_scaled)
            y_pred_lasso = model_lasso.predict(X_test_scaled)
            
            st.success("✅ Modelos entrenados exitosamente!")
            
            # Comparación de métricas
            st.markdown("### 📊 Comparación de Modelos")
            
            metrics_df = pd.DataFrame({
                'Modelo': ['Lineal', 'Ridge', 'Lasso'],
                'MSE': [
                    mean_squared_error(y_test, y_pred_linear),
                    mean_squared_error(y_test, y_pred_ridge),
                    mean_squared_error(y_test, y_pred_lasso)
                ],
                'RMSE': [
                    np.sqrt(mean_squared_error(y_test, y_pred_linear)),
                    np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
                    np.sqrt(mean_squared_error(y_test, y_pred_lasso))
                ],
                'R²': [
                    r2_score(y_test, y_pred_linear),
                    r2_score(y_test, y_pred_ridge),
                    r2_score(y_test, y_pred_lasso)
                ],
                'Features != 0': [
                    n_features_reg,
                    n_features_reg,
                    np.sum(model_lasso.coef_ != 0)
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Visualización de coeficientes
            st.markdown("### 🔍 Comparación de Coeficientes")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].bar(range(n_features_reg), model_linear.coef_, alpha=0.7)
            axes[0].set_title('Regresión Lineal')
            axes[0].set_xlabel('Feature')
            axes[0].set_ylabel('Coeficiente')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].bar(range(n_features_reg), model_ridge.coef_, alpha=0.7, color='orange')
            axes[1].set_title(f'Ridge (α={alpha_ridge})')
            axes[1].set_xlabel('Feature')
            axes[1].set_ylabel('Coeficiente')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].bar(range(n_features_reg), model_lasso.coef_, alpha=0.7, color='green')
            axes[2].set_title(f'Lasso (α={alpha_lasso})')
            axes[2].set_xlabel('Feature')
            axes[2].set_ylabel('Coeficiente')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info(f"💡 **Lasso** redujo a cero {n_features_reg - np.sum(model_lasso.coef_ != 0)} features de {n_features_reg}")

# ============================================================================
# 4. ÁRBOLES DE REGRESIÓN CART
# ============================================================================
elif menu_option == "🌳 4. Árboles CART":
    st.header("🌳 Árboles de Regresión CART")
    
    with st.expander("📖 Documentación Técnica", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📥 Entradas Requeridas
            
            **Tipo de Datos:**
            - **X (Features):** Array o DataFrame
              - Puede contener features numéricas y categóricas
              - No requiere normalización
              - Maneja valores faltantes
            
            - **y (Target):** Array 1D continuo
              - Variable a predecir (regresión)
            
            **Parámetros del Árbol:**
            - `max_depth`: Profundidad máxima del árbol
            - `min_samples_split`: Mínimo de muestras para dividir
            - `min_samples_leaf`: Mínimo de muestras en hoja
            - `max_features`: Features a considerar en cada split
            - `criterion`: 'squared_error', 'absolute_error'
            
            **Criterios de División:**
            - Reducción de varianza (MSE)
            - Ganancia de información
            """)
        
        with col2:
            st.markdown("""
            ### 📤 Salidas Esperadas
            
            **Estructura del Árbol:**
            - Nodos de decisión con reglas
            - Hojas con valores predichos
            - Visualización del árbol completo
            
            **Predicciones:**
            - Valores continuos (media en cada hoja)
            - Path de decisión para cada muestra
            
            **Importancia de Features:**
            - Ranking de variables más importantes
            - Basado en reducción de impureza
            
            **Métricas:**
            - MSE, RMSE, R², MAE
            - Profundidad del árbol
            - Número de hojas
            
            **Visualizaciones:**
            - Árbol de decisión completo
            - Importancia de features
            - Predicciones vs reales
            """)
    
    with st.expander("🎯 Contexto de Aplicación"):
        st.markdown("""
        ### 📊 Casos de Uso
        
        1. **Datos No Lineales**
           - Relaciones complejas entre variables
           - Interacciones automáticas
        
        2. **Interpretabilidad**
           - Fácil de explicar las decisiones
           - Visualización intuitiva
        
        3. **Datos Mixtos**
           - Combina features numéricas y categóricas
           - No requiere preprocesamiento extensivo
        
        4. **Detección de Umbrales**
           - Identifica puntos de corte importantes
           - Segmentación automática
        
        ### ✅ Criterios de Validación
        
        - **Profundidad Óptima:** Evitar overfitting
        - **Poda del Árbol:** Simplificar sin perder precisión
        - **Validación Cruzada:** Estabilidad del modelo
        - **Importancia de Variables:** Análisis de features relevantes
        """)
    
    st.markdown("---")
    st.subheader("⚙️ Configuración del Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples_tree = st.slider("Número de muestras", 50, 500, 150, key="tree_samples")
        n_features_tree = st.slider("Número de features", 2, 10, 4, key="tree_features")
    
    with col2:
        max_depth = st.slider("Profundidad máxima", 2, 20, 5, key="max_depth")
        min_samples_split = st.slider("Min muestras para dividir", 2, 20, 10, key="min_split")
    
    with col3:
        min_samples_leaf = st.slider("Min muestras en hoja", 1, 10, 5, key="min_leaf")
        test_size_tree = st.slider("Tamaño de test (%)", 10, 40, 20, key="tree_test") / 100
    
    if st.button("🚀 Entrenar Árbol de Regresión CART", type="primary"):
        with st.spinner("Entrenando modelo..."):
            # Generar datos
            X, y = generar_datos_regresion(n_samples_tree, n_features_tree, noise=15)
            
            # División train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_tree, random_state=42
            )
            
            # Entrenar modelo
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            st.success("✅ Árbol entrenado exitosamente!")
            
            # Información del árbol
            st.markdown("### 🌳 Información del Árbol")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Profundidad", model.get_depth())
            with col2:
                st.metric("Número de Hojas", model.get_n_leaves())
            with col3:
                st.metric("Features Usados", np.sum(model.feature_importances_ > 0))
            with col4:
                st.metric("Nodos Totales", model.tree_.node_count)
            
            # Métricas
            st.markdown("### 📊 Métricas de Evaluación")
            
            tab1, tab2 = st.tabs(["🎓 Entrenamiento", "🧪 Prueba"])
            
            with tab1:
                mostrar_metricas_regresion(y_train, y_pred_train)
            
            with tab2:
                mostrar_metricas_regresion(y_test, y_pred_test)
            
            # Visualización del árbol
            st.markdown("### 🎨 Visualización del Árbol")
            
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(model, 
                     feature_names=[f'X{i+1}' for i in range(n_features_tree)],
                     filled=True, 
                     rounded=True,
                     ax=ax,
                     fontsize=10)
            ax.set_title('Árbol de Regresión CART', fontsize=16, fontweight='bold')
            st.pyplot(fig)
            
            # Importancia de features
            st.markdown("### 📊 Importancia de Features")
            
            importance_df = pd.DataFrame({
                'Feature': [f'X{i+1}' for i in range(n_features_tree)],
                'Importancia': model.feature_importances_
            }).sort_values('Importancia', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importancia'], color='forestgreen')
            ax.set_xlabel('Importancia')
            ax.set_title('Importancia de Features en el Árbol')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.dataframe(importance_df, use_container_width=True)

# ============================================================================
# 5. K-NEAREST NEIGHBORS REGRESSION
# ============================================================================
elif menu_option == "👥 5. KNN Regression":
    st.header("👥 K-Nearest Neighbors Regression")
    
    with st.expander("📖 Documentación Técnica", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📥 Entradas Requeridas
            
            **Tipo de Datos:**
            - **X (Features):** Array o DataFrame numérico
              - Formato: (n_samples, n_features)
              - **IMPORTANTE:** Requiere normalización/escalado
              - Sensible a la escala de las variables
            
            - **y (Target):** Array 1D continuo
              - Variable a predecir
            
            **Parámetros del Modelo:**
            - `n_neighbors` (k): Número de vecinos (ímpar preferible)
            - `weights`: 'uniform' o 'distance'
              - uniform: todos los vecinos pesan igual
              - distance: vecinos cercanos pesan más
            - `metric`: Métrica de distancia
              - 'euclidean', 'manhattan', 'minkowski'
            - `algorithm`: 'auto', 'ball_tree', 'kd_tree', 'brute'
            
            **Preprocesamiento Crítico:**
            - Escalado obligatorio (StandardScaler o MinMaxScaler)
            - Manejo de outliers
            """)
        
        with col2:
            st.markdown("""
            ### 📤 Salidas Esperadas
            
            **Predicciones:**
            - Promedio (o ponderado) de k vecinos cercanos
            - Sin modelo explícito (lazy learning)
            
            **Para cada predicción:**
            - Índices de los k vecinos más cercanos
            - Distancias a cada vecino
            - Valores de los vecinos
            
            **Métricas:**
            - MSE, RMSE, R², MAE
            - Tiempo de predicción (importante)
            
            **Visualizaciones:**
            - Vecinos cercanos en espacio 2D
            - Superficies de decisión
            - Efecto de k en las predicciones
            - Comparación de diferentes valores de k
            
            **Características:**
            - No hay fase de entrenamiento (memoriza datos)
            - Predicción puede ser lenta con muchos datos
            """)
    
    with st.expander("🎯 Contexto de Aplicación"):
        st.markdown("""
        ### 📊 Casos de Uso
        
        1. **Sistemas de Recomendación**
           - Encontrar usuarios/productos similares
           - Basado en patrones locales
        
        2. **Predicción de Precios Inmobiliarios**
           - Casas similares en ubicación y características
           - Patrones geográficos locales
        
        3. **Interpolación de Datos**
           - Completar valores faltantes
           - Estimaciones basadas en vecindad
        
        4. **Detección de Anomalías**
           - Puntos alejados de sus vecinos
        
        ### ✅ Criterios de Validación
        
        - **Selección de k:** Validación cruzada para k óptimo
        - **Escalado:** Verificar que todas las features estén normalizadas
        - **Curse of Dimensionality:** Cuidado con muchas features
        - **Tiempo de Predicción:** Considerar eficiencia con datos grandes
        - **k pequeño:** Más flexible pero sensible al ruido
        - **k grande:** Más suave pero puede subajustar
        """)
    
    st.markdown("---")
    st.subheader("⚙️ Configuración del Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples_knn = st.slider("Número de muestras", 50, 500, 150, key="knn_samples")
        n_features_knn = st.selectbox("Número de features", [1, 2, 3, 5], key="knn_features")
    
    with col2:
        k_neighbors = st.slider("Número de vecinos (k)", 1, 20, 5, key="k_neighbors")
        weights = st.selectbox("Pesos", ['uniform', 'distance'], key="weights")
    
    with col3:
        metric = st.selectbox("Métrica de distancia", ['euclidean', 'manhattan', 'minkowski'], key="metric")
        test_size_knn = st.slider("Tamaño de test (%)", 10, 40, 20, key="knn_test") / 100
    
    if st.button("🚀 Entrenar Modelo KNN Regression", type="primary"):
        with st.spinner("Entrenando modelo..."):
            # Generar datos
            X, y = generar_datos_regresion(n_samples_knn, n_features_knn, noise=12)
            
            # División train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_knn, random_state=42
            )
            
            # Escalado OBLIGATORIO para KNN
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entrenar modelo
            model = KNeighborsRegressor(
                n_neighbors=k_neighbors,
                weights=weights,
                metric=metric
            )
            model.fit(X_train_scaled, y_train)
            
            # Predicciones
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            st.success("✅ Modelo KNN entrenado exitosamente!")
            
            # Información del modelo
            st.markdown("### 🔍 Información del Modelo")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Vecinos (k)", k_neighbors)
                st.metric("Muestras Train", len(X_train))
            with col2:
                st.metric("Tipo de Pesos", weights)
                st.metric("Muestras Test", len(X_test))
            with col3:
                st.metric("Métrica", metric)
                st.metric("Features", n_features_knn)
            
            # Métricas
            st.markdown("### 📊 Métricas de Evaluación")
            
            tab1, tab2 = st.tabs(["🎓 Entrenamiento", "🧪 Prueba"])
            
            with tab1:
                mostrar_metricas_regresion(y_train, y_pred_train)
                if r2_score(y_train, y_pred_train) > 0.95:
                    st.warning("⚠️ R² muy alto en entrenamiento. Posible sobreajuste con k muy pequeño.")
            
            with tab2:
                mostrar_metricas_regresion(y_test, y_pred_test)
            
            # Visualización para datos 1D o 2D
            if n_features_knn <= 2:
                st.markdown("### 📈 Visualización de Predicciones")
                
                if n_features_knn == 1:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Puntos de entrenamiento
                    ax.scatter(X_train_scaled, y_train, alpha=0.6, s=100, 
                             label='Datos de Entrenamiento', color='blue', edgecolors='black')
                    
                    # Puntos de prueba
                    ax.scatter(X_test_scaled, y_test, alpha=0.6, s=100,
                             label='Datos Reales (Test)', color='green', edgecolors='black')
                    
                    # Predicciones
                    ax.scatter(X_test_scaled, y_pred_test, alpha=0.8, s=100,
                             label='Predicciones KNN', color='red', marker='X', edgecolors='black')
                    
                    ax.set_xlabel('Feature X (Escalado)')
                    ax.set_ylabel('Target Y')
                    ax.set_title(f'KNN Regression (k={k_neighbors})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
                elif n_features_knn == 2:
                    fig = plt.figure(figsize=(15, 5))
                    
                    # Subplot 1: Datos de entrenamiento
                    ax1 = fig.add_subplot(131, projection='3d')
                    scatter1 = ax1.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], y_train,
                                          c=y_train, cmap='viridis', s=50, alpha=0.6)
                    ax1.set_xlabel('Feature 1')
                    ax1.set_ylabel('Feature 2')
                    ax1.set_zlabel('Target')
                    ax1.set_title('Datos de Entrenamiento')
                    plt.colorbar(scatter1, ax=ax1)
                    
                    # Subplot 2: Datos de prueba reales
                    ax2 = fig.add_subplot(132, projection='3d')
                    scatter2 = ax2.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], y_test,
                                          c=y_test, cmap='viridis', s=50, alpha=0.6)
                    ax2.set_xlabel('Feature 1')
                    ax2.set_ylabel('Feature 2')
                    ax2.set_zlabel('Target')
                    ax2.set_title('Datos Reales (Test)')
                    plt.colorbar(scatter2, ax=ax2)
                    
                    # Subplot 3: Predicciones
                    ax3 = fig.add_subplot(133, projection='3d')
                    scatter3 = ax3.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], y_pred_test,
                                          c=y_pred_test, cmap='plasma', s=50, alpha=0.6, marker='^')
                    ax3.set_xlabel('Feature 1')
                    ax3.set_ylabel('Feature 2')
                    ax3.set_zlabel('Predicción')
                    ax3.set_title(f'Predicciones KNN (k={k_neighbors})')
                    plt.colorbar(scatter3, ax=ax3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Análisis de sensibilidad a k
            st.markdown("### 📊 Análisis de Sensibilidad a k")
            
            k_values = range(1, min(21, len(X_train)))
            train_scores = []
            test_scores = []
            
            for k in k_values:
                knn_temp = KNeighborsRegressor(n_neighbors=k, weights=weights, metric=metric)
                knn_temp.fit(X_train_scaled, y_train)
                train_scores.append(r2_score(y_train, knn_temp.predict(X_train_scaled)))
                test_scores.append(r2_score(y_test, knn_temp.predict(X_test_scaled)))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k_values, train_scores, marker='o', label='R² Entrenamiento', linewidth=2)
            ax.plot(k_values, test_scores, marker='s', label='R² Prueba', linewidth=2)
            ax.axvline(x=k_neighbors, color='red', linestyle='--', label=f'k actual = {k_neighbors}')
            ax.set_xlabel('Número de Vecinos (k)')
            ax.set_ylabel('R² Score')
            ax.set_title('Efecto del Número de Vecinos en el Rendimiento')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            best_k = k_values[np.argmax(test_scores)]
            st.info(f"💡 **k óptimo según R² en test:** {best_k} (R² = {max(test_scores):.4f})")

# ============================================================================
# 6. REDES NEURONALES PARA REGRESIÓN
# ============================================================================
elif menu_option == "🧠 6. Redes Neuronales":
    st.header("🧠 Redes Neuronales para Regresión")
    
    with st.expander("📖 Documentación Técnica", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📥 Entradas Requeridas
            
            **Tipo de Datos:**
            - **X (Features):** Array o DataFrame numérico
              - Formato: (n_samples, n_features)
              - Requiere normalización/escalado
              - Puede manejar alta dimensionalidad
            
            - **y (Target):** Array 1D continuo
              - Variable a predecir
              - Recomendable escalar también
            
            **Arquitectura de la Red:**
            - `hidden_layer_sizes`: Tupla con neuronas por capa
              - Ejemplo: (100,) = 1 capa con 100 neuronas
              - Ejemplo: (100, 50) = 2 capas con 100 y 50 neuronas
            
            **Funciones de Activación:**
            - `activation`: 'relu', 'tanh', 'logistic'
              - relu: Rectified Linear Unit (más común)
              - tanh: Tangente hiperbólica
              - logistic: Función sigmoide
            
            **Optimización:**
            - `solver`: 'adam', 'sgd', 'lbfgs'
            - `learning_rate_init`: Tasa de aprendizaje inicial
            - `max_iter`: Número máximo de iteraciones
            - `alpha`: Parámetro de regularización L2
            """)
        
        with col2:
            st.markdown("""
            ### 📤 Salidas Esperadas
            
            **Durante el Entrenamiento:**
            - Curva de pérdida (loss) por época
            - Evolución del error
            - Convergencia del modelo
            
            **Parámetros Aprendidos:**
            - Pesos de cada capa (matrices W)
            - Sesgos de cada capa (vectores b)
            - Número total de parámetros
            
            **Predicciones:**
            - Valores continuos predichos
            - Capacidad de capturar patrones no lineales complejos
            
            **Métricas:**
            - MSE, RMSE, R², MAE
            - Número de iteraciones hasta convergencia
            - Valor final de la función de pérdida
            
            **Visualizaciones:**
            - Arquitectura de la red
            - Curva de aprendizaje
            - Predicciones vs valores reales
            - Evolución del error
            """)
    
    with st.expander("🎯 Contexto de Aplicación"):
        st.markdown("""
        ### 📊 Casos de Uso
        
        1. **Relaciones No Lineales Complejas**
           - Patrones difíciles de modelar con otros algoritmos
           - Múltiples interacciones entre features
        
        2. **Datos de Alta Dimensionalidad**
           - Muchas features con relaciones complejas
           - Aprendizaje de representaciones
        
        3. **Series Temporales**
           - Predicción de valores futuros
           - Detección de patrones temporales
        
        4. **Visión por Computadora**
           - Regresión en imágenes
           - Estimación de poses
        
        ### ✅ Criterios de Validación
        
        - **Convergencia:** Verificar que el modelo converja
        - **Overfitting:** Monitorear diferencia train/test
        - **Arquitectura:** Probar diferentes configuraciones
        - **Early Stopping:** Detener si no mejora
        """)
    
    st.markdown("---")
    st.subheader("⚙️ Configuración del Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples_nn = st.slider("Número de muestras", 100, 1000, 300, key="nn_samples")
        n_features_nn = st.slider("Número de features", 2, 20, 5, key="nn_features")
    
    with col2:
        hidden_layer_1 = st.slider("Neuronas capa 1", 10, 200, 100, key="hidden1")
        hidden_layer_2 = st.slider("Neuronas capa 2", 0, 100, 50, key="hidden2")
    
    with col3:
        activation = st.selectbox("Función activación", ['relu', 'tanh', 'logistic'])
        learning_rate = st.slider("Learning rate", 0.0001, 0.01, 0.001, format="%.4f")
    
    max_iter = st.slider("Máximo de iteraciones", 100, 2000, 500)
    alpha = st.slider("Regularización (alpha)", 0.0001, 0.1, 0.0001, format="%.4f")
    test_size_nn = st.slider("Tamaño de test (%)", 10, 40, 20, key="nn_test") / 100
    
    if st.button("🚀 Entrenar Red Neuronal", type="primary"):
        with st.spinner("Entrenando red neuronal... Esto puede tomar unos segundos..."):
            # Generar datos
            X, y = generar_datos_regresion(n_samples_nn, n_features_nn, noise=15)
            
            # División train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_nn, random_state=42
            )
            
            # Normalización OBLIGATORIA
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            
            # Configurar arquitectura
            if hidden_layer_2 > 0:
                hidden_layers = (hidden_layer_1, hidden_layer_2)
            else:
                hidden_layers = (hidden_layer_1,)
            
            # Entrenar modelo
            model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                solver='adam',
                alpha=alpha,
                learning_rate_init=learning_rate,
                max_iter=max_iter,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            model.fit(X_train_scaled, y_train_scaled)
            
            # Predicciones (desescalar)
            y_pred_train_scaled = model.predict(X_train_scaled)
            y_pred_test_scaled = model.predict(X_test_scaled)
            
            y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
            y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
            
            st.success("✅ Red neuronal entrenada exitosamente!")
            
            # Información de la red
            st.markdown("### 🧠 Arquitectura de la Red")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Capas Ocultas", len(hidden_layers))
                st.metric("Neuronas Totales", sum(hidden_layers))
            with col2:
                st.metric("Iteraciones", model.n_iter_)
                st.metric("Features", n_features_nn)
            with col3:
                st.metric("Activación", activation)
                st.metric("Convergió", "✅" if model.n_iter_ < max_iter else "⚠️")
            with col4:
                total_params = sum([w.size for w in model.coefs_]) + sum([b.size for b in model.intercepts_])
                st.metric("Parámetros", total_params)
                st.metric("Loss Final", f"{model.loss_:.4f}")
            
            # Arquitectura visual
            st.markdown("### 🎨 Visualización de la Arquitectura")
            
            architecture_text = f"""
            **Estructura de la Red:**
            
            - Input Layer: {n_features_nn} neuronas
            """
            for i, neurons in enumerate(hidden_layers, 1):
                architecture_text += f"\n- Hidden Layer {i}: {neurons} neuronas ({activation})"
            architecture_text += f"\n- Output Layer: 1 neurona (linear)"
            
            st.info(architecture_text)
            
            # Métricas
            st.markdown("### 📊 Métricas de Evaluación")
            
            tab1, tab2 = st.tabs(["🎓 Entrenamiento", "🧪 Prueba"])
            
            with tab1:
                mostrar_metricas_regresion(y_train, y_pred_train)
            
            with tab2:
                mostrar_metricas_regresion(y_test, y_pred_test)
            
            # Curva de aprendizaje
            st.markdown("### 📉 Curva de Aprendizaje")
            
            if hasattr(model, 'loss_curve_'):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(model.loss_curve_, linewidth=2, color='blue')
                ax.set_xlabel('Iteración')
                ax.set_ylabel('Loss')
                ax.set_title('Evolución de la Función de Pérdida')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                st.pyplot(fig)
                
                if model.n_iter_ >= max_iter:
                    st.warning("⚠️ La red alcanzó el máximo de iteraciones. Considera aumentar `max_iter`.")
                else:
                    st.success(f"✅ La red convergió en {model.n_iter_} iteraciones.")
            
            # Visualización de predicciones
            st.markdown("### 📈 Predicciones vs Valores Reales")
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Subplot 1: Scatter plot
            axes[0].scatter(y_test, y_pred_test, alpha=0.6, s=50)
            axes[0].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'r--', linewidth=2, label='Predicción Perfecta')
            axes[0].set_xlabel('Valores Reales')
            axes[0].set_ylabel('Predicciones')
            axes[0].set_title('Predicciones vs Valores Reales')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Subplot 2: Residuos
            residuals = y_test - y_pred_test
            axes[1].scatter(y_pred_test, residuals, alpha=0.6, s=50)
            axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[1].set_xlabel('Predicciones')
            axes[1].set_ylabel('Residuos')
            axes[1].set_title('Análisis de Residuos')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Análisis de pesos
            st.markdown("### ⚖️ Análisis de Pesos")
            
            st.info(f"""
            **Información de los Pesos:**
            - Primera capa: {model.coefs_[0].shape[0]} entradas → {model.coefs_[0].shape[1]} neuronas
            - Total de conexiones en capa 1: {model.coefs_[0].size}
            - Rango de pesos en capa 1: [{model.coefs_[0].min():.4f}, {model.coefs_[0].max():.4f}]
            """)
            
            # Distribución de pesos de la primera capa
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(model.coefs_[0].flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.set_xlabel('Valor del Peso')
            ax.set_ylabel('Frecuencia')
            ax.set_title('Distribución de Pesos en la Primera Capa')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# ============================================================================
# PIE DE PÁGINA
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Dashboard de Aprendizaje Supervisado</strong></p>
    <p>Curso: Aprendizaje Supervisado - Unidad I</p>
    <p>Implementación completa de 6 algoritmos de Machine Learning</p>
    <p style='font-size: 0.8em; margin-top: 10px;'>
        Desarrollado con ❤️ usando Streamlit, Scikit-learn, y Python
    </p>
</div>
""", unsafe_allow_html=True)