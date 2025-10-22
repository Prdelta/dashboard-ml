# 🤖 Dashboard de Aprendizaje Supervisado

## 👨‍🎓 Información del Estudiante

- **Nombre:** Apaza Bruna Luis Wilson
- **Universidad:** Universidad Nacional del Altiplano (UNAP)
- **Facultad:** Facultad de Ingeniería Estadística e Informática
- **Curso:** Aprendizaje Supervisado
- **Modalidad:** Trabajo Individual
- **Unidad:** I

---

## 📋 Descripción del Proyecto

Este dashboard interactivo integra y documenta **6 algoritmos de aprendizaje supervisado** implementados durante la Unidad I del curso de Aprendizaje Supervisado.

## 🎯 Objetivo

Consolidar el aprendizaje práctico mediante la implementación, visualización y análisis de algoritmos de Machine Learning, permitiendo:

- ✅ Exploración interactiva de cada algoritmo
- ✅ Visualización de resultados y métricas
- ✅ Pruebas con datos reales y sintéticos
- ✅ Documentación técnica completa

## 🔧 Algoritmos Implementados

### 1. 📈 Regresión Lineal
Modelado de relaciones lineales entre variables. Ideal para predicción de valores continuos cuando existe una relación lineal entre features y target.

**Características principales:**
- Ecuación: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
- Entrenamiento rápido
- Alta interpretabilidad
- Sensible a outliers

### 2. 🎯 Regresión Logística
Clasificación binaria y multiclase basada en probabilidades.

**Características principales:**
- Función sigmoide para probabilidades
- Regularización L1/L2 disponible
- Interpretable mediante coeficientes
- Eficiente para clasificación

### 3. 🔒 Ridge y Lasso
Regresión con regularización para prevenir overfitting.

**Ridge (L2):**
- Penaliza suma de cuadrados de coeficientes
- Reduce magnitud pero no elimina features
- Útil con multicolinealidad

**Lasso (L1):**
- Penaliza valor absoluto de coeficientes
- Puede reducir coeficientes a cero
- Selección automática de features

### 4. 🌳 Árboles de Regresión CART
Modelos basados en decisiones jerárquicas.

**Características principales:**
- No requiere normalización
- Maneja datos no lineales
- Alta interpretabilidad
- Propenso a overfitting sin poda

### 5. 👥 K-Nearest Neighbors (KNN)
Predicción basada en vecinos más cercanos.

**Características principales:**
- Lazy learning (sin fase de entrenamiento)
- Requiere normalización obligatoria
- Sensible al valor de k
- Predicción puede ser lenta

### 6. 🧠 Redes Neuronales
Deep learning para capturar patrones complejos.

**Características principales:**
- Múltiples capas ocultas
- Funciones de activación no lineales
- Captura relaciones complejas
- Requiere más datos y tiempo

## 💻 Tecnologías Utilizadas

### Lenguaje y Framework Principal
- **Python 3.8+** - Lenguaje de programación
- **Streamlit 1.28+** - Framework para aplicaciones web interactivas

### Librerías de Machine Learning
- **Scikit-learn 1.3+** - Algoritmos de ML y métricas
- **NumPy 1.24+** - Computación numérica
- **Pandas 2.0+** - Manipulación de datos

### Visualización
- **Matplotlib 3.7+** - Gráficos estáticos
- **Seaborn 0.12+** - Visualización estadística

## 📦 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Paso 1: Clonar o Descargar el Proyecto
```bash
# Si tienes git instalado
git clone <url-del-repositorio>
cd dashboard-ml

# O simplemente descarga el archivo .py
```

### Paso 2: Crear un Entorno Virtual (Recomendado)
```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias
```bash
pip install streamlit numpy pandas matplotlib seaborn scikit-learn
```

O usando un archivo requirements.txt:
```bash
pip install -r requirements.txt
```

## 🚀 Cómo Usar

### Ejecutar la Aplicación
```bash
streamlit run dashboard_ml.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Navegación

1. **Página de Inicio (🏠):**
   - Descripción general del proyecto
   - Comparación de algoritmos
   - Instrucciones de uso

2. **Seleccionar un Algoritmo:**
   - Usa el menú lateral para elegir
   - Cada algoritmo tiene su propia página

3. **Configurar Parámetros:**
   - Ajusta sliders y selectores
   - Configura datos de prueba
   - Modifica hiperparámetros

4. **Entrenar Modelo:**
   - Haz clic en el botón "Entrenar"
   - Observa el proceso en tiempo real

5. **Analizar Resultados:**
   - Revisa métricas de evaluación
   - Explora visualizaciones interactivas
   - Compara rendimiento

## 📊 Estructura de Cada Algoritmo

Cada sección de algoritmo incluye:

### 📖 Documentación Técnica
- **Entradas Requeridas:** Formato de datos, tipos, restricciones
- **Salidas Esperadas:** Predicciones, métricas, visualizaciones
- **Parámetros del Modelo:** Hiperparámetros disponibles

### 🎯 Contexto de Aplicación
- **Casos de Uso:** Situaciones reales de aplicación
- **Criterios de Validación:** Cómo evaluar el modelo
- **Ventajas y Limitaciones:** Cuándo usar o no usar

### ⚙️ Configuración Interactiva
- Controles deslizantes para parámetros
- Generación de datos sintéticos
- División train/test personalizable

### 📈 Resultados y Visualizaciones
- Métricas de evaluación (MSE, RMSE, R², MAE, Accuracy)
- Gráficos de predicciones vs valores reales
- Análisis de residuos
- Matrices de confusión (clasificación)
- Importancia de features

## 📁 Estructura del Proyecto

```
dashboard-ml/
│
├── dashboard_ml.py          # Script principal de Streamlit
├── README.md                # Este archivo
├── requirements.txt         # Dependencias del proyecto
│
└── docs/                    # Documentación adicional (opcional)
    ├── imagenes/
    └── ejemplos/
```

## 📋 Características del Dashboard

### ✅ Especificaciones Técnicas
- Documentación completa de entradas y salidas
- Explicación de parámetros
- Requerimientos de preprocesamiento

### ✅ Visualizaciones Interactivas
- Gráficos dinámicos con Matplotlib
- Análisis de resultados en tiempo real
- Comparación de modelos

### ✅ Casos de Prueba
- Datos sintéticos generados automáticamente
- Configuración personalizable
- Ejemplos con datasets reales

### ✅ Métricas de Evaluación
- MSE, RMSE, R², MAE para regresión
- Accuracy, Precision, Recall para clasificación
- Matrices de confusión
- Análisis de residuos

## 🎓 Casos de Uso por Algoritmo

### Regresión Lineal
- Predicción de precios de viviendas
- Estimación de ventas
- Análisis de tendencias

### Regresión Logística
- Detección de spam
- Diagnóstico médico
- Aprobación de créditos

### Ridge y Lasso
- Datos con alta dimensionalidad
- Multicolinealidad
- Selección automática de features

### Árboles CART
- Datos no lineales
- Necesidad de interpretabilidad
- Datos mixtos (numéricos y categóricos)

### KNN
- Sistemas de recomendación
- Predicción de precios inmobiliarios
- Detección de anomalías

### Redes Neuronales
- Relaciones muy complejas
- Alta dimensionalidad
- Series temporales

## 🔍 Métricas de Evaluación

### Para Regresión
- **MSE (Mean Squared Error):** Promedio de errores al cuadrado
- **RMSE (Root Mean Squared Error):** Raíz cuadrada del MSE
- **R² (R-squared):** Proporción de varianza explicada (0-1)
- **MAE (Mean Absolute Error):** Promedio de errores absolutos

### Para Clasificación
- **Accuracy:** Porcentaje de predicciones correctas
- **Precision:** Proporción de positivos correctos
- **Recall:** Proporción de positivos identificados
- **F1-Score:** Media armónica de precision y recall
- **Matriz de Confusión:** Análisis detallado de errores

## ⚠️ Consideraciones Importantes

### Preprocesamiento
- **KNN y Redes Neuronales:** Requieren normalización obligatoria
- **Árboles CART:** No requieren normalización
- **Ridge/Lasso:** Benefician de normalización

### Overfitting
- Monitorear diferencia entre métricas de train y test
- Usar validación cruzada
- Ajustar regularización cuando sea posible

### Selección de Modelos
- Comenzar con modelos simples (Regresión Lineal)
- Aumentar complejidad si es necesario
- Considerar interpretabilidad vs performance

## 🐛 Solución de Problemas

### Error: Module not found
```bash
pip install <nombre-del-modulo>
```

### La aplicación no se abre
- Verificar que el puerto 8501 esté libre
- Usar otro puerto: `streamlit run dashboard_ml.py --server.port 8502`

### Gráficos no se muestran
- Actualizar Matplotlib: `pip install --upgrade matplotlib`

### Errores de convergencia en Redes Neuronales
- Aumentar `max_iter`
- Ajustar `learning_rate_init`
- Verificar normalización de datos

## 📚 Recursos Adicionales

### Documentación Oficial
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://docs.streamlit.io/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)

### Tutoriales Recomendados
- Machine Learning Mastery
- Towards Data Science
- Real Python

## 👤 Información del Trabajo

- **Modalidad:** Individual
- **Curso:** Aprendizaje Supervisado - Unidad I
- **Objetivo:** Consolidación de conocimientos mediante implementación práctica
- **Entrega:** Última semana de la Unidad I

## 📝 Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

## 🤝 Contribuciones

Este es un proyecto educativo. Sugerencias y mejoras son bienvenidas.

## 📧 Contacto

Para dudas o sugerencias sobre el proyecto, contactar al desarrollador.

---

**Última actualización:** 2025

**Versión:** 1.0.0

**Estado:** ✅ Completo y Funcional