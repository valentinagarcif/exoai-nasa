import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIGURACIÓN ESTÉTICA PROFESIONAL
# ================================
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
print("🚀 EXO-AI - NASA SPACE APPS CHALLENGE")
print("=" * 50)

# ================================
# CARGA Y EXPLORACIÓN INICIAL
# ================================
def load_and_explore_data():
    print("📥 Cargando dataset KOI de NASA...")
    df = pd.read_csv("data/koi.csv", low_memory=False)
    
    print(f"📊 DIMENSIONES: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"🎯 MEMORIA USADA: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

# ================================
# ANÁLISIS DE LA VARIABLE TARGET
# ================================
def analyze_target(df):
    print("\n🎯 ANÁLISIS DE LA VARIABLE TARGET")
    print("=" * 40)
    
    if 'koi_disposition' in df.columns:
        target_counts = df['koi_disposition'].value_counts()
        print("Distribución de clases:")
        for cls, count in target_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {cls}: {count:,} ({percentage:.1f}%)")
        
        # Visualización profesional
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        ax1.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Distribución de Clases KOI', fontsize=14, fontweight='bold')
        
        # Bar plot
        sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax2, palette=colors)
        ax2.set_title('Cantidad por Clase', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Cantidad')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print("❌ Variable target 'koi_disposition' no encontrada")

# ================================
# ANÁLISIS DE FEATURES CRÍTICAS
# ================================
def analyze_features(df):
    print("\n🔍 ANÁLISIS DE FEATURES CRÍTICAS")
    print("=" * 40)
    
    # Features astronómicas clave
    key_features = [
        'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
        'koi_depth', 'koi_prad', 'koi_teq', 'koi_srad', 'koi_smass', 'koi_kepmag'
    ]
    
    available_features = [f for f in key_features if f in df.columns]
    print(f"✅ Features disponibles: {len(available_features)}/{len(key_features)}")
    
    # Estadísticas descriptivas
    print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
    stats_df = df[available_features].describe()
    print(stats_df.round(4))
    
    return available_features

# ================================
# ANÁLISIS DE VALORES FALTANTES
# ================================
def analyze_missing_values(df, features):
    print("\n🔍 ANÁLISIS DE VALORES FALTANTES")
    print("=" * 40)
    
    missing_data = df[features].isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Feature': features,
        'Missing_Count': missing_data.values,
        'Missing_Percent': missing_percent.values
    }).sort_values('Missing_Percent', ascending=False)
    
    print("Valores faltantes por feature:")
    for _, row in missing_df.iterrows():
        if row['Missing_Percent'] > 0:
            print(f"  {row['Feature']}: {row['Missing_Count']:,} ({row['Missing_Percent']:.1f}%)")
    
    # Visualización de missing values
    if missing_data.sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        missing_plot_data = missing_df[missing_df['Missing_Percent'] > 0]
        
        if len(missing_plot_data) > 0:
            sns.barplot(data=missing_plot_data, x='Missing_Percent', y='Feature', 
                       palette='Reds_r', ax=ax)
            ax.set_title('Porcentaje de Valores Faltantes por Feature', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Porcentaje Faltante (%)')
            plt.tight_layout()
            plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
            plt.show()

# ================================
# ANÁLISIS DE CORRELACIONES
# ================================
def analyze_correlations(df, features):
    print("\n📊 ANÁLISIS DE CORRELACIONES")
    print("=" * 40)
    
    # Matriz de correlación solo con features numéricas completas
    numeric_features = df[features].select_dtypes(include=[np.number])
    complete_features = numeric_features.columns[numeric_features.isnull().sum() == 0]
    
    if len(complete_features) > 1:
        correlation_matrix = df[complete_features].corr()
        
        # Heatmap profesional
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   annot=True, fmt=".2f", ax=ax)
        
        ax.set_title('Matriz de Correlación - Features de Exoplanetas', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlaciones más fuertes
        print("\n🔗 CORRELACIONES MÁS FUERTES:")
        corr_pairs = correlation_matrix.unstack().sort_values(key=abs, ascending=False)
        unique_pairs = corr_pairs[~corr_pairs.index.duplicated(keep='first')]
        top_correlations = unique_pairs[1:6]  # Excluir autocorrelación
        
        for (feat1, feat2), corr in top_correlations.items():
            print(f"  {feat1} ↔ {feat2}: {corr:.3f}")

# ================================
# ANÁLISIS DE OUTLIERS
# ================================
def analyze_outliers(df, features):
    print("\n📏 ANÁLISIS DE OUTLIERS")
    print("=" * 40)
    
    numeric_features = df[features].select_dtypes(include=[np.number])
    
    for feature in numeric_features.columns[:6]:  # Analizar primeras 6 features
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_percent = (len(outliers) / len(df)) * 100
        
        if outlier_percent > 0:
            print(f"  {feature}: {len(outliers):,} outliers ({outlier_percent:.1f}%)")

# ================================
# DISTRIBUCIONES DE FEATURES
# ================================
def analyze_distributions(df, features):
    print("\n📈 ANÁLISIS DE DISTRIBUCIONES")
    print("=" * 40)
    
    numeric_features = df[features].select_dtypes(include=[np.number])
    
    # Grid de distribuciones
    n_features = min(6, len(numeric_features.columns))
    features_to_plot = numeric_features.columns[:n_features]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(features_to_plot):
        if i < len(axes):
            df[feature].hist(bins=50, ax=axes[i], alpha=0.7, color='skyblue')
            axes[i].set_title(f'Distribución de {feature}', fontweight='bold')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frecuencia')
    
    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# ANÁLISIS POR CLASE
# ================================
def analyze_by_class(df, features):
    if 'koi_disposition' not in df.columns:
        return
        
    print("\n🎯 ANÁLISIS POR CLASE")
    print("=" * 40)
    
    numeric_features = df[features].select_dtypes(include=[np.number])
    
    # Boxplots por clase para primeras 4 features
    n_features = min(4, len(numeric_features.columns))
    features_to_plot = numeric_features.columns[:n_features]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(features_to_plot):
        if i < len(axes):
            sns.boxplot(data=df, x='koi_disposition', y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} por Clase', fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('features_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()

# ================================
# FUNCIÓN PRINCIPAL
# ================================
def main():
    print("🚀 INICIANDO ANÁLISIS COMPLETO DE DATOS NASA KOI")
    print("=" * 60)
    
    # Cargar datos
    df = load_and_explore_data()
    
    # Análisis comprehensivo
    analyze_target(df)
    available_features = analyze_features(df)
    analyze_missing_values(df, available_features)
    analyze_correlations(df, available_features)
    analyze_outliers(df, available_features)
    analyze_distributions(df, available_features)
    analyze_by_class(df, available_features)
    
    print("\n" + "=" * 60)
    print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("📊 Gráficos guardados para la presentación")
    print("🎯 Listo para entrenamiento del modelo")

if __name__ == "__main__":
    main()