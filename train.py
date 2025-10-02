import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("🚀 EXO-AI - ENTRENAMIENTO DE MODELO NASA")
print("=" * 50)

# -----------------------------
# 1. CARGAR Y VERIFICAR DATOS
# -----------------------------
print("📥 Cargando dataset Kepler...")
try:
    df = pd.read_csv("data/koi.csv", low_memory=False)
    print(f"✅ Dataset cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
except FileNotFoundError:
    print("❌ Error: No se encuentra data/koi.csv")
    print("💡 Ejecuta primero download_data.py")
    exit()

# -----------------------------
# 2. DEFINIR FEATURES ASTRONÓMICAS CLAVE
# -----------------------------
# Basado en investigación científica real de NASA
KEY_FEATURES = [
    'koi_period',           # Período orbital (días)
    'koi_time0bk',          # Tiempo de tránsito de referencia
    'koi_impact',           # Parámetro de impacto
    'koi_duration',         # Duración del tránsito (horas)
    'koi_depth',            # Profundidad del tránsito (ppm)
    'koi_prad',             # Radio planetario (radios terrestres)
    'koi_teq',              # Temperatura de equilibrio (K)
    'koi_srad',             # Radio estelar (radios solares)
    'koi_smass',            # Masa estelar (masas solares)
    'koi_kepmag'            # Magnitud Kepler
]

# Verificar features disponibles
available_features = [f for f in KEY_FEATURES if f in df.columns]
print(f"🔍 FEATURES DISPONIBLES: {len(available_features)}/{len(KEY_FEATURES)}")

if len(available_features) < 5:
    print("❌ Muy pocas features disponibles. Revisa el dataset.")
    exit()

# -----------------------------
# 3. PREPARAR DATOS
# -----------------------------
print("\n🎯 PREPARANDO DATOS...")

# Filtrar solo filas con target y features disponibles
df_clean = df[available_features + ['koi_disposition']].dropna()

print(f"📊 Datos después de limpieza: {df_clean.shape[0]:,} filas")
print("Distribución de clases:")
class_dist = df_clean['koi_disposition'].value_counts()
for cls, count in class_dist.items():
    print(f"  {cls}: {count:,} ({(count/len(df_clean))*100:.1f}%)")

# Separar features y target
X = df_clean[available_features]
y = df_clean['koi_disposition']

# Codificar target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Guardar el label encoder
joblib.dump(label_encoder, "label_encoder.pkl")
print(f"🎯 Clases codificadas: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# -----------------------------
# 4. MANEJAR DESBALANCE DE CLASES
# -----------------------------
print("\n⚖️ CALCULANDO PESOS PARA CLASES DESBALANCEADAS...")
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))
print("📊 Pesos de clases:", class_weight_dict)

# -----------------------------
# 5. DIVIDIR DATOS
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y_encoded  # ¡IMPORTANTE! Mantener proporción de clases
)

print(f"📚 Datos de entrenamiento: {X_train.shape[0]:,} filas")
print(f"🧪 Datos de prueba: {X_test.shape[0]:,} filas")

# -----------------------------
# 6. ESCALAR DATOS
# -----------------------------
print("\n📐 ESCALANDO FEATURES...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# 7. OPTIMIZACIÓN DE HIPERPARÁMETROS
# -----------------------------
print("\n🎯 OPTIMIZANDO HIPERPARÁMETROS...")

# Modelo base con class weights
base_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight=class_weight_dict,
    n_jobs=-1  # Usar todos los cores
)

# Entrenar modelo base rápido para tener algo funcional
print("⚡ Entrenando modelo base...")
base_model.fit(X_train_scaled, y_train)

# Predicción inicial
y_pred_base = base_model.predict(X_test_scaled)
accuracy_base = accuracy_score(y_test, y_pred_base)
f1_base = f1_score(y_test, y_pred_base, average='weighted')

print(f"📊 Modelo Base - Accuracy: {accuracy_base:.4f}, F1-Score: {f1_base:.4f}")

# -----------------------------
# 8. ENTRENAR MODELO FINAL
# -----------------------------
print("\n🚀 ENTRENANDO MODELO FINAL...")

# Modelo final optimizado
final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_scaled, y_train)

# -----------------------------
# 9. EVALUACIÓN COMPLETA
# -----------------------------
print("\n📈 EVALUANDO MODELO FINAL...")

# Predicciones
y_pred = final_model.predict(X_test_scaled)
y_pred_proba = final_model.predict_proba(X_test_scaled)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print("🎯 MÉTRICAS DEL MODELO:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   F1-Score: {f1:.4f}")

# Reporte de clasificación detallado
print("\n📊 REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# -----------------------------
# 10. ANÁLISIS DE FEATURE IMPORTANCE
# -----------------------------
print("\n🔍 ANALIZANDO FEATURE IMPORTANCE...")

feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("📊 FEATURES MÁS IMPORTANTES:")
for _, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Guardar feature importance
feature_importance.to_csv('feature_importance.csv', index=False)

# Visualización
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
plt.title('Feature Importance - Exoplanet Classification')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 11. MATRIZ DE CONFUSIÓN
# -----------------------------
print("\n📋 GENERANDO MATRIZ DE CONFUSIÓN...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión - Modelo Exoplanet')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 12. GUARDAR MODELO Y METADATOS
# -----------------------------
print("\n💾 GUARDANDO MODELO Y METADATOS...")

# Guardar modelo
joblib.dump(final_model, "exoplanet_model.pkl")

# Guardar features usadas
with open("features.json", "w") as f:
    json.dump(available_features, f)

# Guardar métricas
metrics = {
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'n_features': len(available_features),
    'n_samples': len(df_clean),
    'class_distribution': class_dist.to_dict(),
    'feature_importance': feature_importance.to_dict('records')
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
print(f"🎯 Accuracy final: {accuracy:.4f}")
print(f"📊 F1-Score: {f1:.4f}")
print(f"🔍 Features usadas: {len(available_features)}")
print(f"📚 Muestras de entrenamiento: {X_train.shape[0]:,}")

print("\n🚀 MODELO LISTO PARA LA APLICACIÓN!")