import os
import urllib.request
import pandas as pd
import time
from datetime import datetime
import sys

print("🚀 EXO-AI - NASA DATA DOWNLOADER")
print("=" * 50)

# ================================
# CONFIGURACIÓN
# ================================
OUTDIR = "data"
OUTFILE = os.path.join(OUTDIR, "koi.csv")

# URL MEJORADA - Más datos y features específicas
URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+koi+where+koi_disposition+is+not+null&format=csv"

# ================================
# FUNCIÓN DE DESCARGA CON PROGRESO
# ================================
def download_with_progress(url, filename):
    """
    Descarga con barra de progreso visual
    """
    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        bar_length = 30
        filled_length = int(bar_length * percent // 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        sys.stdout.write(f'\r📥 Descargando: |{bar}| {percent:.1f}%')
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filename, progress_callback)
        sys.stdout.write('\n')
        return True
    except Exception as e:
        print(f"\n❌ Error en descarga: {e}")
        return False

# ================================
# VALIDACIÓN DE DATOS DESCARGADOS
# ================================
def validate_dataset(filepath):
    """
    Valida que el dataset tenga las columnas necesarias
    """
    try:
        print("🔍 Validando dataset...")
        df = pd.read_csv(filepath, low_memory=False)
        
        # Columnas críticas que deben existir
        critical_columns = ['koi_disposition', 'koi_period', 'koi_prad']
        missing_columns = [col for col in critical_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Columnas críticas faltantes: {missing_columns}")
            return False
        
        print(f"✅ Dataset válido: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        print(f"🎯 Distribución de clases:")
        if 'koi_disposition' in df.columns:
            dist = df['koi_disposition'].value_counts()
            for cls, count in dist.items():
                print(f"   {cls}: {count:,} ({(count/len(df))*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validando dataset: {e}")
        return False

# ================================
# FUNCIÓN PRINCIPAL MEJORADA
# ================================
def download_koi():
    """
    Descarga y valida la tabla KOI desde NASA Exoplanet Archive
    """
    print("🌍 Conectando con NASA Exoplanet Archive...")
    print(f"📁 Output: {OUTFILE}")
    
    # Crear directorio si no existe
    os.makedirs(OUTDIR, exist_ok=True)
    
    # Verificar si ya existe
    if os.path.exists(OUTFILE):
        print("✅ Dataset ya existe. Validando...")
        if validate_dataset(OUTFILE):
            print("💡 Para forzar re-descarga, borra el archivo:", OUTFILE)
            return OUTFILE
        else:
            print("🔄 Dataset corrupto, procediendo con descarga...")
            os.remove(OUTFILE)
    
    # Descargar dataset
    print("🛰️ Iniciando descarga desde NASA...")
    start_time = time.time()
    
    if not download_with_progress(URL, OUTFILE):
        # FALLBACK: URL alternativa
        print("🔄 Intentando con URL alternativa...")
        ALT_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi"
        if not download_with_progress(ALT_URL, OUTFILE):
            print("❌ No se pudo descargar el dataset.")
            print("💡 Descarga manualmente desde:")
            print("   https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=koi")
            return None
    
    download_time = time.time() - start_time
    print(f"✅ Descarga completada en {download_time:.1f} segundos")
    
    # Validar dataset descargado
    if validate_dataset(OUTFILE):
        print("🎉 DATOS NASA LISTOS PARA ENTRENAMIENTO!")
        
        # Información adicional del dataset
        df = pd.read_csv(OUTFILE, low_memory=False)
        file_size = os.path.getsize(OUTFILE) / (1024 * 1024)  # MB
        print(f"📊 Resumen final:")
        print(f"   • Filas: {df.shape[0]:,}")
        print(f"   • Columnas: {df.shape[1]}")
        print(f"   • Tamaño: {file_size:.2f} MB")
        print(f"   • Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return OUTFILE
    else:
        print("❌ Dataset descargado pero inválido")
        return None

# ================================
# FUNCIÓN PARA DATOS DE PRUEBA
# ================================
def create_sample_data():
    """
    Crea un dataset de prueba pequeño para desarrollo
    """
    sample_file = os.path.join(OUTDIR, "koi_sample.csv")
    
    if not os.path.exists(OUTFILE):
        print("📝 Creando dataset de prueba...")
        # Datos de ejemplo basados en estructura real
        sample_data = {
            'koi_disposition': ['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE'] * 10,
            'koi_period': [365.0, 200.5, 150.2] * 10,
            'koi_prad': [1.0, 2.5, 0.8] * 10,
            'koi_depth': [100.0, 50.5, 75.2] * 10,
            'koi_duration': [12.0, 8.5, 10.2] * 10,
        }
        
        df_sample = pd.DataFrame(sample_data)
        df_sample.to_csv(sample_file, index=False)
        print(f"✅ Dataset de prueba creado: {sample_file}")
        return sample_file
    
    return OUTFILE

# ================================
# EJECUCIÓN PRINCIPAL
# ================================
if __name__ == "__main__":
    print("🚀 INICIANDO DESCARGA DE DATOS NASA...")
    print("=" * 50)
    
    try:
        result_file = download_koi()
        
        if result_file:
            print(f"\n🎉 ¡ÉXITO! Dataset listo en: {result_file}")
            print("💡 Ahora puedes ejecutar:")
            print("   python check_data.py  # Para análisis exploratorio")
            print("   python train.py       # Para entrenar el modelo")
        else:
            print("\n❌ No se pudo obtener dataset válido.")
            print("💡 Creando datos de prueba...")
            test_file = create_sample_data()
            print(f"📝 Usando datos de prueba: {test_file}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Descarga cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        print("💡 Contacta al equipo de desarrollo")
    
    print("\n" + "=" * 50)