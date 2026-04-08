import os
import logging
import threading
import numpy as np
import cv2
from flask import Flask
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from PIL import Image, ImageChops, ImageEnhance, ImageStat

# --- CONFIGURACIÓN ---
TOKEN = "8699029540:AAE9TGMSC5fvW2Fldhuc_keYQAYxM_ooW_s"

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# --- WEB SERVER FALSO PARA RENDER (Mantiene el servicio vivo) ---
web_app = Flask(__name__)
@web_app.route('/')
def health_check(): 
    return "Validador Vivo", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host='0.0.0.0', port=port)

# --- LÓGICA FORENSE AVANZADA ---
def analyze_image(image_path):
    # 1. Convertir a escala de grises para análisis de textura
    img_cv = cv2.imread(image_path, 0)
    
    # 2. Análisis de Ruido Local (Buscamos zonas "lisas" sospechosas)
    # Calculamos la desviación estándar local en bloques de 15x15
    local_std = cv2.blur(img_cv, (15,15))
    _, thr = cv2.threshold(local_std, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    diff_textura = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    noise_score = np.mean(diff_textura) # Un score alto indica muchas zonas lisas/parches

    # 3. Bordes Canny (Detecta texto con nitidez artificial)
    edges = cv2.Canny(img_cv, 100, 200)
    edge_score = np.mean(edges)

    return noise_score, edge_score

def generate_dictamen(noise_score, edge_score):
    # --- UMBRALES CALIBRADOS (Subidos para evitar falsos positivos) ---
    NOISE_THRESHOLD_HIGH = 25.0 # Antes 15.0
    EDGE_THRESHOLD_HIGH = 18.0  # Antes 10.0
    
    risk_points = 0
    detalles = []

    # Detección de parches digitales (Zonas demasiado lisas)
    if noise_score > NOISE_THRESHOLD_HIGH:
        risk_points += 3
        detalles.append("- 🔴 **ANOMALÍA DE TEXTURA**: Se detectaron zonas artificialmente lisas (posible parche digital).")
    elif noise_score > 15.0:
        risk_points += 1
        detalles.append("- 🟡 Textura del papel ligeramente inconsistente.")

    # Bordes sospechosos (Texto demasiado perfecto)
    if edge_score > EDGE_THRESHOLD_HIGH:
        risk_points += 2
        detalles.append("- 🔴 **NITIDEZ EXTREMA**: Los bordes del texto son demasiado perfectos para una foto física (posible texto insertado).")

    # Veredicto final
    if risk_points >= 3:
        return "🔴 **RIESGO CRÍTICO DETECTADO**", "\n".join(detalles)
    elif risk_points >= 1:
        return "🟡 **RIESGO MODERADO**", "\n".join(detalles)
    else:
        return "🟢 **CONFIABLE**", "- No se detectan anomalías automáticas evidentes."

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.photo[-1].get_file()
    input_path = f"input_{update.effective_chat.id}.jpg"
    
    try:
        await file.download_to_drive(input_path)
        await update.message.reply_text("🔍 Iniciando análisis forense avanzado de textura y bordes...")

        # Ejecutar análisis matemático
        noise_score, edge_score = analyze_image(input_path)
        
        # Generar el dictamen tipo semáforo
        titulo, detalles = generate_dictamen(noise_score, edge_score)
        
        # Formatear el mensaje de respuesta para los economistas
        mensaje_final = (
            f"📊 **DICTAMEN DE INTEGRIDAD**\n"
            f"---------------------------------\n"
            f"Veredicto: {titulo}\n\n"
            f"**Detalles Técnicos:**\n"
            f"{detalles}\n\n"
            f"_Recuerde: Este análisis es una herramienta de apoyo, no sustituye el criterio humano._"
        )
        await update.message.reply_text(mensaje_final, parse_mode=constants.ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("❌ Ocurrió un error durante el análisis.")
    finally:
        # Limpieza de archivos temporales
        if os.path.exists(input_path): os.remove(input_path)

if __name__ == '__main__':
    # Ejecutar el servidor web en un hilo secundario
    threading.Thread(target=run_flask, daemon=True).start()
    
    # Configurar e iniciar el Bot de Telegram
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    
    print("Bot con umbrales actualizados escuchando...")
    app.run_polling()
