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

# --- WEB SERVER FALSO PARA RENDER ---
web_app = Flask(__name__)
@web_app.route('/')
def health_check(): return "Validador Vivo", 200
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

    # 3. Bordes Canny (Ya lo teníamos, es bueno para texto nítido)
    edges = cv2.Canny(img_cv, 100, 200)
    edge_score = np.mean(edges)

    return noise_score, edge_score

def generate_dictamen(noise_score, edge_score):
    # Umbrales empíricos (ajustar según pruebas)
    NOISE_THRESHOLD_HIGH = 15.0 # Sensibilidad para detectar parches lisos
    EDGE_THRESHOLD_HIGH = 10.0
    
    risk_points = 0
    detalles = []

    # Detección de parches digitales (lisos)
    if noise_score > NOISE_THRESHOLD_HIGH:
        risk_points += 3
        detalles.append("- 🔴 **PARCHE DETECTADO**: Se detectaron zonas sospechosamente lisas en el documento (típico de montos sobrepuestos digitales).")
    elif noise_score > 8.0:
        risk_points += 1
        detalles.append("- 🟡 Textura del papel inconsistente.")

    # Bordes sospechosos (Texto demasiado nítido)
    if edge_score > EDGE_THRESHOLD_HIGH:
        risk_points += 2
        detalles.append("- 🔴 Letras o números demasiado nítidos para ser una foto real.")

    # Veredicto
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

        # Ejecutar análisis
        noise_score, edge_score = analyze_image(input_path)
        
        # Generar Dictamen
        titulo, detalles = generate_dictamen(noise_score, edge_score)
        
        # Enviar Dictamen formateado
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
        if os.path.exists(input_path): os.remove(input_path)

if __name__ == '__main__':
    threading.Thread(target=run_flask, daemon=True).start()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
