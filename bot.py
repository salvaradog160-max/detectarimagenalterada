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

# --- WEB SERVER PARA RENDER ---
web_app = Flask(__name__)
@web_app.route('/')
def health_check(): return "Analizador Forense Activo", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host='0.0.0.0', port=port)

# --- MOTOR DE ANÁLISIS MULTICANAL ---
def analyze_forensics(image_path):
    # 1. ANÁLISIS ELA (Error Level Analysis)
    quality = 90
    original = Image.open(image_path).convert('RGB')
    temp_resave = "temp_resave.jpg"
    original.save(temp_resave, 'JPEG', quality=quality)
    temporary = Image.open(temp_resave)
    
    ela_image = ImageChops.difference(original, temporary)
    stat = ImageStat.Stat(ela_image)
    ela_score = sum(stat.mean) # Magnitud de error de compresión
    
    # Preparamos la imagen ELA visual para el reporte
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_visual = ImageEnhance.Brightness(ela_image).enhance(scale)

    # 2. ANÁLISIS DE TEXTURA Y BORDES (OpenCV)
    img_cv = cv2.imread(image_path, 0)
    
    # Detección de parches lisos (Textura)
    local_std = cv2.blur(img_cv, (15,15))
    _, thr = cv2.threshold(local_std, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    diff_textura = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    noise_score = np.mean(diff_textura)

    # Detección de nitidez artificial (Bordes)
    edges = cv2.Canny(img_cv, 100, 200)
    edge_score = np.mean(edges)

    os.remove(temp_resave)
    return ela_visual, ela_score, noise_score, edge_score

def get_verdict(ela_score, noise_score, edge_score):
    # Umbrales calibrados
    # ELA > 12, Noise > 25, Edges > 18
    risk_points = 0
    detalles = []

    if noise_score > 25.0:
        risk_points += 3
        detalles.append("- 🔴 **TEXTURA**: Parche digital detectado (fondo demasiado liso).")
    
    if edge_score > 18.0:
        risk_points += 2
        detalles.append("- 🔴 **NITIDEZ**: Bordes de texto con perfección artificial.")

    if ela_score > 12.0:
        risk_points += 1
        detalles.append("- 🟡 **COMPRESIÓN**: Inconsistencia en metadatos de píxeles (ELA).")

    if risk_points >= 3:
        return "🔴 **RIESGO CRÍTICO / POSIBLE ALTERACIÓN**", "\n".join(detalles)
    elif risk_points >= 1:
        return "🟡 **RIESGO MODERADO**", "\n".join(detalles)
    else:
        return "🟢 **CONFIABLE**", "- No se detectan anomalías evidentes en los canales forenses."

# --- MANEJADOR DEL BOT ---
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Descargar la imagen
    photo_file = await update.message.photo[-1].get_file()
    input_fn = f"img_{update.effective_chat.id}.jpg"
    output_ela_fn = f"ela_{update.effective_chat.id}.jpg"
    
    try:
        await photo_file.download_to_drive(input_fn)
        await update.message.reply_text("🕵️ Analizando capas, textura y compresión... un momento.")

        # Procesar
        ela_vis, ela_s, noise_s, edge_s = analyze_forensics(input_fn)
        ela_vis.save(output_ela_fn)

        # Dictamen
        titulo, detalles = get_verdict(ela_s, noise_s, edge_s)

        # Enviar Revelador ELA
        await update.message.reply_photo(photo=open(output_ela_fn, 'rb'), caption="🖼️ Mapa de Error de Compresión (ELA)")

        # Enviar Dictamen Final
        reporte = (
            f"📊 **DICTAMEN DE INTEGRIDAD**\n"
            f"---------------------------------\n"
            f"Veredicto: {titulo}\n\n"
            f"**Análisis por Canales:**\n"
            f"{detalles}\n\n"
            f"_Criterio humano sugerido: Revise si el monto brilla en el mapa superior._"
        )
        await update.message.reply_text(reporte, parse_mode=constants.ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("❌ Error al procesar la imagen.")
    finally:
        if os.path.exists(input_fn): os.remove(input_fn)
        if os.path.exists(output_ela_fn): os.remove(output_ela_fn)

if __name__ == '__main__':
    threading.Thread(target=run_flask, daemon=True).start()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_document))
    print("Bot Forense V3.0 en marcha...")
    app.run_polling()
