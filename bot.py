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
    return "Analizador Forense Activo", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host='0.0.0.0', port=port)

# --- MOTOR DE ANÁLISIS MULTICANAL (ELA + Textura + Bordes) ---
def analyze_forensics(image_path):
    # 1. ANÁLISIS ELA (Nivel de Error de Compresión)
    quality = 90
    original = Image.open(image_path).convert('RGB')
    temp_resave = "temp_resave.jpg"
    original.save(temp_resave, 'JPEG', quality=quality)
    temporary = Image.open(temp_resave)
    
    ela_image = ImageChops.difference(original, temporary)
    stat = ImageStat.Stat(ela_image)
    ela_score = sum(stat.mean) # Magnitud de error de compresión
    
    # Preparamos la imagen ELA visual (los "rayos X")
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_visual = ImageEnhance.Brightness(ela_image).enhance(scale)

    # 2. ANÁLISIS DE TEXTURA Y BORDES (OpenCV)
    img_cv = cv2.imread(image_path, 0)
    
    # Detección de parches lisos (Textura) - Usamos los umbrales calibrados
    local_std = cv2.blur(img_cv, (15,15))
    _, thr = cv2.threshold(local_std, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    diff_textura = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    noise_score = np.mean(diff_textura)

    # Detección de nitidez artificial (Bordes) - Usamos los umbrales calibrados
    edges = cv2.Canny(img_cv, 100, 200)
    edge_score = np.mean(edges)

    os.remove(temp_resave)
    # Devolvemos la imagen ELA visualizada y los scores técnicos
    return ela_visual, ela_score, noise_score, edge_score

def get_verdict(ela_score, noise_score, edge_score):
    # --- UMBRALES CALIBRADOS ---
    risk_points = 0
    detalles = []

    if noise_score > 25.0: # Detección de parches Power Point
        risk_points += 3
        detalles.append("- 🔴 **TEXTURA**: Parche digital detectado (fondo demasiado liso).")
    
    if edge_score > 18.0: # Detección de texto vectorial nítido
        risk_points += 2
        detalles.append("- 🔴 **NITIDEZ**: Bordes de texto con perfección artificial.")

    if ela_score > 12.0: # Detección de inconsistencia de píxeles
        risk_points += 1
        detalles.append("- 🟡 **COMPRESIÓN**: Inconsistencia en metadatos de píxeles (ELA).")

    # Veredicto final tipo semáforo
    if risk_points >= 3:
        return "🔴 **RIESGO CRÍTICO / POSIBLE ALTERACIÓN**", "\n".join(detalles)
    elif risk_points >= 1:
        return "🟡 **RIESGO MODERADO**", "\n".join(detalles)
    else:
        return "🟢 **CONFIABLE**", "- No se detectan anomalías evidentes en los canales forenses."

# --- MANEJADOR DE MENSAJES DEL BOT ---
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Descargar la imagen temporalmente
    photo_file = await update.message.photo[-1].get_file()
    input_fn = f"img_{update.effective_chat.id}.jpg"
    output_ela_fn = f"ela_{update.effective_chat.id}.jpg"
    
    try:
        await photo_file.download_to_drive(input_fn)
        await update.message.reply_text("🕵️ Analizando capas, textura y compresión... un momento.")

        # Ejecutar el motor de análisis multicanal
        ela_vis, ela_s, noise_s, edge_s = analyze_forensics(input_fn)
        # Guardamos la imagen ELA visual para enviarla
        ela_vis.save(output_ela_fn)

        # Generar el dictamen
        titulo, detalles = get_verdict(ela_s, noise_s, edge_s)

        # --- AQUÍ ESTÁ LA CORRECCIÓN: ENVIAR LA IMAGEN "RAYOS X" ---
        await update.message.reply_photo(
            photo=open(output_ela_fn, 'rb'), 
            caption="🖼️ **Mapa de Error de Compresión (ELA)**\n_Si el monto brilla sospechosamente, hay alteración._",
            parse_mode=constants.ParseMode.MARKDOWN
        )

        # Enviar el Dictamen Final formateado
        reporte = (
            f"📊 **DICTAMEN DE INTEGRIDAD**\n"
            f"---------------------------------\n"
            f"Veredicto: {titulo}\n\n"
            f"**Análisis por Canales:**\n"
            f"{detalles}\n\n"
            f"_Recuerde: Este análisis es una herramienta de apoyo, no sustituye el criterio humano._"
        )
        await update.message.reply_text(reporte, parse_mode=constants.ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("❌ Error al procesar la imagen.")
    finally:
        # Limpieza de archivos temporales
        if os.path.exists(input_fn): os.remove(input_fn)
        if os.path.exists(output_ela_fn): os.remove(output_ela_fn)

if __name__ == '__main__':
    # Ejecutar el servidor web en un hilo secundario
    threading.Thread(target=run_flask, daemon=True).start()
    
    # Configurar e iniciar el Bot de Telegram
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_document))
    
    print("Bot Forense V3.1 Escuchando con 'Rayos X' restaurados...")
    app.run_polling()
