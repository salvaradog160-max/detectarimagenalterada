import os
import logging
import threading
import numpy as np
import cv2
import uuid
import piexif
from flask import Flask
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from PIL import Image, ImageChops, ImageEnhance, ImageStat

# --- CONFIGURACIÓN ---
TOKEN = "8699029540:AAE9TGMSC5fvW2Fldhuc_keYQAYxM_ooW_s"

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

web_app = Flask(__name__)
@web_app.route('/')
def health_check(): return "Analizador Forense 3.5 Activo", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host='0.0.0.0', port=port)

# --- ANÁLISIS DE METADATOS (ADN DEL ARCHIVO) ---
def analyze_exif(image_path):
    software_sospechoso = ["photoshop", "canva", "picsart", "adobe", "gimp", "screenshot", "editor"]
    hallazgo = None
    try:
        img = Image.open(image_path)
        exif_data = img.info.get("exif", b"")
        if exif_data:
            exif_dict = piexif.load(exif_data)
            software = str(exif_dict.get("0th", {}).get(piexif.ImageIFD.Software, b"")).lower()
            for app in software_sospechoso:
                if app in software:
                    hallazgo = app.capitalize()
                    break
    except: pass
    return hallazgo

# --- MOTOR FORENSE INDEPENDIENTE ---
def analyze_forensics(image_path):
    # Generar ID único para evitar que se mezclen resultados
    uid = str(uuid.uuid4())[:8]
    temp_resave = f"resave_{uid}.jpg"
    
    # 1. ELA (Error Level Analysis)
    original = Image.open(image_path).convert('RGB')
    original.save(temp_resave, 'JPEG', quality=90)
    temporary = Image.open(temp_resave)
    ela_image = ImageChops.difference(original, temporary)
    stat = ImageStat.Stat(ela_image)
    ela_score = sum(stat.mean)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela_visual = ImageEnhance.Brightness(ela_image).enhance(scale)

    # 2. TEXTURA Y BORDES (OpenCV)
    img_cv = cv2.imread(image_path, 0)
    # Suavizamos el análisis de textura para evitar falsos positivos en documentos digitales
    blur = cv2.GaussianBlur(img_cv, (5,5), 0)
    noise_score = np.std(blur) # Usamos desviación estándar para medir la "naturalidad"

    edges = cv2.Canny(img_cv, 100, 200)
    edge_score = np.mean(edges)

    if os.path.exists(temp_resave): os.remove(temp_resave)
    return ela_visual, ela_score, noise_score, edge_score

def get_verdict(ela_s, noise_s, edge_s, soft):
    puntos = 0
    hallazgos = []

    # Alerta por Software
    if soft:
        puntos += 4
        hallazgos.append(f"- 🚩 **SOFTWARE**: Detectado rastro de {soft}.")

    # Alerta por Textura (Ajustada: los parches suelen tener desviación muy baja < 2.0)
    if noise_s < 2.0:
        puntos += 3
        hallazgos.append("- ⚠️ **TEXTURA**: Área sospechosamente lisa (posible parche digital).")

    # Alerta por Bordes (Nitidez digital)
    if edge_s > 30.0:
        puntos += 2
        hallazgos.append("- ⚠️ **NITIDEZ**: Bordes de texto con perfección digital.")

    # Veredicto
    if puntos >= 4:
        return "🔴 **RIESGO ALTO / POSIBLE ALTERACIÓN**", "\n".join(hallazgos)
    elif puntos >= 2:
        return "🟡 **RIESGO MODERADO**", "\n".join(hallazgos)
    else:
        return "🟢 **CONFIABLE**", "- No se detectan anomalías evidentes."

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg_id = update.effective_message.message_id
    in_file = f"in_{msg_id}.jpg"
    out_ela = f"ela_{msg_id}.jpg"
    
    try:
        photo = await update.message.photo[-1].get_file()
        await photo.download_to_drive(in_file)
        await update.message.reply_text("🕵️ Analizando integridad del documento...")

        # Ejecución única
        soft = analyze_exif(in_file)
        ela_vis, ela_s, noise_s, edge_s = analyze_forensics(in_file)
        ela_vis.save(out_ela)

        titulo, detalles = get_verdict(ela_s, noise_s, edge_s, soft)

        # Enviar Imagen ELA
        await update.message.reply_photo(
            photo=open(out_ela, 'rb'), 
            caption="🖼️ **Análisis ELA (Rayos X)**\n_Observe si hay brillos aislados en montos o fechas._"
        )

        # Enviar Dictamen
        reporte = (
            f"📊 **DICTAMEN DE INTEGRIDAD**\n"
            f"---------------------------------\n"
            f"Sugerencia: {titulo}\n\n"
            f"**Hallazgos técnicos:**\n"
            f"{detalles}\n\n"
            f"_Recuerde: Este análisis es una herramienta de apoyo que no sustituye el ojo humano ni el criterio del analista._"
        )
        await update.message.reply_text(reporte, parse_mode=constants.ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("❌ Error al procesar. Reintente.")
    finally:
        if os.path.exists(in_file): os.remove(in_file)
        if os.path.exists(out_ela): os.remove(out_ela)

if __name__ == '__main__':
    threading.Thread(target=run_flask, daemon=True).start()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
