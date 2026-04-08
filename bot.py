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
def health_check(): return "Analizador Forense 3.4 Activo", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host='0.0.0.0', port=port)

# --- NUEVA FUNCIÓN: ANÁLISIS DE METADATOS ---
def analyze_exif(image_path):
    software_sospechoso = ["photoshop", "canva", "picsart", "adobe", "gimp", "screenshot", "editor"]
    hallazgo_software = None
    try:
        img = Image.open(image_path)
        exif_dict = piexif.load(img.info.get("exif", b""))
        
        # Revisamos los campos donde el software suele dejar su firma
        software = str(exif_dict.get("0th", {}).get(piexif.ImageIFD.Software, b"")).lower()
        model = str(exif_dict.get("0th", {}).get(piexif.ImageIFD.Model, b"")).lower()
        
        for app in software_sospechoso:
            if app in software or app in model:
                hallazgo_software = app.capitalize()
                break
    except Exception:
        pass # Muchos archivos no tienen EXIF, lo cual es normal en Telegram
    return hallazgo_software

# --- MOTOR FORENSE INTEGRADO ---
def analyze_forensics(image_path):
    unique_id = str(uuid.uuid4())[:8]
    temp_resave = f"temp_{unique_id}.jpg"
    
    # 1. ELA (Compresión)
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

    # 2. TEXTURA (Ruido Local)
    img_cv = cv2.imread(image_path, 0)
    local_std = cv2.blur(img_cv, (15,15))
    _, thr = cv2.threshold(local_std, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    diff_textura = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    noise_score = np.mean(diff_textura)

    # 3. BORDES (Nitidez)
    edges = cv2.Canny(img_cv, 100, 200)
    edge_score = np.mean(edges)

    if os.path.exists(temp_resave): os.remove(temp_resave)
    
    return ela_visual, ela_score, noise_score, edge_score

def get_verdict(ela_score, noise_score, edge_score, software_detectado):
    puntos = 0
    hallazgos = []

    # REGLA DE ORO: Si hay software de edición, alerta máxima
    if software_detectado:
        puntos += 5
        hallazgos.append(f"- 🔴 **METADATOS**: Archivo creado/editado con {software_detectado}.")

    if noise_score > 45.0: 
        puntos += 3
        hallazgos.append("- ⚠️ Inconsistencia en textura (posible parche digital).")
    
    if edge_score > 28.0: 
        puntos += 2
        hallazgos.append("- ⚠️ Bordes con nitidez artificial (común en capturas).")

    if ela_score > 18.0:
        puntos += 1
        hallazgos.append("- ⚠️ Inconsistencia leve en píxeles.")

    if puntos >= 4:
        return "🔴 **RIESGO CRÍTICO / POSIBLE ALTERACIÓN**", "\n".join(hallazgos)
    elif puntos >= 2:
        return "🟡 **OBSERVACIÓN RECOMENDADA**", "\n".join(hallazgos)
    else:
        return "🟢 **CONFIABLE**", "- No se detectan anomalías evidentes."

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg_id = update.effective_message.message_id
    input_file = f"in_{msg_id}.jpg"
    output_ela = f"ela_{msg_id}.jpg"
    
    try:
        photo = await update.message.photo[-1].get_file()
        await photo.download_to_drive(input_file)
        
        await update.message.reply_text("🕵️ Escaneando ADN del archivo, textura y compresión...")

        # Análisis
        soft_detect = analyze_exif(input_file)
        ela_img, ela_s, noise_s, edge_s = analyze_forensics(input_file)
        ela_img.save(output_ela)

        veredicto, detalles = get_verdict(ela_s, noise_s, edge_s, soft_detect)

        # 1. Enviar Rayos X
        await update.message.reply_photo(
            photo=open(output_ela, 'rb'), 
            caption="🖼️ **Análisis ELA (Rayos X)**",
            parse_mode=constants.ParseMode.MARKDOWN
        )

        # 2. Enviar Reporte
        reporte = (
            f"📊 **ESTADO DE INTEGRIDAD**\n"
            f"---------------------------------\n"
            f"Sugerencia: {veredicto}\n\n"
            f"**Hallazgos técnicos:**\n"
            f"{detalles}\n\n"
            f"_Este bot detecta firmas de software y parches digitales._"
        )
        await update.message.reply_text(reporte, parse_mode=constants.ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("❌ Error al procesar. Intente de nuevo.")
    finally:
        if os.path.exists(input_file): os.remove(input_file)
        if os.path.exists(output_ela): os.remove(output_ela)

if __name__ == '__main__':
    threading.Thread(target=run_flask, daemon=True).start()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
