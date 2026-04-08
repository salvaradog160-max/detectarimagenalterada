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
def health_check(): return "Analizador Forense 3.7 Activo", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host='0.0.0.0', port=port)

# --- ADN DEL ARCHIVO (METADATOS) ---
def analyze_exif(image_path):
    software_sospechoso = ["photoshop", "canva", "picsart", "adobe", "gimp", "screenshot", "editor"]
    try:
        img = Image.open(image_path)
        info = img.info.get("exif")
        if info:
            exif_dict = piexif.load(info)
            soft = str(exif_dict.get("0th", {}).get(piexif.ImageIFD.Software, b"")).lower()
            for app in software_sospechoso:
                if app in soft: return app.capitalize()
    except: pass
    return None

# --- MOTOR FORENSE AVANZADO (CADA REVISIÓN ES ÚNICA) ---
def analyze_forensics(image_path):
    # Generar ID único por mensaje para evitar cruce de datos
    uid = str(uuid.uuid4())[:8]
    temp_resave = f"res_{uid}.jpg"
    
    # 1. ELA (Nivel de Error)
    img_pil = Image.open(image_path).convert('RGB')
    img_pil.save(temp_resave, 'JPEG', quality=90)
    resaved = Image.open(temp_resave)
    
    # Calcular diferencia de compresión
    ela_pil = ImageChops.difference(img_pil, resaved)
    stat = ImageStat.Stat(ela_pil)
    ela_score = sum(stat.mean)
    
    # Amplificar brillo para visualización
    extrema = ela_pil.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela_scaled = ImageEnhance.Brightness(ela_pil).enhance(255.0 / max_diff)
    
    # Limpieza inmediata
    if os.path.exists(temp_resave): os.remove(temp_resave)

    # Convertir a OpenCV para análisis de textura y marcación
    img_cv = cv2.imread(image_path, 0)
    ela_cv = cv2.cvtColor(np.array(ela_scaled), cv2.COLOR_RGB2BGR)
    
    # 2. ANÁLISIS DE TEXTURA LOCAL
    # Calculamos la varianza local
    kernel_size = 15 # Bloques más grandes para capturar montos completos
    local_mean = cv2.blur(img_cv, (kernel_size, kernel_size))
    local_sqr_mean = cv2.blur(img_cv**2, (kernel_size, kernel_size))
    local_var = local_sqr_mean - local_mean**2
    
    # Detectamos parches muertos (lisos) y buscamos el más grande
    _, thr = cv2.threshold(local_var.astype('uint8'), 5, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    anomalia_coordenadas = None
    if contours:
        # Buscamos el contorno más grande que sea sospechoso de parche
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500: # Solo si el parche es significativo
            x, y, w, h = cv2.boundingRect(largest_contour)
            anomalia_coordenadas = (x, y, w, h)
            # --- AQUÍ ESTÁ LA MEJORA: MARCAMOS CON LÍNEA ROJA ---
            cv2.rectangle(ela_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(ela_cv, "POSIBLE PARCHE DIGITAL", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 3. ANÁLISIS DE BORDES (NITIDEZ DIGITAL)
    edges = cv2.Canny(img_cv, 100, 200)
    edge_score = np.mean(edges)

    return ela_cv, ela_score, anomalia_coordenadas, edge_score

def get_verdict(soft, ela_s, anomalia_coordenadas, edge_s):
    puntos = 0
    hallazgos = []

    if soft:
        puntos += 5
        hallazgos.append(f"- 🔴 **METADATOS**: Software '{soft}' detectado.")

    if anomalia_coordenadas:
        puntos += 3
        hallazgos.append("- 🔴 **TEXTURA**: Parche digital detectado (marcado en rojo).")

    if edge_s > 30.0:
        puntos += 2
        hallazgos.append("- ⚠️ **NITIDEZ**: Bordes con perfección digital.")

    if pontos >= 4:
        return "🔴 **RIESGO ALTO / POSIBLE ALTERACIÓN**", "\n".join(hallazgos)
    elif pontos >= 2:
        return "🟡 **RIESGO MODERADO**", "\n".join(hallazgos)
    else:
        return "🟢 **CONFIABLE**", "- No se detectan anomalías evidentes en esta revisión única."

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    task_id = f"{update.effective_chat.id}_{update.message.message_id}"
    input_f = f"in_{task_id}.jpg"
    output_f = f"ela_{task_id}.jpg"
    
    try:
        f = await update.message.photo[-1].get_file()
        await f.download_to_drive(input_f)
        await update.message.reply_text("🕵️ Iniciando análisis forense visual único...")

        # Ejecutar análisis integrado
        soft = analyze_exif(input_f)
        ela_marked, ela_s, anomalia_coord, edge_s = analyze_forensics(input_f)
        
        # Guardar imagen con la marca roja
        cv2.imwrite(output_f, ela_marked)

        titulo, detalles = get_verdict(soft, ela_s, anomalia_coord, edge_s)

        # 1. Enviar Revelado con Marca Roja
        await update.message.reply_photo(
            photo=open(output_f, 'rb'), 
            caption="🖼️ **Análisis Forense con Marcador de Riesgo**"
        )

        # 2. Enviar Reporte para la Mesa de Crédito
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
        await update.message.reply_text("❌ Error en el proceso. Reintente.")
    finally:
        for f in [input_f, output_f]:
            if os.path.exists(f): os.remove(f)

if __name__ == '__main__':
    threading.Thread(target=run_flask, daemon=True).start()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_document))
    print("Bot 3.7 listo y escuchando con marcador visual...")
    app.run_polling()
