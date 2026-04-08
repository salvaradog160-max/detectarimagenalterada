import os
import logging
import threading
import numpy as np
import cv2
import uuid
from flask import Flask
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from PIL import Image, ImageChops, ImageEnhance, ImageStat

# --- CONFIGURACIÓN ---
TOKEN = "8699029540:AAE9TGMSC5fvW2Fldhuc_keYQAYxM_ooW_s"

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# --- SERVIDOR WEB (Salvavidas para Render) ---
web_app = Flask(__name__)
@web_app.route('/')
def health_check(): return "Analizador Forense 3.2 Activo", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host='0.0.0.0', port=port)

# --- MOTOR FORENSE ATÓMICO (Sin memoria entre ejecuciones) ---
def analyze_forensics(image_path):
    # Generamos identificadores únicos para evitar conflictos de archivos
    unique_id = str(uuid.uuid4())[:8]
    temp_resave = f"temp_{unique_id}.jpg"
    
    # 1. ANÁLISIS ELA (Error Level Analysis)
    original = Image.open(image_path).convert('RGB')
    original.save(temp_resave, 'JPEG', quality=90)
    temporary = Image.open(temp_resave)
    
    ela_image = ImageChops.difference(original, temporary)
    stat = ImageStat.Stat(ela_image)
    ela_score = sum(stat.mean)
    
    # Generar visualización de "Rayos X"
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela_visual = ImageEnhance.Brightness(ela_image).enhance(scale)

    # 2. ANÁLISIS DE TEXTURA LOCAL (Detección de Parches)
    img_cv = cv2.imread(image_path, 0)
    # Calculamos la desviación estándar local para detectar zonas "lisas" (parches)
    local_std = cv2.blur(img_cv, (15,15))
    _, thr = cv2.threshold(local_std, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    diff_textura = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    noise_score = np.mean(diff_textura)

    # 3. ANÁLISIS DE BORDES (Detección de Texto Vectorial)
    edges = cv2.Canny(img_cv, 100, 200)
    edge_score = np.mean(edges)

    # Limpiar archivo temporal de resalvado
    if os.path.exists(temp_resave): os.remove(temp_resave)
    
    return ela_visual, ela_score, noise_score, edge_score

def get_verdict(ela_score, noise_score, edge_score):
    # Umbrales calibrados para reducir Falsos Positivos
    # Puntuación acumulativa para dictamen
    puntos = 0
    hallazgos = []

    if noise_score > 35.0: 
        puntos += 3
        hallazgos.append("- 🔴 **ANOMALÍA DE TEXTURA**: Se detectó un área sospechosamente lisa (típico de parches digitales).")
    
    if edge_score > 22.0: 
        puntos += 2
        hallazgos.append("- 🔴 **NITIDEZ ARTIFICIAL**: Texto con bordes demasiado perfectos para papel físico.")

    if ela_score > 15.0:
        puntos += 1
        hallazgos.append("- 🟡 **COMPRESIÓN**: Inconsistencia leve en el nivel de error de los píxeles.")

    # Generación de Dictamen Consultivo
    if puntos >= 3:
        return "🔴 **ATENCIÓN: ALTA PROBABILIDAD DE ALTERACIÓN**", "\n".join(hallazgos)
    elif puntos >= 1:
        return "🟡 **OBSERVACIÓN RECOMENDADA**", "\n".join(hallazgos)
    else:
        return "🟢 **SIN ANOMALÍAS DETECTADAS**", "- El documento parece íntegro bajo los criterios actuales de validación."

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Usar ID de mensaje para archivos únicos por petición
    task_id = update.effective_message.message_id
    input_file = f"in_{task_id}.jpg"
    output_ela = f"ela_{task_id}.jpg"
    
    try:
        photo = await update.message.photo[-1].get_file()
        await photo.download_to_drive(input_file)
        
        await update.message.reply_text("🕵️ Analizando capas, textura y compresión... un momento.")

        # Procesar de forma aislada
        ela_img, ela_s, noise_s, edge_s = analyze_forensics(input_file)
        ela_img.save(output_ela)

        veredicto, detalles = get_verdict(ela_s, noise_s, edge_s)

        # 1. Enviar Revelado (Rayos X)
        await update.message.reply_photo(
            photo=open(output_ela, 'rb'), 
            caption="🖼️ **Análisis ELA (Rayos X)**\n_Si el monto brilla de forma aislada, hay edición._",
            parse_mode=constants.ParseMode.MARKDOWN
        )

        # 2. Enviar Estado de Integridad
        reporte = (
            f"📊 **ESTADO DE INTEGRIDAD**\n"
            f"---------------------------------\n"
            f"Sugerencia: {veredicto}\n\n"
            f"**Hallazgos técnicos:**\n"
            f"{detalles}\n\n"
            f"_Nota: El economista debe verificar que el ticket coincida con la sucursal y fecha reportada._"
        )
        await update.message.reply_text(reporte, parse_mode=constants.ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Error en tarea {task_id}: {e}")
        await update.message.reply_text("❌ Error al procesar la imagen. Intente de nuevo.")
    finally:
        # Limpieza rigurosa después de cada uso
        if os.path.exists(input_file): os.remove(input_file)
        if os.path.exists(output_ela): os.remove(output_ela)

if __name__ == '__main__':
    threading.Thread(target=run_flask, daemon=True).start()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    print("Bot 3.2 listo y escuchando...")
    app.run_polling()
