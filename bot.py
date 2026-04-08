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
# Chat IDs autorizados (Opcional, para seguridad)
# AUTHORIZED_CHATS = [12345678, 87654321] 

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
    # 1. ELA (Nivel de Error)
    quality = 90
    original = Image.open(image_path).convert('RGB')
    temp_resave = "temp_resave.jpg"
    original.save(temp_resave, 'JPEG', quality=quality)
    temporary = Image.open(temp_resave)
    ela_image = ImageChops.difference(original, temporary)
    
    # Calcular Score de Ruido ELA
    stat = ImageStat.Stat(ela_image)
    ela_score = sum(stat.mean) # Promedio de brillo en ELA
    
    # Amplificar ELA para visualización
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_visual = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # 2. Análisis de Bordes (Detectar texto nítido insertado)
    img_cv = cv2.imread(image_path, 0)
    edges = cv2.Canny(img_cv, 100, 200)
    edge_score = np.mean(edges) # Promedio de bordes

    os.remove(temp_resave)
    return ela_visual, ela_score, edge_score

def generate_dictamen(ela_score, edge_score):
    # Umbrales empíricos (ajustar según pruebas)
    ELA_THRESHOLD_HIGH = 15.0
    EDGE_THRESHOLD_HIGH = 10.0
    
    risk_points = 0
    detalles = []

    if ela_score > ELA_THRESHOLD_HIGH:
        risk_points += 2
        detalles.append("- 🔴 Brillo ELA inusual (Posible manipulación de píxeles).")
    elif ela_score > 8.0:
        risk_points += 1
        detalles.append("- 🟡 Ruido ELA moderado (Revisar con cuidado).")
    
    if edge_score > EDGE_THRESHOLD_HIGH:
        risk_points += 2
        detalles.append("- 🔴 Bordes de texto sospechosamente nítidos (Posible texto insertado).")

    # Veredicto
    if risk_points >= 3:
        return "🔴 **RIESGO CRÍTICO DETECTADO**", "\n".join(detalles)
    elif risk_points >= 1:
        return "🟡 **RIESGO MODERADO**", "\n".join(detalles)
    else:
        return "🟢 **CONFIABLE**", "- No se detectan anomalías automáticas evidentes."

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Seguridad (Descomentar si quieres limitar uso)
    # if update.effective_chat.id not in AUTHORIZED_CHATS: return

    file = await update.message.photo[-1].get_file()
    input_path = f"input_{update.effective_chat.id}.jpg"
    output_ela_path = f"ela_{update.effective_chat.id}.jpg"
    
    try:
        await file.download_to_drive(input_path)
        await update.message.reply_text("🔍 Iniciando análisis forense multicanal...")

        # Ejecutar análisis
        ela_visual, ela_score, edge_score = analyze_image(input_path)
        ela_visual.save(output_ela_path)
        
        # Generar Dictamen
        titulo, detalles = generate_dictamen(ela_score, edge_score)
        
        # Enviar Imagen ELA
        await update.message.reply_photo(photo=open(output_ela_path, 'rb'))
        
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
        if os.path.exists(output_ela_path): os.remove(output_ela_path)

if __name__ == '__main__':
    threading.Thread(target=run_flask, daemon=True).start()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    print("Bot avanzado escuchando...")
    app.run_polling()
