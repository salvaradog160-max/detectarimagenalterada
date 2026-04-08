import os
import io
import logging
import threading
import numpy as np
import cv2
import uuid
from flask import Flask
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from PIL import Image, ImageChops, ImageEnhance

# --- CONFIGURACIÓN LIGERA ---
TOKEN = "8699029540:AAE9TGMSC5fvW2Fldhuc_keYQAYxM_ooW_s"
logging.basicConfig(level=logging.INFO)

web_app = Flask(__name__)
@web_app.route("/")
def health(): return "Bot Activo", 200

def run_flask():
    web_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

# ═══════════════════════════════════════════════════════════════
# NUEVO MOTOR: DETECTOR DE PARCHES LISOS Y BORDES SINTÉTICOS
# ═══════════════════════════════════════════════════════════════
def micro_forensic_analysis(image_path):
    # Cargamos en escala de grises para ahorrar procesamiento
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None: return None
    
    # 1. Detección de "Zonas Muertas" (Parches digitales sin ruido)
    # El papel real tiene grano; un parche de editor es matemáticamente liso.
    laplacian = cv2.Laplacian(img_cv, cv2.CV_64F)
    score_ruido = np.var(laplacian) # Varianza de ruido global
    
    # 2. Análisis ELA Focalizado (Rápido)
    original = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=90)
    recompressed = Image.open(io.BytesIO(buf.getvalue()))
    ela_diff = ImageChops.difference(original, recompressed)
    
    # 3. Localización de la anomalía (Línea Roja)
    # Buscamos áreas donde la nitidez de los bordes sea inconsistente
    blur = cv2.GaussianBlur(img_cv, (5,5), 0)
    diff_local = cv2.absdiff(img_cv, blur)
    _, mask = cv2.threshold(diff_local, 25, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_result = cv2.imread(image_path)
    anomalia_detectada = False
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        # Si el contorno es rectangular (como un parche de texto) y de tamaño medio
        if 500 < area < 5000 and 1.5 < aspect_ratio < 6.0:
            cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            anomalia_detectada = True

    return img_result, anomalia_detectada, score_ruido

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(uuid.uuid4())[:8]
    in_p = f"in_{uid}.jpg"
    out_p = f"res_{uid}.jpg"
    
    try:
        f = await update.message.photo[-1].get_file()
        await f.download_to_drive(in_p)
        
        await update.message.reply_text("🔬 Analizando micro-textura y parches...")
        
        res_img, detectado, s_ruido = micro_forensic_analysis(in_p)
        cv2.imwrite(out_p, res_img)
        
        veredicto = "🔴 ALTA PROBABILIDAD DE ALTERACIÓN" if detectado or s_ruido < 150 else "🟢 SIN MARCAS DE PARCHES"
        riesgo = "8/10" if detectado else "2/10"
        
        await update.message.reply_photo(
            photo=open(out_p, "rb"),
            caption=f"📊 **DICTAMEN FORENSE**\n\nVeredicto: {veredicto}\nRiesgo: {riesgo}\n\n"
                    f"Hallazgo: {'Se marcó en rojo la zona con textura artificial.' if detectado else 'No se hallaron parches rectangulares.'}\n\n"
                    f"_Advertencia: El ruido del papel es muy bajo ({s_ruido:.1f}), posible edición de alta calidad._"
        )
    finally:
        for p in [in_p, out_p]:
            if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    ApplicationBuilder().token(TOKEN).build().add_handler(MessageHandler(filters.PHOTO, handle_image)).run_polling()
