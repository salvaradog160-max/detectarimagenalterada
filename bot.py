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

# --- CONFIGURACIÓN ---
TOKEN = "8699029540:AAE9TGMSC5fvW2Fldhuc_keYQAYxM_ooW_s"
logging.basicConfig(level=logging.INFO)

web_app = Flask(__name__)
@web_app.route("/")
def health(): return "Analizador Forense 4.6 Activo ✅", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)

# ═══════════════════════════════════════════════════════════════
# MOTOR MICRO-FORENSE (Ligero y Eficiente para Render)
# ═══════════════════════════════════════════════════════════════
def micro_analysis(image_path):
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None: return None, False, 0, 0
    
    # 1. Score de Ruido (Laplacian): El papel real tiene grano, el digital no.
    laplacian = cv2.Laplacian(img_cv, cv2.CV_64F)
    score_ruido = np.var(laplacian)
    
    # 2. Análisis ELA Rápido (Compresión)
    original = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=90)
    recompressed = Image.open(io.BytesIO(buf.getvalue()))
    ela_diff = ImageChops.difference(original, recompressed)
    stat = np.array(ela_diff).mean() # Score ELA

    # 3. Detector de Parches Rectangulares (Marcado Rojo)
    blur = cv2.GaussianBlur(img_cv, (5,5), 0)
    diff_local = cv2.absdiff(img_cv, blur)
    _, mask = cv2.threshold(diff_local, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_result = cv2.imread(image_path)
    parches_encontrados = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        # Detectamos rectángulos sospechosos (típicos de cambios en montos)
        if 400 < area < 6000 and 1.2 < aspect_ratio < 8.0:
            cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 0, 255), 2)
            parches_encontrados += 1

    return img_result, parches_encontrados > 0, score_ruido, stat

# ═══════════════════════════════════════════════════════════════
# HANDLER Y DICTAMEN DETALLADO
# ═══════════════════════════════════════════════════════════════
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(uuid.uuid4())[:8]
    in_p, out_p = f"in_{uid}.jpg", f"res_{uid}.jpg"
    
    try:
        f = await update.message.photo[-1].get_file()
        await f.download_to_drive(in_p)
        await update.message.reply_text("🔬 *Iniciando Escaneo Forense Digital...*", parse_mode="Markdown")
        
        res_img, detectado, s_ruido, s_ela = micro_analysis(in_p)
        cv2.imwrite(out_p, res_img)
        
        # Lógica de Scoring para el Dictamen
        puntos = 0
        hallazgos = []
        
        if detectado:
            puntos += 6
            hallazgos.append("🔴 **Parche Detectado**: Se hallaron rectángulos con textura artificial (marcados en rojo).")
        if s_ruido < 130:
            puntos += 3
            hallazgos.append(f"🟡 **Bajo Ruido ({s_ruido:.1f})**: El papel es demasiado liso, sugiere edición digital.")
        if s_ela > 5.0:
            puntos += 1
            hallazgos.append(f"🟡 **Inconsistencia ELA**: Compresión JPEG no uniforme.")

        risk_score = min(puntos, 10)
        veredicto = "🔴 RIESGO ALTO — Probable Alteración" if risk_score >= 6 else "🟢 CONFIABLE — Sin anomalías claras"
        
        # Enviamos la imagen con los cuadros rojos
        await update.message.reply_photo(photo=open(out_p, "rb"))
        
        # Enviamos el DICTAMEN DETALLADO
        barra = "█" * risk_score + "░" * (10 - risk_score)
        reporte = (
            f"📊 *DICTAMEN FORENSE DIGITAL*\n"
            f"{'─' * 30}\n"
            f"*Veredicto:* {veredicto}\n"
            f"*Riesgo:* `[{barra}]` {risk_score}/10\n\n"
            f"*Hallazgos Técnicos:*\n"
            f"{chr(10).join(hallazgos) if hallazgos else '✅ No se detectan parches ni anomalías de textura.'}\n\n"
            f"{'─' * 30}\n"
            f"⚠️ _Este análisis es una herramienta de apoyo, no sustituye el ojo humano._"
        )
        
        await update.message.reply_text(reporte, parse_mode=constants.ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("❌ Error técnico. Intente con otra imagen.")
    finally:
        for p in [in_p, out_p]:
            if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
