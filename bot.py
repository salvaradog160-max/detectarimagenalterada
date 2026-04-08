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
from PIL import Image, ImageChops

# --- CONFIGURACIÓN ---
TOKEN = "8699029540:AAE9TGMSC5fvW2Fldhuc_keYQAYxM_ooW_s"
logging.basicConfig(level=logging.INFO)

web_app = Flask(__name__)
@web_app.route("/")
def health(): return "Analizador Forense 4.7 Activo ✅", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)

# ═══════════════════════════════════════════════════════════════
# MOTOR DE ANÁLISIS (Detección de Parches y Ruido)
# ═══════════════════════════════════════════════════════════════
def perform_analysis(image_path):
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None: return None, False, 0, 0
    
    # 1. Medición de Ruido (Laplacian)
    laplacian = cv2.Laplacian(img_cv, cv2.CV_64F)
    score_ruido = float(np.var(laplacian))
    
    # 2. Análisis ELA (Compresión)
    original = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=90)
    recompressed = Image.open(io.BytesIO(buf.getvalue()))
    ela_diff = ImageChops.difference(original, recompressed)
    score_ela = float(np.array(ela_diff).mean())

    # 3. Detector de Parches Visuales
    img_result = cv2.imread(image_path)
    blur = cv2.GaussianBlur(img_cv, (5,5), 0)
    diff_local = cv2.absdiff(img_cv, blur)
    _, mask = cv2.threshold(diff_local, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    parche_detectado = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(w)/h
        # Filtro para detectar rectángulos de texto alterado
        if 400 < area < 8000 and 1.1 < ratio < 10.0:
            cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 0, 255), 3)
            parche_detectado = True

    return img_result, parche_detectado, score_ruido, score_ela

# ═══════════════════════════════════════════════════════════════
# HANDLER PRINCIPAL (Garantiza el Dictamen)
# ═══════════════════════════════════════════════════════════════
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(uuid.uuid4())[:8]
    in_file, out_file = f"in_{uid}.jpg", f"res_{uid}.jpg"
    
    try:
        # Descarga
        f = await update.message.photo[-1].get_file()
        await f.download_to_drive(in_file)
        await update.message.reply_text("🔬 *Iniciando escaneo forense digital...*", parse_mode="Markdown")
        
        # Procesamiento
        res_img, detectado, s_ruido, s_ela = perform_analysis(in_file)
        cv2.imwrite(out_file, res_img)
        
        # Cálculo de Riesgo
        puntos = 0
        hallazgos = []
        if detectado:
            puntos += 6
            hallazgos.append("🔴 **Parche Detectado**: Se hallaron rectángulos con textura artificial (marcados en rojo).")
        if s_ruido < 135:
            puntos += 3
            hallazgos.append(f"🟡 **Bajo Ruido ({s_ruido:.1f})**: El papel es demasiado liso, sugiere edición digital.")
        if s_ela > 5.5:
            puntos += 1
            hallazgos.append("🟡 **Inconsistencia ELA**: Compresión JPEG no uniforme.")

        risk_score = min(puntos, 10)
        veredicto = "🔴 RIESGO ALTO — Probable Alteración" if risk_score >= 6 else "🟢 CONFIABLE — Sin anomalías claras"
        barra = "█" * risk_score + "░" * (10 - risk_score)

        # 1. Enviar imagen marcada
        with open(out_file, "rb") as photo:
            await update.message.reply_photo(photo=photo, caption="📸 Imagen analizada con marcadores de riesgo.")
        
        # 2. Enviar el DICTAMEN (Formato corregido)
        reporte = (
            f"📊 *DICTAMEN FORENSE DIGITAL*\n"
            f"──────────────────────────────\n"
            f"*Veredicto:* {veredicto}\n"
            f"*Riesgo:* `[{barra}]` {risk_score}/10\n\n"
            f"*Hallazgos Técnicos:*\n"
            f"{chr(10).join(hallazgos) if hallazgos else '✅ No se detectan anomalías de textura.'}\n\n"
            f"──────────────────────────────\n"
            f"⚠️ _Este análisis es una herramienta de apoyo, no sustituye el ojo humano._"
        )
        
        await update.message.reply_text(reporte, parse_mode=constants.ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Error técnico: {e}")
        await update.message.reply_text("❌ Error al procesar la imagen. Intente de nuevo.")
    finally:
        # Borrado garantizado de archivos
        for p in [in_file, out_file]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
