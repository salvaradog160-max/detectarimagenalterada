import os
import io
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
logging.basicConfig(level=logging.INFO)

web_app = Flask(__name__)
@web_app.route("/")
def health(): return "Analizador Forense 4.9 Activo ✅", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)

# ═══════════════════════════════════════════════════════════════
# MÓDULOS DE ANÁLISIS
# ═══════════════════════════════════════════════════════════════

def get_ela(image_path):
    original = Image.open(image_path).convert('RGB')
    buf = io.BytesIO()
    original.save(buf, format='JPEG', quality=90)
    recompressed = Image.open(io.BytesIO(buf.getvalue()))
    ela_diff = ImageChops.difference(original, recompressed)
    
    # Calcular Score
    stat = ImageStat.Stat(ela_diff)
    score = float(sum(stat.mean) / 3)
    
    # Brillo Dinámico para que NO salga negro
    extrema = ela_diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    ela_final = ImageEnhance.Brightness(ela_diff).enhance(255.0 / max_diff)
    
    return score, ela_final

def get_noise_cv(image_path):
    img = cv2.imread(image_path, 0)
    if img is None: return 0.0
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return float(np.std(laplacian) / (np.mean(np.abs(laplacian)) + 1e-6))

def get_exif(image_path):
    hallazgos = []
    puntos = 0
    try:
        exif_dict = piexif.load(image_path)
        soft = str(exif_dict.get("0th", {}).get(piexif.ImageIFD.Software, b"")).lower()
        if any(x in soft for x in ["photoshop", "canva", "picsart", "adobe"]):
            hallazgos.append(f"🔴 Software de edición: {soft}")
            puntos += 4
        else:
            hallazgos.append("✅ Sin software de edición en metadatos.")
    except:
        hallazgos.append("🟡 Sin metadatos EXIF (posible screenshot)")
        puntos += 1
    return puntos, hallazgos

# ═══════════════════════════════════════════════════════════════
# HANDLER PRINCIPAL
# ═══════════════════════════════════════════════════════════════

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(uuid.uuid4())[:8]
    in_p, ela_p = f"in_{uid}.jpg", f"ela_{uid}.jpg"
    
    try:
        f = await update.message.photo[-1].get_file()
        await f.download_to_drive(in_p)
        await update.message.reply_text("🔬 *Iniciando escaneo forense digital...*", parse_mode="Markdown")
        
        # Ejecutar Módulos
        ela_score, ela_img = get_ela(in_p)
        noise_cv = get_noise_cv(in_p)
        exif_pts, exif_h = get_exif(in_p)
        
        # Guardar y enviar imagen ELA
        ela_img.save(ela_p)
        with open(ela_p, "rb") as photo:
            await update.message.reply_photo(photo=photo, caption="🖼️ Mapa ELA (Compresión)")

        # Lógica de Scoring
        risk = 0.0
        detalles = []
        
        if ela_score > 5.0:
            risk += 4.0
            detalles.append(f"🔴 **ELA Crítico** ({ela_score:.2f}): Zonas editadas.")
        else:
            detalles.append(f"🟢 ELA normal ({ela_score:.2f}): Compresión uniforme.")

        if noise_cv > 0.8:
            risk += 2.0
            detalles.append(f"🔴 **Ruido Inconsistente** (CV={noise_cv:.2f})")
        else:
            detalles.append(f"🟢 Ruido uniforme (CV={noise_cv:.2f})")
            
        risk = min(risk + exif_pts, 10.0)
        
        # Veredicto
        if risk >= 6.0: titulo = "🔴 RIESGO ALTO — Probable Alteración"
        elif risk >= 3.0: titulo = "🟡 RIESGO MODERADO — Revisión humana"
        else: titulo = "🟢 CONFIABLE — Sin anomalías"

        # CONSTRUCCIÓN DEL MENSAJE FINAL (DICTAMEN)
        barra = "█" * int(risk) + "░" * (10 - int(risk))
        reporte = (
            f"📊 *DICTAMEN FORENSE DIGITAL*\n"
            f"{'─' * 32}\n\n"
            f"*Veredicto:* {titulo}\n"
            f"*Riesgo:* `[{barra}]` {risk:.1f}/10\n"
            f"_Análisis por micro-textura y metadatos._\n\n"
            f"*Análisis Detallado:*\n"
            f"{chr(10).join(detalles)}\n\n"
            f"*📋 Metadatos EXIF:*\n"
            f"{chr(10).join(['  ' + h for h in exif_h])}\n\n"
            f"{'─' * 32}\n"
            f"⚠️ _Este análisis es una herramienta de apoyo, no sustituye el ojo humano ni el criterio del analista._"
        )
        
        await update.message.reply_text(reporte, parse_mode=constants.ParseMode.MARKDOWN)

    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text(f"❌ Error al procesar: {str(e)}")
    finally:
        for p in [in_p, ela_p]:
            if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
