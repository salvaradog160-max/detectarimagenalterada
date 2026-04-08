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
from PIL import Image, ImageChops, ImageEnhance, ImageStat, ImageDraw
from scipy import ndimage
from skimage.util import img_as_float

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
TOKEN = "8699029540:AAE9TGMSC5fvW2Fldhuc_keYQAYxM_ooW_s"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

web_app = Flask(__name__)
@web_app.route("/")
def health_check(): return "Bot Forense Activo ✅", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)

# ═══════════════════════════════════════════════════════════════
# MÓDULO 1 — ELA MEJORADO
# ═══════════════════════════════════════════════════════════════
def ela_analysis(image_path: str):
    original = Image.open(image_path).convert("RGB")
    # Redimensionar si es muy grande para ahorrar CPU
    if max(original.size) > 1500:
        original.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
    
    scores = []
    ela_acumulado = None

    for quality in [75, 90]: # Dos pasadas son suficientes y más rápidas
        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        diff = ImageChops.difference(original, recompressed)
        arr = np.array(diff, dtype=np.float32)
        scores.append(arr.mean())
        if ela_acumulado is None: ela_acumulado = arr
        else: ela_acumulado = np.maximum(ela_acumulado, arr)

    ela_score = float(np.mean(scores))
    ela_norm = np.clip(ela_acumulado * (255.0 / (np.percentile(ela_acumulado, 99) + 1)), 0, 255).astype(np.uint8)
    
    gray = cv2.cvtColor(ela_norm, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return Image.fromarray(ela_norm), ela_score, heatmap

# ═══════════════════════════════════════════════════════════════
# MÓDULO 3 — CLONADO (Optimizado para no colapsar la RAM)
# ═══════════════════════════════════════════════════════════════
def clone_detection(image_path: str, block_size: int = 32):
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None: return 0.0, None
    
    # Reducir imagen para el proceso de clonado (pesadísimo en RAM)
    scale = 0.5
    img_small = cv2.resize(img_cv, (0,0), fx=scale, fy=scale)
    h, w = img_small.shape
    blocks = {}
    clone_pairs = []

    for y in range(0, h - block_size, 8): # Step de 8 para mayor velocidad
        for x in range(0, w - block_size, 8):
            block = img_small[y:y + block_size, x:x + block_size].astype(np.float32)
            if block.shape != (block_size, block_size): continue
            dct = cv2.dct(block)
            signature = tuple(np.round(dct[:4, :4].flatten(), 0)) # Firma más robusta
            if signature in blocks:
                prev_x, prev_y = blocks[signature]
                if np.sqrt((x - prev_x)**2 + (y - prev_y)**2) > block_size:
                    clone_pairs.append(((int(prev_x/scale), int(prev_y/scale)), (int(x/scale), int(y/scale))))
            else:
                blocks[signature] = (x, y)

    clone_score = min(len(clone_pairs) / 5.0, 10.0)
    marked_image = None
    if clone_pairs:
        img_color = cv2.imread(image_path)
        for (x1, y1), (x2, y2) in clone_pairs[:15]:
            cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img_color, (x1, y1), 5, (0, 0, 255), -1)
        marked_image = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

    return clone_score, marked_image

# (Mantener Módulos 2, 4 y 5 igual, son sólidos)
# ... [Insertar aquí tus funciones jpeg_ghost_analysis, noise_inconsistency_analysis y exif_analysis] ...

# ═══════════════════════════════════════════════════════════════
# MANEJADOR DE IMÁGENES (Con limpieza garantizada)
# ═══════════════════════════════════════════════════════════════
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(uuid.uuid4())[:8]
    paths = {
        "in": f"in_{uid}.jpg", "ela": f"ela_{uid}.jpg",
        "heat": f"heat_{uid}.jpg", "clone": f"clone_{uid}.jpg"
    }

    try:
        file = await update.message.photo[-1].get_file()
        await file.download_to_drive(paths["in"])
        status_msg = await update.message.reply_text("🔬 *Iniciando Análisis Multiespectral...*", parse_mode="Markdown")

        # Ejecución
        ela_pil, ela_score, heatmap_cv = ela_analysis(paths["in"])
        clone_score, clone_img = clone_detection(paths["in"])
        # (Llamar a los otros módulos aquí...)

        # Enviar resultados
        ela_pil.save(paths["ela"])
        cv2.imwrite(paths["heat"], heatmap_cv)
        
        await update.message.reply_photo(photo=open(paths["ela"], "rb"), caption="🖼️ Mapa ELA (Compresión)")
        await update.message.reply_photo(photo=open(paths["heat"], "rb"), caption="🌡️ Heatmap ELA (Sospecha)")

        if clone_img:
            clone_img.save(paths["clone"])
            await update.message.reply_photo(photo=open(paths["clone"], "rb"), caption="🔁 Detección de Clonado")

        # Dictamen Final (Usar tu lógica de scoring)
        await update.message.reply_text("✅ Análisis finalizado. Revisa los mapas visuales.")

    except Exception as e:
        logging.error(f"Error: {e}")
        await update.message.reply_text("❌ Error técnico en el análisis.")
    finally:
        for p in paths.values():
            if os.path.exists(p): os.remove(p)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
