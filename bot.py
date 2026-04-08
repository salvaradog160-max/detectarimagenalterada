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
from PIL import Image, ImageChops, ImageEnhance, ImageStat, ImageDraw, ImageFont
from scipy import ndimage
from skimage.filters import sobel
from skimage.util import img_as_float

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
TOKEN = os.environ.get("TELEGRAM_TOKEN", "TU_TOKEN_AQUI")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ─────────────────────────────────────────
# WEB SERVER PARA RENDER (HEALTH CHECK)
# ─────────────────────────────────────────
web_app = Flask(__name__)

@web_app.route("/")
def health_check():
    return "Bot Forense Activo ✅", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    web_app.run(host="0.0.0.0", port=port)


# ═══════════════════════════════════════════════════════════════
# MÓDULO 1 — ELA (Error Level Analysis) MEJORADO
# Detecta: recompresión selectiva, zonas pegadas de otra imagen
# ═══════════════════════════════════════════════════════════════
def ela_analysis(image_path: str):
    """
    ELA con múltiples calidades para reducir falsos positivos.
    Retorna (imagen_ela_PIL, score_ela, mapa_heatmap_cv2).
    """
    original = Image.open(image_path).convert("RGB")
    scores = []
    ela_acumulado = None

    for quality in [75, 80, 85]:
        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        diff = ImageChops.difference(original, recompressed)

        arr = np.array(diff, dtype=np.float32)
        scores.append(arr.mean())

        if ela_acumulado is None:
            ela_acumulado = arr
        else:
            ela_acumulado = np.maximum(ela_acumulado, arr)

    ela_score = float(np.mean(scores))

    # Normalizar para visualización
    ela_norm = np.clip(ela_acumulado * (255.0 / (ela_acumulado.max() + 1e-6)), 0, 255).astype(np.uint8)
    ela_pil = Image.fromarray(ela_norm)

    # Heatmap en color para facilitar lectura humana
    gray = cv2.cvtColor(ela_norm, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    return ela_pil, ela_score, heatmap


# ═══════════════════════════════════════════════════════════════
# MÓDULO 2 — GHOST JPEG (Doble Compresión)
# Detecta: imagen guardada múltiples veces → indicio de edición
# ═══════════════════════════════════════════════════════════════
def jpeg_ghost_analysis(image_path: str):
    """
    Compara la imagen recomprimida a distintas calidades.
    Si alguna calidad produce diferencia MUY baja, hubo doble compresión.
    Retorna (ghost_score: float, calidad_sospechosa: int | None).
    """
    original = Image.open(image_path).convert("RGB")
    original_arr = np.array(original, dtype=np.float32)
    min_diff = float("inf")
    best_quality = None

    for q in range(50, 96, 5):
        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        test = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
        diff = np.mean(np.abs(original_arr - test))
        if diff < min_diff:
            min_diff = diff
            best_quality = q

    # Si la diferencia mínima es muy baja en alguna calidad media,
    # la imagen ya fue guardada antes a esa calidad.
    ghost_score = min_diff
    suspicious_quality = best_quality if min_diff < 1.5 else None
    return ghost_score, suspicious_quality


# ═══════════════════════════════════════════════════════════════
# MÓDULO 3 — DETECCIÓN DE REGIONES CLONADAS (Copy-Move)
# Detecta: zonas copiadas y pegadas dentro de la misma imagen
# ═══════════════════════════════════════════════════════════════
def clone_detection(image_path: str, block_size: int = 16):
    """
    Divide la imagen en bloques y compara sus hashes DCT.
    Si dos bloques son casi idénticos en distintas posiciones → clonado.
    Retorna (clone_score: float, imagen_marcada_PIL | None).
    """
    img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        return 0.0, None

    h, w = img_cv.shape
    blocks = {}
    clone_pairs = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = img_cv[y:y + block_size, x:x + block_size].astype(np.float32)
            dct = cv2.dct(block)
            # Usar solo las 6 primeras frecuencias bajas como firma
            signature = tuple(np.round(dct[:3, :3].flatten(), 1))
            if signature in blocks:
                prev_x, prev_y = blocks[signature]
                dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                if dist > block_size * 2:  # Ignorar bloques adyacentes
                    clone_pairs.append(((prev_x, prev_y), (x, y)))
            else:
                blocks[signature] = (x, y)

    clone_score = min(len(clone_pairs) / 10.0, 10.0)  # Normalizado 0-10

    marked_image = None
    if clone_pairs:
        img_color = cv2.imread(image_path)
        for (x1, y1), (x2, y2) in clone_pairs[:20]:  # Marcar máx 20 pares
            cv2.rectangle(img_color, (x1, y1), (x1 + block_size, y1 + block_size), (0, 0, 255), 1)
            cv2.rectangle(img_color, (x2, y2), (x2 + block_size, y2 + block_size), (255, 0, 0), 1)
            cv2.line(img_color, (x1, y1), (x2, y2), (0, 255, 0), 1)
        marked_image = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

    return clone_score, marked_image


# ═══════════════════════════════════════════════════════════════
# MÓDULO 4 — ANÁLISIS DE RUIDO DEL SENSOR
# Detecta: inconsistencia de ruido entre zonas (pegado de partes)
# ═══════════════════════════════════════════════════════════════
def noise_inconsistency_analysis(image_path: str):
    """
    Divide la imagen en bloques y mide el nivel de ruido en cada uno.
    Inconsistencias grandes indican regiones de diferente origen.
    Retorna (inconsistency_score: float, descripcion: str).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img_float = img_as_float(img.astype(np.uint8))

    # Extraer ruido restando versión suavizada
    blurred = ndimage.gaussian_filter(img, sigma=2)
    noise_map = img - blurred

    # Dividir en bloques de 32x32 y calcular desviación estándar local
    h, w = noise_map.shape
    block = 32
    stds = []
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            region = noise_map[y:y + block, x:x + block]
            stds.append(float(np.std(region)))

    if not stds:
        return 0.0, "Imagen demasiado pequeña para análisis de ruido."

    std_array = np.array(stds)
    mean_std = np.mean(std_array)
    # Coeficiente de variación: qué tan dispares son los niveles de ruido
    cv = float(np.std(std_array) / (mean_std + 1e-6))

    if cv > 0.8:
        desc = "Alta inconsistencia de ruido entre zonas (posible composición)"
    elif cv > 0.4:
        desc = "Inconsistencia moderada de ruido"
    else:
        desc = "Ruido uniforme en toda la imagen"

    return cv, desc


# ═══════════════════════════════════════════════════════════════
# MÓDULO 5 — METADATOS EXIF
# Detecta: software de edición, fechas inconsistentes, GPS ausente
# ═══════════════════════════════════════════════════════════════
def exif_analysis(image_path: str):
    """
    Lee metadatos EXIF y busca señales de edición.
    Retorna (exif_score: int, hallazgos: list[str]).
    """
    hallazgos = []
    exif_score = 0

    try:
        exif_data = piexif.load(image_path)
    except Exception:
        hallazgos.append("⚠️ Sin metadatos EXIF (imagen posiblemente procesada o screenshot)")
        return 2, hallazgos

    # Software de edición
    image_ifd = exif_data.get("0th", {})
    software = image_ifd.get(piexif.ImageIFD.Software, b"")
    if isinstance(software, bytes):
        software = software.decode("utf-8", errors="ignore").strip()
    if software:
        editores = ["photoshop", "gimp", "lightroom", "affinity", "canva",
                    "pixelmator", "paint.net", "snapseed", "facetune"]
        if any(e in software.lower() for e in editores):
            hallazgos.append(f"🔴 Software de edición detectado: **{software}**")
            exif_score += 4
        else:
            hallazgos.append(f"ℹ️ Software: {software}")

    # Fechas: creación vs modificación
    dt_original = exif_data.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal, b"")
    dt_modified = image_ifd.get(piexif.ImageIFD.DateTime, b"")
    if isinstance(dt_original, bytes): dt_original = dt_original.decode("utf-8", errors="ignore")
    if isinstance(dt_modified, bytes): dt_modified = dt_modified.decode("utf-8", errors="ignore")

    if dt_original and dt_modified and dt_original != dt_modified:
        hallazgos.append(f"🟡 Fecha de captura ({dt_original}) ≠ fecha de modificación ({dt_modified})")
        exif_score += 2

    # Fabricante / modelo de cámara
    make = image_ifd.get(piexif.ImageIFD.Make, b"")
    model = image_ifd.get(piexif.ImageIFD.Model, b"")
    if isinstance(make, bytes): make = make.decode("utf-8", errors="ignore")
    if isinstance(model, bytes): model = model.decode("utf-8", errors="ignore")
    if not make and not model:
        hallazgos.append("🟡 Sin datos de cámara/dispositivo")
        exif_score += 1
    else:
        hallazgos.append(f"ℹ️ Dispositivo: {make} {model}".strip())

    # GPS
    gps_data = exif_data.get("GPS", {})
    if not gps_data:
        hallazgos.append("ℹ️ Sin datos GPS (normal en imágenes escaneadas o sin ubicación)")

    if not hallazgos:
        hallazgos.append("✅ Metadatos EXIF sin anomalías detectadas")

    return exif_score, hallazgos


# ═══════════════════════════════════════════════════════════════
# MOTOR DE DECISIÓN — Sistema de scoring ponderado
# ═══════════════════════════════════════════════════════════════
def get_verdict(ela_score, ghost_score, ghost_quality, clone_score,
                noise_cv, exif_score, exif_hallazgos):
    """
    Combina todos los módulos en un dictamen final con puntuación 0-10.
    """
    risk = 0.0
    detalles = []

    # — ELA —
    # Umbral calibrado: imágenes auténticas suelen tener ELA < 8
    if ela_score > 15.0:
        risk += 3.0
        detalles.append("🔴 **ELA Crítico**: Diferencias de compresión muy altas → zonas editadas.")
    elif ela_score > 8.0:
        risk += 1.5
        detalles.append(f"🟡 **ELA Moderado** ({ela_score:.2f}): Posible edición parcial.")
    else:
        detalles.append(f"🟢 ELA normal ({ela_score:.2f}): Compresión uniforme.")

    # — Ghost JPEG —
    if ghost_quality is not None:
        risk += 2.0
        detalles.append(f"🔴 **Doble Compresión Detectada** (calidad ~{ghost_quality}%): La imagen fue guardada/editada al menos dos veces.")
    else:
        detalles.append(f"🟢 Sin evidencia de doble compresión JPEG.")

    # — Clone Detection —
    if clone_score >= 3.0:
        risk += 3.0
        detalles.append(f"🔴 **Clonado Detectado**: {int(clone_score * 10)} bloques copiados internamente.")
    elif clone_score >= 1.0:
        risk += 1.5
        detalles.append(f"🟡 Posible clonado leve ({int(clone_score * 10)} coincidencias).")
    else:
        detalles.append("🟢 Sin regiones clonadas detectadas.")

    # — Ruido del sensor —
    if noise_cv > 0.8:
        risk += 2.0
        detalles.append(f"🔴 **Ruido Inconsistente** (CV={noise_cv:.2f}): Diferentes zonas tienen distintos niveles de grano → composición probable.")
    elif noise_cv > 0.4:
        risk += 1.0
        detalles.append(f"🟡 Ligera inconsistencia de ruido (CV={noise_cv:.2f}).")
    else:
        detalles.append(f"🟢 Ruido uniforme (CV={noise_cv:.2f}).")

    # — EXIF —
    risk += min(exif_score, 4.0)  # Máximo 4 puntos de EXIF
    detalles.append("\n📋 **Metadatos EXIF:**")
    detalles.extend([f"  {h}" for h in exif_hallazgos])

    # Score final normalizado 0-10
    risk_normalized = min(risk / 14.0 * 10.0, 10.0)

    if risk_normalized >= 6.0:
        titulo = "🔴 RIESGO ALTO — Probable falsificación"
        resumen = "Múltiples indicadores apuntan a manipulación digital."
    elif risk_normalized >= 3.0:
        titulo = "🟡 RIESGO MODERADO — Requiere revisión humana"
        resumen = "Se detectaron anomalías que pueden indicar edición."
    else:
        titulo = "🟢 CONFIABLE — Sin anomalías significativas"
        resumen = "La imagen no muestra señales claras de manipulación."

    return titulo, resumen, "\n".join(detalles), round(risk_normalized, 1)


# ═══════════════════════════════════════════════════════════════
# HANDLER PRINCIPAL DE TELEGRAM
# ═══════════════════════════════════════════════════════════════
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    task_id = f"{update.effective_chat.id}_{update.message.message_id}"
    input_path = f"in_{task_id}.jpg"
    ela_path = f"ela_{task_id}.jpg"
    heat_path = f"heat_{task_id}.jpg"
    clone_path = f"clone_{task_id}.jpg"

    try:
        # Descargar imagen
        file = await update.message.photo[-1].get_file()
        await file.download_to_drive(input_path)

        await update.message.reply_text(
            "🔬 *Análisis forense iniciado…*\n"
            "Ejecutando 5 módulos de detección. Un momento…",
            parse_mode=constants.ParseMode.MARKDOWN
        )

        # ── Ejecutar todos los módulos ──
        ela_pil, ela_score, heatmap_cv = ela_analysis(input_path)
        ghost_score, ghost_quality = jpeg_ghost_analysis(input_path)
        clone_score, clone_img = clone_detection(input_path)
        noise_cv, noise_desc = noise_inconsistency_analysis(input_path)
        exif_score, exif_hallazgos = exif_analysis(input_path)

        # ── Veredicto final ──
        titulo, resumen, detalles, risk_score = get_verdict(
            ela_score, ghost_score, ghost_quality,
            clone_score, noise_cv, exif_score, exif_hallazgos
        )

        # ── Enviar imágenes de análisis ──
        ela_pil.save(ela_path)
        await update.message.reply_photo(
            photo=open(ela_path, "rb"),
            caption=(
                "🖼️ *Análisis ELA — Mapa de errores de compresión*\n"
                "Zonas brillantes = mayor anomalía de compresión."
            ),
            parse_mode=constants.ParseMode.MARKDOWN
        )

        cv2.imwrite(heat_path, heatmap_cv)
        await update.message.reply_photo(
            photo=open(heat_path, "rb"),
            caption=(
                "🌡️ *Mapa de Calor ELA*\n"
                "Rojo/amarillo = zonas de alta sospecha."
            ),
            parse_mode=constants.ParseMode.MARKDOWN
        )

        if clone_img is not None:
            clone_img.save(clone_path)
            await update.message.reply_photo(
                photo=open(clone_path, "rb"),
                caption=(
                    "🔁 *Mapa de Regiones Clonadas*\n"
                    "Líneas verdes conectan bloques copiados. "
                    "Rectángulos rojos/azules = fuente/destino del clonado."
                ),
                parse_mode=constants.ParseMode.MARKDOWN
            )

        # ── Dictamen final ──
        barra = "█" * int(risk_score) + "░" * (10 - int(risk_score))
        mensaje_final = (
            f"📊 *DICTAMEN FORENSE DIGITAL*\n"
            f"{'─' * 32}\n\n"
            f"*Veredicto:* {titulo}\n"
            f"*Riesgo:* `[{barra}]` {risk_score}/10\n"
            f"_{resumen}_\n\n"
            f"*Análisis Detallado:*\n"
            f"{detalles}\n\n"
            f"{'─' * 32}\n"
            f"⚠️ _Este análisis es orientativo. El criterio humano es indispensable._"
        )

        await update.message.reply_text(
            mensaje_final,
            parse_mode=constants.ParseMode.MARKDOWN
        )

    except Exception as e:
        logging.error(f"Error procesando imagen [{task_id}]: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Ocurrió un error durante el análisis. "
            "Verifica que enviaste una imagen válida (no un archivo comprimido)."
        )
    finally:
        for path in [input_path, ela_path, heat_path, clone_path]:
            if os.path.exists(path):
                os.remove(path)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Bot Forense de Imágenes*\n\n"
        "Envíame una foto y analizaré si fue alterada digitalmente usando:\n\n"
        "🔬 *5 módulos de detección:*\n"
        "1. ELA — Errores de compresión JPEG\n"
        "2. Ghost JPEG — Doble compresión\n"
        "3. Copy-Move — Regiones clonadas\n"
        "4. Ruido del sensor — Inconsistencias\n"
        "5. Metadatos EXIF — Software de edición\n\n"
        "_Envía la imagen directamente como foto (no como archivo)._",
        parse_mode=constants.ParseMode.MARKDOWN
    )


# ─────────────────────────────────────────
# INICIO
# ─────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logging.info("🤖 Bot forense iniciado y escuchando...")
    app.run_polling()
