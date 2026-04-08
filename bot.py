"""
╔══════════════════════════════════════════════════════════════╗
║   BOT FORENSE DE DOCUMENTOS — Calibrado para comprobantes   ║
║   Detecta: montos sobrepuestos, parches, clonado, EXIF      ║
╚══════════════════════════════════════════════════════════════╝

CORRECCIONES PRINCIPALES vs versión anterior:
  - ELA: umbral bajado (documentos fotografiados tienen ELA muy bajo ~0.3-2.0)
  - Ruido CV: umbral bajado (CV>0.8 era demasiado alto, documentos llegan a CV=2+)
  - NUEVO: Detección de parches lisos (zona editada sobre papel texturizado)
  - NUEVO: Detección de inconsistencia de fuente/nitidez por zonas
  - NUEVO: Análisis de histograma local (zona con fondo uniforme = sospecha)
  - Scoring: el ruido inconsistente ahora SÍ suma riesgo correctamente
"""

import os
import io
import logging
import threading
import uuid

import numpy as np
import cv2
import piexif
from flask import Flask
from telegram import Update, constants
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from PIL import Image, ImageChops, ImageEnhance, ImageStat, ImageFilter

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────
TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
if not TOKEN:
    raise RuntimeError("❌ Define TELEGRAM_TOKEN en las variables de entorno de Render.")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

MAX_SIDE_PX  = 1400   # Proteger RAM en Render Free
MAX_CLONE_PX = 900

# ──────────────────────────────────────────────────────────────
# FLASK — health-check
# ──────────────────────────────────────────────────────────────
flask_app = Flask(__name__)

@flask_app.route("/")
def health():
    return "Bot Forense Activo ✅", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host="0.0.0.0", port=port)


# ══════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════

def _resize_if_needed(img_cv: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img_cv.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_cv

def _load_pil(path: str, max_side: int = MAX_SIDE_PX) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def _load_cv_gray(path: str, max_side: int = MAX_SIDE_PX) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("OpenCV no pudo leer la imagen.")
    return _resize_if_needed(img, max_side)

def _load_cv_color(path: str, max_side: int = MAX_SIDE_PX) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("OpenCV no pudo leer la imagen.")
    return _resize_if_needed(img, max_side)


# ══════════════════════════════════════════════════════════════
# MÓDULO 1 — ELA  (Error Level Analysis) multi-calidad
#
# PROBLEMA ANTERIOR: umbral > 5.0 era demasiado alto.
# Un comprobante bancario fotografiado con zona pegada
# produce ELA ~0.3–3.0 (muy bajo) porque la edición fue hecha
# antes de imprimir o porque el parche tiene compresión similar.
# SOLUCIÓN: umbrales rebajados + visualización amplificada x10.
# ══════════════════════════════════════════════════════════════
def mod_ela(path: str):
    """
    Retorna (score_float, imagen_ela_PIL).
    Umbrales corregidos:
      score > 3.0 → crítico   (antes era > 5.0, muy alto)
      score > 1.5 → moderado  (antes no existía este rango)
    """
    original = _load_pil(path)
    acum = None

    for q in (75, 82, 90):
        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        recomp = Image.open(buf).convert("RGB")
        diff = np.abs(
            np.array(original, dtype=np.float32) -
            np.array(recomp,   dtype=np.float32)
        )
        acum = diff if acum is None else np.maximum(acum, diff)

    score = float(acum.mean())

    # Amplificar x10 para visualización (documentos tienen diff muy baja)
    peak = acum.max() or 1.0
    vis  = np.clip(acum * (255.0 / peak), 0, 255).astype(np.uint8)
    ela_img = Image.fromarray(vis)

    return score, ela_img


# ══════════════════════════════════════════════════════════════
# MÓDULO 2 — DETECCIÓN DE PARCHES LISOS  ← NUEVO / CLAVE
#
# Qué detecta: zona con fondo uniformemente blanco/gris sobre
# papel con textura natural → típico de monto sobrepuesto digitalmente.
#
# Cómo funciona:
#   1. Calcula varianza local en bloques de 20x20
#   2. Bloques con varianza muy baja = zona "lisa" (sin grano de papel)
#   3. Si hay bloques lisos rodeados de bloques con textura → PARCHE
# ══════════════════════════════════════════════════════════════
def mod_patch_detection(path: str):
    """
    Retorna (patch_score 0-10, n_bloques_sospechosos, imagen_marcada_PIL|None).
    """
    gray = _load_cv_gray(path)
    h, w = gray.shape
    bs = 20  # bloque de 20x20 px

    var_map = []
    coords  = []

    for y in range(0, h - bs, bs):
        row_vars = []
        row_coords = []
        for x in range(0, w - bs, bs):
            block = gray[y:y+bs, x:x+bs].astype(np.float32)
            v = float(np.var(block))
            row_vars.append(v)
            row_coords.append((x, y))
        var_map.append(row_vars)
        coords.append(row_coords)

    if not var_map:
        return 0.0, 0, None

    flat_vars = [v for row in var_map for v in row]
    median_var = float(np.median(flat_vars))
    # Umbral: un bloque es "liso" si su varianza es menor al 8% de la mediana
    # (calibrado para papel con grano natural)
    SMOOTH_THRESHOLD = median_var * 0.08

    smooth_blocks = []
    for ry, row in enumerate(var_map):
        for rx, v in enumerate(row):
            if v < SMOOTH_THRESHOLD and median_var > 50:
                # Solo contar si la imagen tiene textura significativa
                x, y = coords[ry][rx]
                smooth_blocks.append((x, y))

    n = len(smooth_blocks)
    # Score: proporcional a la cantidad de bloques lisos vs total
    total_blocks = max(len(flat_vars), 1)
    ratio = n / total_blocks
    patch_score = min(ratio * 50, 10.0)  # 20% bloques lisos → score 10

    marked = None
    if n > 0:
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (x, y) in smooth_blocks[:200]:
            cv2.rectangle(img_color, (x, y), (x+bs, y+bs), (0, 80, 255), 1)
        marked = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

    return patch_score, n, marked


# ══════════════════════════════════════════════════════════════
# MÓDULO 3 — INCONSISTENCIA DE NITIDEZ POR ZONAS  ← NUEVO
#
# Qué detecta: texto/número insertado digitalmente suele ser
# más nítido que el texto impreso-fotografiado real.
# También detecta zonas con diferente nivel de desenfoque
# (típico cuando se pega un recorte de otra imagen).
# ══════════════════════════════════════════════════════════════
def mod_sharpness_map(path: str):
    """
    Retorna (sharpness_cv float, descripcion str).
    Calcula la nitidez local (varianza del Laplaciano) por bloques.
    CV alto → zonas con nitidez muy diferente entre sí → sospechoso.
    """
    gray = _load_cv_gray(path)
    h, w = gray.shape
    bs = 40

    lap_vars = []
    for y in range(0, h - bs, bs):
        for x in range(0, w - bs, bs):
            block = gray[y:y+bs, x:x+bs]
            lap   = cv2.Laplacian(block, cv2.CV_64F)
            lap_vars.append(float(np.var(lap)))

    if len(lap_vars) < 4:
        return 0.0, "Imagen demasiado pequeña."

    arr = np.array(lap_vars)
    cv  = float(arr.std() / (arr.mean() + 1e-6))

    if cv > 2.5:
        desc = "Nitidez muy inconsistente entre zonas (probable inserción digital)"
    elif cv > 1.5:
        desc = "Inconsistencia moderada de nitidez"
    else:
        desc = "Nitidez uniforme en el documento"

    return cv, desc


# ══════════════════════════════════════════════════════════════
# MÓDULO 4 — RUIDO DEL SENSOR (Corregido)
#
# PROBLEMA ANTERIOR: el CV=2.05 no sumaba riesgo porque el
# umbral era > 0.8 y la lógica no funcionaba con el scoring.
# SOLUCIÓN: umbrales ajustados, ahora CV > 0.5 ya es moderado.
# ══════════════════════════════════════════════════════════════
def mod_noise(path: str):
    """
    Retorna (cv float, descripcion str).
    CV alto = zonas con diferente nivel de ruido = composición probable.
    Umbrales calibrados para documentos fotografiados:
      CV > 1.2 → alto (antes era 0.8, demasiado permisivo)
      CV > 0.5 → moderado (antes era 0.4)
    """
    gray    = _load_cv_gray(path).astype(np.float64)
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0).astype(np.float64)
    noise   = gray - blurred

    bs = 32
    h, w = noise.shape
    stds = [
        float(np.std(noise[y:y+bs, x:x+bs]))
        for y in range(0, h - bs, bs)
        for x in range(0, w - bs, bs)
    ]

    if len(stds) < 4:
        return 0.0, "Imagen demasiado pequeña."

    arr = np.array(stds)
    cv  = float(arr.std() / (arr.mean() + 1e-6))

    if cv > 1.2:
        desc = "Ruido muy inconsistente entre zonas del documento"
    elif cv > 0.5:
        desc = "Inconsistencia moderada de ruido"
    else:
        desc = "Ruido uniforme"

    return cv, desc


# ══════════════════════════════════════════════════════════════
# MÓDULO 5 — JPEG GHOST (Doble compresión)
# ══════════════════════════════════════════════════════════════
def mod_jpeg_ghost(path: str):
    """Retorna (detectado bool, calidad_sospechosa int|None)."""
    original = _load_pil(path)
    orig_arr = np.array(original, dtype=np.float32)
    min_diff = float("inf")
    best_q   = None

    for q in range(55, 96, 5):
        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        test_arr = np.array(Image.open(buf).convert("RGB"), dtype=np.float32)
        diff = float(np.mean(np.abs(orig_arr - test_arr)))
        if diff < min_diff:
            min_diff, best_q = diff, q

    detected = min_diff < 2.0   # umbral ligeramente más permisivo
    return detected, (best_q if detected else None)


# ══════════════════════════════════════════════════════════════
# MÓDULO 6 — METADATOS EXIF
# ══════════════════════════════════════════════════════════════
EDITORES = [
    "photoshop", "gimp", "lightroom", "affinity", "canva",
    "picsart", "pixelmator", "snapseed", "paint.net",
    "facetune", "adobe", "meitu", "inshot",
]

def mod_exif(path: str):
    """Retorna (puntos_riesgo int, hallazgos list[str])."""
    hallazgos = []
    pts = 0

    try:
        exif = piexif.load(path)
    except Exception:
        hallazgos.append("🟡 Sin metadatos EXIF (screenshot o procesado externamente)")
        return 1, hallazgos

    ifd0  = exif.get("0th",  {})
    exifd = exif.get("Exif", {})

    def _d(raw) -> str:
        return raw.decode("utf-8", errors="ignore").strip() if isinstance(raw, bytes) else ""

    # Software
    software = _d(ifd0.get(piexif.ImageIFD.Software, b""))
    if software:
        if any(e in software.lower() for e in EDITORES):
            hallazgos.append(f"🔴 Software de edición detectado: **{software}**")
            pts += 4
        else:
            hallazgos.append(f"ℹ️ Software: {software}")
    else:
        hallazgos.append("ℹ️ Sin campo Software en EXIF")

    # Fechas
    dt_orig = _d(exifd.get(piexif.ExifIFD.DateTimeOriginal, b""))
    dt_mod  = _d(ifd0.get(piexif.ImageIFD.DateTime, b""))
    if dt_orig and dt_mod and dt_orig != dt_mod:
        hallazgos.append(
            f"🟡 Fechas inconsistentes:\n"
            f"    📅 Captura: {dt_orig}\n"
            f"    ✏️  Modificado: {dt_mod}"
        )
        pts += 2
    elif dt_orig:
        hallazgos.append(f"✅ Fecha: {dt_orig}")

    # Dispositivo
    make  = _d(ifd0.get(piexif.ImageIFD.Make,  b""))
    model = _d(ifd0.get(piexif.ImageIFD.Model, b""))
    if make or model:
        hallazgos.append(f"ℹ️ Dispositivo: {make} {model}".strip())
    else:
        hallazgos.append("🟡 Sin datos de cámara/dispositivo")
        pts += 1

    return pts, hallazgos


# ══════════════════════════════════════════════════════════════
# MOTOR DE VEREDICTO — Scoring ponderado 0-10
#
# Pesos calibrados para documentos físicos fotografiados:
#   - Parche liso:       hasta 4.0 pts  (señal más directa)
#   - Nitidez incons.:   hasta 2.5 pts
#   - Ruido incons.:     hasta 2.0 pts
#   - ELA:               hasta 2.0 pts
#   - Ghost JPEG:        hasta 2.0 pts
#   - EXIF:              hasta 1.5 pts
#   Total máximo:        14.0 pts → normalizado a 10
# ══════════════════════════════════════════════════════════════
def build_verdict(ela_score, ghost_ok, ghost_q,
                  patch_score, patch_n,
                  sharp_cv, noise_cv,
                  exif_pts, exif_h):

    risk    = 0.0
    details = []

    # 1. PARCHE LISO (señal más directa de monto sobrepuesto)
    if patch_score >= 4.0:
        risk += 4.0
        details.append(
            f"🔴 **PARCHE DETECTADO** ({patch_n} bloques lisos): "
            "Zona con fondo artificialmente liso sobre papel texturizado. "
            "Típico de monto o texto sobrepuesto digitalmente."
        )
    elif patch_score >= 1.5:
        risk += 2.0
        details.append(
            f"🟡 **Zona sospechosamente lisa** ({patch_n} bloques): "
            "Posible parche o borrado de contenido."
        )
    else:
        details.append("🟢 Sin zonas lisas anómalas sobre el papel.")

    # 2. NITIDEZ INCONSISTENTE
    if sharp_cv > 2.5:
        risk += 2.5
        details.append(
            f"🔴 **Nitidez muy inconsistente** (CV={sharp_cv:.2f}): "
            "Hay zonas con texto demasiado nítido respecto al resto del documento."
        )
    elif sharp_cv > 1.5:
        risk += 1.2
        details.append(f"🟡 Inconsistencia de nitidez moderada (CV={sharp_cv:.2f}).")
    else:
        details.append(f"🟢 Nitidez uniforme (CV={sharp_cv:.2f}).")

    # 3. RUIDO DEL SENSOR
    if noise_cv > 1.2:
        risk += 2.0
        details.append(
            f"🔴 **Ruido muy inconsistente** (CV={noise_cv:.2f}): "
            "Zonas con diferente granularidad → probable composición."
        )
    elif noise_cv > 0.5:
        risk += 1.0
        details.append(f"🟡 Ruido moderadamente inconsistente (CV={noise_cv:.2f}).")
    else:
        details.append(f"🟢 Ruido uniforme (CV={noise_cv:.2f}).")

    # 4. ELA — umbrales bajos calibrados para documentos
    if ela_score > 3.0:
        risk += 2.0
        details.append(f"🔴 **ELA crítico** ({ela_score:.2f}): compresión heterogénea.")
    elif ela_score > 1.5:
        risk += 1.0
        details.append(f"🟡 **ELA moderado** ({ela_score:.2f}): leve inconsistencia de compresión.")
    else:
        details.append(f"🟢 ELA normal ({ela_score:.2f}).")

    # 5. GHOST JPEG
    if ghost_ok:
        risk += 2.0
        details.append(
            f"🔴 **Doble compresión JPEG** (calidad ~{ghost_q}%): "
            "La imagen fue guardada/editada al menos dos veces."
        )
    else:
        details.append("🟢 Sin doble compresión JPEG.")

    # 6. EXIF
    risk += min(exif_pts * 0.375, 1.5)
    details.append("\n📋 **Metadatos EXIF:**")
    details += [f"  {h}" for h in exif_h]

    # Normalizar a 0-10  (máximo teórico ≈ 14)
    score_10 = round(min(risk / 14.0 * 10.0, 10.0), 1)

    if score_10 >= 6.0:
        veredicto = "🔴 RIESGO ALTO — Probable alteración"
        resumen   = "Múltiples indicadores señalan manipulación digital del documento."
    elif score_10 >= 3.0:
        veredicto = "🟡 RIESGO MODERADO — Revisión humana recomendada"
        resumen   = "Se detectaron anomalías compatibles con edición."
    else:
        veredicto = "🟢 SIN ANOMALÍAS DETECTADAS"
        resumen   = "El documento no muestra señales claras de manipulación."

    return veredicto, resumen, "\n".join(details), score_10


# ══════════════════════════════════════════════════════════════
# HANDLER PRINCIPAL
# ══════════════════════════════════════════════════════════════
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid      = str(uuid.uuid4())[:8]
    in_p     = f"in_{uid}.jpg"
    ela_p    = f"ela_{uid}.jpg"
    patch_p  = f"patch_{uid}.jpg"
    paths    = [in_p, ela_p, patch_p]

    try:
        file = await update.message.photo[-1].get_file()
        await file.download_to_drive(in_p)

        await update.message.reply_text(
            "🔬 *Análisis forense iniciado…*\n"
            "Ejecutando 6 módulos de detección. Un momento ⏳",
            parse_mode=constants.ParseMode.MARKDOWN,
        )

        # ── Ejecutar todos los módulos ────────────────────────
        ela_score,  ela_img              = mod_ela(in_p)
        ghost_ok,   ghost_q              = mod_jpeg_ghost(in_p)
        patch_score, patch_n, patch_img  = mod_patch_detection(in_p)
        sharp_cv,   _                    = mod_sharpness_map(in_p)
        noise_cv,   _                    = mod_noise(in_p)
        exif_pts,   exif_h               = mod_exif(in_p)

        veredicto, resumen, detalles, score_10 = build_verdict(
            ela_score, ghost_ok, ghost_q,
            patch_score, patch_n,
            sharp_cv, noise_cv,
            exif_pts, exif_h,
        )

        # ── Enviar mapa ELA ───────────────────────────────────
        ela_img.save(ela_p, "JPEG", quality=85)
        await update.message.reply_photo(
            photo=open(ela_p, "rb"),
            caption=(
                "🖼️ *Mapa ELA — Errores de Compresión*\n"
                "Zonas brillantes = anomalías de recompresión JPEG."
            ),
            parse_mode=constants.ParseMode.MARKDOWN,
        )

        # ── Enviar mapa de parches (si hay zonas sospechosas) ─
        if patch_img is not None and patch_n > 0:
            patch_img.save(patch_p, "JPEG", quality=85)
            await update.message.reply_photo(
                photo=open(patch_p, "rb"),
                caption=(
                    "🟠 *Mapa de Zonas Lisas (Parches)*\n"
                    "Rectángulos naranjas = bloques con fondo "
                    "artificialmente liso sobre papel texturizado."
                ),
                parse_mode=constants.ParseMode.MARKDOWN,
            )

        # ── Dictamen final ────────────────────────────────────
        barra   = "█" * int(score_10) + "░" * (10 - int(score_10))
        reporte = (
            f"📊 *DICTAMEN FORENSE DIGITAL*\n"
            f"{'─' * 32}\n\n"
            f"*Veredicto:* {veredicto}\n"
            f"*Riesgo:*  `[{barra}]`  {score_10}/10\n"
            f"_{resumen}_\n\n"
            f"*Análisis Detallado:*\n{detalles}\n\n"
            f"{'─' * 32}\n"
            f"⚠️ _Herramienta de apoyo. No sustituye el criterio del analista._"
        )
        await update.message.reply_text(
            reporte,
            parse_mode=constants.ParseMode.MARKDOWN,
        )

    except Exception as exc:
        log.error("Error procesando imagen", exc_info=True)
        await update.message.reply_text(
            f"❌ Error: `{exc}`\n"
            "Envía la imagen *directamente como foto* (no como archivo).",
            parse_mode=constants.ParseMode.MARKDOWN,
        )
    finally:
        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Bot Forense de Documentos*\n\n"
        "Envíame la foto del comprobante y lo analizaré con 6 módulos:\n\n"
        "1️⃣ *Parches lisos* — Zonas con fondo artificial sobre papel\n"
        "2️⃣ *Nitidez por zonas* — Texto demasiado nítido = inserción digital\n"
        "3️⃣ *Ruido del sensor* — Inconsistencias entre regiones\n"
        "4️⃣ *ELA* — Errores de compresión JPEG\n"
        "5️⃣ *JPEG Ghost* — Doble compresión (guardado tras edición)\n"
        "6️⃣ *Metadatos EXIF* — Software de edición y fechas\n\n"
        "📌 _Envía la imagen como *foto*, no como archivo adjunto._",
        parse_mode=constants.ParseMode.MARKDOWN,
    )


# ──────────────────────────────────────────────────────────────
# INICIO
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    log.info("🌐 Flask iniciado.")

    bot = ApplicationBuilder().token(TOKEN).build()
    bot.add_handler(MessageHandler(filters.PHOTO,                       handle_image))
    bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,     handle_text))

    log.info("🤖 Bot forense escuchando...")
    bot.run_polling()
