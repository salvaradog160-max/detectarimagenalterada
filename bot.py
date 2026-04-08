"""
╔══════════════════════════════════════════════════════════════╗
║        BOT FORENSE DE IMÁGENES — Optimizado Render Free      ║
║  RAM: <200MB  |  Sin GPU  |  5 módulos de detección          ║
╚══════════════════════════════════════════════════════════════╝
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
from PIL import Image, ImageChops, ImageEnhance, ImageStat

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN  (token SIEMPRE desde variable de entorno)
# ──────────────────────────────────────────────────────────────
TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
if not TOKEN:
    raise RuntimeError("❌ Define la variable de entorno TELEGRAM_TOKEN en Render.")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

# Límite de tamaño para no agotar RAM en Render Free (512 MB)
MAX_SIDE_PX   = 1200   # redimensiona si algún lado supera esto
MAX_CLONE_PX  = 800    # redimensiona solo para el módulo de clonado (pesado)
CLONE_BLOCK   = 16     # tamaño de bloque DCT para copy-move


# ──────────────────────────────────────────────────────────────
# FLASK — health-check para que Render no apague el servicio
# ──────────────────────────────────────────────────────────────
flask_app = Flask(__name__)

@flask_app.route("/")
def health():
    return "Analizador Forense Activo ✅", 200

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    flask_app.run(host="0.0.0.0", port=port)


# ══════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════

def _load_pil(path: str, max_side: int = MAX_SIDE_PX) -> Image.Image:
    """Abre y redimensiona si es necesario (protege RAM)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def _load_cv_gray(path: str, max_side: int = MAX_SIDE_PX) -> np.ndarray:
    """Abre en escala de grises y redimensiona."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo leer la imagen con OpenCV.")
    h, w = img.shape
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


# ══════════════════════════════════════════════════════════════
# MÓDULO 1 — ELA  (Error Level Analysis)
# Qué detecta: zonas recomprimidas a distinta calidad → edición
# Mejora vs v4.9: multi-calidad 75/82/90, amplificación dinámica
# ══════════════════════════════════════════════════════════════
def mod_ela(path: str):
    """
    Retorna (score, imagen_ela_amplificada).
    Score alto → compresión heterogénea → sospecha de edición.
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

    # Amplificación visual dinámica (nunca negro)
    peak  = acum.max() or 1.0
    vis   = np.clip(acum * (200.0 / peak), 0, 255).astype(np.uint8)
    ela_img = Image.fromarray(vis)

    return score, ela_img


# ══════════════════════════════════════════════════════════════
# MÓDULO 2 — JPEG GHOST (Doble compresión)
# Qué detecta: imagen ya guardada antes → fue editada y re-guardada
# ══════════════════════════════════════════════════════════════
def mod_jpeg_ghost(path: str):
    """
    Retorna (doble_compresion_detectada, calidad_sospechosa|None).
    Busca la calidad donde recomprimir produce diferencia mínima.
    """
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

    # Umbral empírico: diferencia < 1.8 → calidad ya usada antes
    detected = min_diff < 1.8
    return detected, (best_q if detected else None)


# ══════════════════════════════════════════════════════════════
# MÓDULO 3 — COPY-MOVE (Regiones clonadas)
# Qué detecta: zonas copiadas/pegadas dentro de la misma imagen
# Optimizado: trabaja a MAX_CLONE_PX para no agotar RAM
# ══════════════════════════════════════════════════════════════
def mod_clone(path: str):
    """
    Retorna (n_pares_clonados, imagen_marcada|None).
    Usa firmas DCT de bloques para encontrar duplicados.
    """
    img = _load_cv_gray(path, max_side=MAX_CLONE_PX)
    h, w = img.shape
    bs = CLONE_BLOCK

    sigs  = {}
    pairs = []

    for y in range(0, h - bs, bs):
        for x in range(0, w - bs, bs):
            block = img[y:y+bs, x:x+bs].astype(np.float32)
            dct   = cv2.dct(block)
            sig   = tuple(np.round(dct[:3, :3].flatten(), 0).astype(int))
            if sig in sigs:
                px, py = sigs[sig]
                dist = ((x - px)**2 + (y - py)**2) ** 0.5
                if dist > bs * 3:
                    pairs.append(((px, py), (x, y)))
            else:
                sigs[sig] = (x, y)

    if not pairs:
        return 0, None

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in pairs[:30]:
        cv2.rectangle(img_color, (x1, y1), (x1+bs, y1+bs), (0, 0, 220), 1)
        cv2.rectangle(img_color, (x2, y2), (x2+bs, y2+bs), (220, 0, 0), 1)
        cv2.line(img_color,
                 (x1+bs//2, y1+bs//2),
                 (x2+bs//2, y2+bs//2), (0, 200, 0), 1)

    marked = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    return len(pairs), marked


# ══════════════════════════════════════════════════════════════
# MÓDULO 4 — RUIDO DEL SENSOR (Inconsistencia por bloques)
# Qué detecta: zonas pegadas de otra imagen (diferente ruido)
# Mejor que v4.9: mide VARIACIÓN ENTRE BLOQUES, no global
# ══════════════════════════════════════════════════════════════
def mod_noise(path: str):
    """
    Retorna (coef_variacion, descripcion).
    CV alto → ruido inconsistente entre zonas → composición probable.
    """
    img     = _load_cv_gray(path).astype(np.float64)
    blurred = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 0).astype(np.float64)
    noise   = img - blurred

    bs = 32
    h, w = noise.shape
    stds = [
        float(np.std(noise[y:y+bs, x:x+bs]))
        for y in range(0, h - bs, bs)
        for x in range(0, w - bs, bs)
    ]

    if len(stds) < 4:
        return 0.0, "Imagen demasiado pequeña para análisis de ruido."

    arr  = np.array(stds)
    cv   = float(arr.std() / (arr.mean() + 1e-6))

    if cv > 0.80:
        desc = "Alta inconsistencia de ruido entre zonas"
    elif cv > 0.40:
        desc = "Inconsistencia moderada de ruido"
    else:
        desc = "Ruido uniforme en toda la imagen"

    return cv, desc


# ══════════════════════════════════════════════════════════════
# MÓDULO 5 — METADATOS EXIF
# Qué detecta: software de edición, fechas inconsistentes
# ══════════════════════════════════════════════════════════════
EDITORES = [
    "photoshop", "gimp", "lightroom", "affinity",
    "canva", "picsart", "pixelmator", "snapseed",
    "paint.net", "facetune", "adobe", "meitu",
]

def mod_exif(path: str):
    """Retorna (puntos_riesgo, lista_hallazgos)."""
    hallazgos = []
    pts = 0

    try:
        exif = piexif.load(path)
    except Exception:
        hallazgos.append("🟡 Sin metadatos EXIF — posible screenshot o procesado")
        return 1, hallazgos

    ifd0  = exif.get("0th",  {})
    exifd = exif.get("Exif", {})

    # — Software —
    raw_soft = ifd0.get(piexif.ImageIFD.Software, b"")
    software = (raw_soft.decode("utf-8", errors="ignore").strip()
                if isinstance(raw_soft, bytes) else str(raw_soft))
    if software:
        if any(e in software.lower() for e in EDITORES):
            hallazgos.append(f"🔴 Software de edición: **{software}**")
            pts += 4
        else:
            hallazgos.append(f"ℹ️ Software: {software}")
    else:
        hallazgos.append("ℹ️ Sin campo Software en EXIF")

    # — Fechas —
    def _d(raw) -> str:
        return raw.decode("utf-8", errors="ignore").strip() if isinstance(raw, bytes) else ""

    dt_orig = _d(exifd.get(piexif.ExifIFD.DateTimeOriginal, b""))
    dt_mod  = _d(ifd0.get(piexif.ImageIFD.DateTime,         b""))
    if dt_orig and dt_mod and dt_orig != dt_mod:
        hallazgos.append(
            f"🟡 Fecha captura ≠ modificación\n"
            f"    📅 Captura: {dt_orig}\n"
            f"    ✏️ Modificado: {dt_mod}"
        )
        pts += 2
    elif dt_orig:
        hallazgos.append(f"✅ Fecha: {dt_orig}")

    # — Dispositivo —
    make  = _d(ifd0.get(piexif.ImageIFD.Make,  b""))
    model = _d(ifd0.get(piexif.ImageIFD.Model, b""))
    if make or model:
        hallazgos.append(f"ℹ️ Dispositivo: {make} {model}".strip())
    else:
        hallazgos.append("🟡 Sin datos de cámara/dispositivo")
        pts += 1

    if not hallazgos:
        hallazgos.append("✅ Metadatos sin anomalías detectadas")

    return pts, hallazgos


# ══════════════════════════════════════════════════════════════
# MOTOR DE VEREDICTO — Scoring ponderado 0-10
# ══════════════════════════════════════════════════════════════
def build_verdict(ela_score, ghost_ok, ghost_q,
                  clone_n, noise_cv, exif_pts, exif_h):
    risk    = 0.0
    details = []

    # 1. ELA (máx 3 pts)
    if ela_score > 12.0:
        risk += 3.0
        details.append(f"🔴 **ELA crítico** ({ela_score:.2f}): zonas con compresión heterogénea.")
    elif ela_score > 6.0:
        risk += 1.5
        details.append(f"🟡 **ELA moderado** ({ela_score:.2f}): leve inconsistencia de compresión.")
    else:
        details.append(f"🟢 ELA normal ({ela_score:.2f}): compresión uniforme.")

    # 2. Ghost JPEG (máx 2.5 pts)
    if ghost_ok:
        risk += 2.5
        details.append(f"🔴 **Doble compresión** detectada (~{ghost_q}%): imagen re-guardada tras edición.")
    else:
        details.append("🟢 Sin evidencia de doble compresión JPEG.")

    # 3. Copy-Move (máx 3 pts)
    if clone_n >= 20:
        risk += 3.0
        details.append(f"🔴 **Clonado masivo**: {clone_n} bloques copiados internamente.")
    elif clone_n >= 5:
        risk += 1.8
        details.append(f"🟡 **Posible clonado**: {clone_n} bloques coincidentes.")
    elif clone_n > 0:
        risk += 0.8
        details.append(f"🟡 Leve coincidencia de bloques: {clone_n} pares.")
    else:
        details.append("🟢 Sin regiones clonadas detectadas.")

    # 4. Ruido (máx 2 pts)
    if noise_cv > 0.80:
        risk += 2.0
        details.append(f"🔴 **Ruido inconsistente** (CV={noise_cv:.2f}): probable composición.")
    elif noise_cv > 0.40:
        risk += 1.0
        details.append(f"🟡 Ligera inconsistencia de ruido (CV={noise_cv:.2f}).")
    else:
        details.append(f"🟢 Ruido uniforme (CV={noise_cv:.2f}).")

    # 5. EXIF (máx 4 pts → escalado a máx 1.5 en total)
    risk += min(exif_pts * 0.375, 1.5)
    details.append("\n📋 **Metadatos EXIF:**")
    details += [f"  {h}" for h in exif_h]

    # Normalizar 0-10  (máximo teórico: 3+2.5+3+2+1.5 = 12)
    score_10 = round(min(risk / 12.0 * 10.0, 10.0), 1)

    if score_10 >= 6.0:
        veredicto = "🔴 RIESGO ALTO — Probable alteración"
        resumen   = "Múltiples indicadores apuntan a manipulación digital."
    elif score_10 >= 3.0:
        veredicto = "🟡 RIESGO MODERADO — Requiere revisión humana"
        resumen   = "Se detectaron anomalías que pueden indicar edición."
    else:
        veredicto = "🟢 CONFIABLE — Sin anomalías significativas"
        resumen   = "La imagen no muestra señales claras de manipulación."

    return veredicto, resumen, "\n".join(details), score_10


# ══════════════════════════════════════════════════════════════
# HANDLERS DE TELEGRAM
# ══════════════════════════════════════════════════════════════

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid   = str(uuid.uuid4())[:8]
    in_p  = f"in_{uid}.jpg"
    ela_p = f"ela_{uid}.jpg"
    cln_p = f"cln_{uid}.jpg"
    paths = [in_p, ela_p, cln_p]

    try:
        file = await update.message.photo[-1].get_file()
        await file.download_to_drive(in_p)

        await update.message.reply_text(
            "🔬 *Análisis forense iniciado…*\n"
            "Ejecutando 5 módulos. Un momento ⏳",
            parse_mode=constants.ParseMode.MARKDOWN,
        )

        # ── Ejecutar módulos ──────────────────────────────────
        ela_score,  ela_img   = mod_ela(in_p)
        ghost_ok,   ghost_q   = mod_jpeg_ghost(in_p)
        clone_n,    clone_img = mod_clone(in_p)
        noise_cv,   _         = mod_noise(in_p)
        exif_pts,   exif_h    = mod_exif(in_p)

        veredicto, resumen, detalles, score_10 = build_verdict(
            ela_score, ghost_ok, ghost_q,
            clone_n, noise_cv, exif_pts, exif_h,
        )

        # ── Enviar mapa ELA ───────────────────────────────────
        ela_img.save(ela_p, "JPEG", quality=85)
        await update.message.reply_photo(
            photo=open(ela_p, "rb"),
            caption=(
                "🖼️ *Mapa ELA — Errores de Compresión*\n"
                "Zonas brillantes = mayor anomalía de recompresión."
            ),
            parse_mode=constants.ParseMode.MARKDOWN,
        )

        # ── Enviar mapa de clonado (solo si hay pares) ────────
        if clone_img is not None:
            clone_img.save(cln_p, "JPEG", quality=85)
            await update.message.reply_photo(
                photo=open(cln_p, "rb"),
                caption=(
                    "🔁 *Mapa Copy-Move — Bloques Clonados*\n"
                    "🔴 Fuente  🔵 Destino  🟢 Conexión entre pares."
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
        "👋 *Bot Forense de Imágenes*\n\n"
        "Envíame una foto y la analizaré con 5 módulos:\n\n"
        "1️⃣ *ELA* — Errores de compresión JPEG\n"
        "2️⃣ *JPEG Ghost* — Doble compresión (edición previa)\n"
        "3️⃣ *Copy-Move* — Regiones clonadas internamente\n"
        "4️⃣ *Ruido del sensor* — Inconsistencias entre zonas\n"
        "5️⃣ *Metadatos EXIF* — Software de edición y fechas\n\n"
        "📌 _Envía la imagen como *foto*, no como archivo adjunto._",
        parse_mode=constants.ParseMode.MARKDOWN,
    )


# ──────────────────────────────────────────────────────────────
# INICIO
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    log.info("🌐 Flask health-check iniciado.")

    bot = ApplicationBuilder().token(TOKEN).build()
    bot.add_handler(MessageHandler(filters.PHOTO,                    handle_image))
    bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,  handle_text))

    log.info("🤖 Bot forense escuchando...")
    bot.run_polling()
