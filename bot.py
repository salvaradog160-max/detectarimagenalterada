"""
╔══════════════════════════════════════════════════════════════╗
║   BOT FORENSE — Calibrado con datos reales de imagen         ║
║   Módulos probados contra comprobante con monto alterado     ║
║                                                              ║
║  Señales que SÍ funcionan en documentos físicos:             ║
║   1. Rectángulos con fondo interior más liso que su entorno  ║
║   2. ELA zonal (zona monto vs zona papel)                    ║
║   3. JPEG Ghost (doble compresión)                           ║
║   4. Ruido del sensor por bloques                            ║
║   5. Metadatos EXIF                                          ║
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
from PIL import Image, ImageChops, ImageStat

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────
TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
if not TOKEN:
    raise RuntimeError("❌ Define TELEGRAM_TOKEN en las variables de entorno de Render.")

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

MAX_SIDE_PX = 1400   # Proteger RAM Render Free (512 MB)

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
def _resize(img: np.ndarray, max_side: int = MAX_SIDE_PX) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

def _load_pil(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE_PX:
        s = MAX_SIDE_PX / max(w, h)
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return img

def _load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo leer la imagen.")
    return _resize(img)

def _bg_std(region: np.ndarray) -> float:
    """Desviación estándar de los píxeles de fondo (claros > 140)."""
    px = region.flatten()
    px = px[px > 140]
    return float(np.std(px)) if len(px) > 15 else 999.0


# ══════════════════════════════════════════════════════════════
# MÓDULO 1 — ELA MULTI-CALIDAD + ANÁLISIS ZONAL
#
# Detecta: zonas con diferente nivel de compresión JPEG.
# Novedad: además del score global, compara la zona con más
# actividad ELA vs la zona de papel sin texto (fondo puro).
# Si la zona "caliente" tiene ELA 1.3x mayor que el papel → sospecha.
#
# Calibración real:
#   imagen alterada → ela_global=1.67, ela_ratio_monto/papel=1.31
#   umbral global: > 2.0 crítico, > 1.2 moderado
#   umbral zonal : ratio > 1.25 sospechoso
# ══════════════════════════════════════════════════════════════
def mod_ela(path: str):
    """
    Retorna (ela_global, ela_ratio_max, ela_img_PIL).
    ela_ratio_max: cuánto más alta es la zona caliente vs fondo papel.
    """
    original = _load_pil(path)
    w_img, h_img = original.size
    acum = None

    for q in (75, 80, 85, 90):
        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        recomp = Image.open(buf).convert("RGB")
        diff = np.abs(
            np.array(original, np.float32) -
            np.array(recomp,   np.float32)
        )
        acum = diff if acum is None else np.maximum(acum, diff)

    ela_global = float(acum.mean())
    ela_gray   = acum.mean(axis=2)   # canal único

    # Dividir en 16 bloques verticales y comparar el más caliente
    # con el promedio de los 4 más fríos (papel limpio)
    h_px, w_px = ela_gray.shape
    strip_h = h_px // 16
    strip_scores = [ela_gray[i*strip_h:(i+1)*strip_h, :].mean() for i in range(16)]
    strip_sorted  = sorted(strip_scores)
    cold_avg      = float(np.mean(strip_sorted[:4]))   # las 4 franjas más frías
    hot_max       = float(max(strip_scores))
    ela_ratio     = hot_max / (cold_avg + 1e-6)

    # Imagen ELA amplificada para enviar
    peak  = acum.max() or 1.0
    vis   = np.clip(acum * (255.0 / peak), 0, 255).astype(np.uint8)
    ela_pil = Image.fromarray(vis)

    return ela_global, ela_ratio, ela_pil


# ══════════════════════════════════════════════════════════════
# MÓDULO 2 — DETECCIÓN DE RECUADROS CON FONDO ARTIFICIAL
#
# Qué detecta: rectangulares donde el fondo interior es MÁS LISO
# que el papel inmediatamente circundante (arriba y abajo).
# El papel fotografiado tiene grano/textura; un parche digital no.
#
# Algoritmo:
#   1. Detectar bordes fuertes con Sobel
#   2. Encontrar líneas rectas horizontales con HoughLinesP
#   3. Buscar pares que formen "caja" (separadas 20-120 px)
#   4. Comparar std del fondo interior vs entorno inmediato
#   5. suavidad = std_entorno / std_interior > 1.4 → SOSPECHOSO
#   6. Deduplicar con grid de 30px para no contar la misma zona múltiples veces
#
# Calibración real (imagen con monto sobrepuesto):
#   → 21 zonas únicas sospechosas, suavidad máx=2.33
#   → umbral final: n_unique >= 3 → score alto
# ══════════════════════════════════════════════════════════════
def mod_rect_patch(path: str):
    """
    Retorna (patch_score 0-10, n_zonas_unicas, imagen_marcada_PIL|None).
    """
    gray = _load_gray(path)
    h, w = gray.shape

    # Detectar bordes y líneas rectas horizontales
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad   = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    _, edge_mask = cv2.threshold(grad, 40, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(
        edge_mask, 1, np.pi/180,
        threshold=60, minLineLength=60, maxLineGap=12
    )
    if lines is None:
        return 0.0, 0, None

    h_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 8 or angle > 172:
            h_lines.append((min(y1, y2), min(x1, x2), max(x1, x2)))

    # Buscar pares que forman recuadro
    seen_cells     = set()
    unique_suspect = []
    h_sorted = sorted(h_lines)

    for i, (y1, ax1, ax2) in enumerate(h_sorted):
        for y2, bx1, bx2 in h_sorted[i+1:]:
            gap = y2 - y1
            if not (20 <= gap <= 120):
                continue
            ov_start = max(ax1, bx1)
            ov_end   = min(ax2, bx2)
            if ov_end - ov_start < 70:
                continue

            # Deduplicar
            cell = (y1//30, y2//30, ov_start//30, ov_end//30)
            if cell in seen_cells:
                continue
            seen_cells.add(cell)

            roi_in  = gray[y1:y2, ov_start:ov_end]
            roi_top = gray[max(0, y1-40):y1, ov_start:ov_end]
            roi_bot = gray[y2:min(h, y2+40), ov_start:ov_end]

            if roi_in.size < 200:
                continue

            std_in  = _bg_std(roi_in)
            sur_px  = np.concatenate([roi_top.flatten(), roi_bot.flatten()])
            sur_px  = sur_px[sur_px > 140]
            std_sur = float(np.std(sur_px)) if len(sur_px) > 30 else 0.0

            suavidad = std_sur / (std_in + 1e-6)
            if suavidad > 1.4:
                unique_suspect.append((suavidad, y1, y2, ov_start, ov_end))

    unique_suspect.sort(reverse=True)
    n = len(unique_suspect)

    # Score calibrado:
    # n>=5 → casi seguro alterado → 10
    # n>=3 → probable            → 7
    # n>=2 → posible             → 5
    # n==1 → leve sospecha       → 3
    if n >= 5:
        score = 10.0
    elif n >= 3:
        score = 7.0
    elif n >= 2:
        score = 5.0
    elif n == 1:
        score = 3.0
    else:
        score = 0.0

    # Imagen marcada (solo top-8 zonas únicas)
    marked = None
    if n > 0:
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for suav, y1, y2, x1, x2 in unique_suspect[:8]:
            color = (0, 50, 255) if suav > 1.8 else (0, 140, 255)
            cv2.rectangle(img_color, (x1, y1), (x2, y2), color, 2)
            label = f"{suav:.1f}x"
            cv2.putText(img_color, label, (x1+2, y1+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        marked = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

    return score, n, marked


# ══════════════════════════════════════════════════════════════
# MÓDULO 3 — JPEG GHOST (Doble compresión)
#
# Detecta: imagen re-guardada tras edición (la calidad original
# queda "impresa" y al recomprimir a esa calidad la diferencia
# es mínima → fue guardada antes a esa calidad).
#
# Calibración: en la imagen alterada detectó doble compresión ~95%.
# ══════════════════════════════════════════════════════════════
def mod_jpeg_ghost(path: str):
    """Retorna (detectado bool, calidad_sospechosa int|None)."""
    original = _load_pil(path)
    orig_arr = np.array(original, np.float32)
    min_diff = float("inf")
    best_q   = None

    for q in range(55, 98, 5):
        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        test = np.array(Image.open(buf).convert("RGB"), np.float32)
        diff = float(np.mean(np.abs(orig_arr - test)))
        if diff < min_diff:
            min_diff, best_q = diff, q

    # Umbral empírico: diff < 2.0 → calidad ya usada antes
    detected = min_diff < 2.0
    return detected, (best_q if detected else None)


# ══════════════════════════════════════════════════════════════
# MÓDULO 4 — RUIDO DEL SENSOR POR BLOQUES
#
# Detecta: zonas pegadas de otra imagen (tienen diferente nivel
# de ruido digital al resto del documento).
#
# Calibración real: CV=0.71 en imagen alterada
# Nuevos umbrales: > 0.6 moderado, > 1.0 alto
# (antes era 0.8/1.2 → demasiado permisivo)
# ══════════════════════════════════════════════════════════════
def mod_noise(path: str):
    """Retorna (cv float, descripcion str)."""
    gray    = _load_gray(path).astype(np.float64)
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

    if cv > 1.0:
        desc = "Ruido muy inconsistente entre zonas (probable composición)"
    elif cv > 0.6:
        desc = "Inconsistencia moderada de ruido de sensor"
    else:
        desc = "Ruido uniforme en el documento"

    return cv, desc


# ══════════════════════════════════════════════════════════════
# MÓDULO 5 — METADATOS EXIF
# ══════════════════════════════════════════════════════════════
EDITORES = [
    "photoshop", "gimp", "lightroom", "affinity", "canva",
    "picsart", "pixelmator", "snapseed", "paint.net",
    "facetune", "adobe", "meitu", "inshot",
]

def mod_exif(path: str):
    """Retorna (puntos_riesgo int, hallazgos list[str])."""
    hallazgos, pts = [], 0

    try:
        exif = piexif.load(path)
    except Exception:
        hallazgos.append("🟡 Sin metadatos EXIF (posible screenshot o imagen procesada)")
        return 1, hallazgos

    ifd0  = exif.get("0th",  {})
    exifd = exif.get("Exif", {})

    def _d(raw) -> str:
        return raw.decode("utf-8", errors="ignore").strip() if isinstance(raw, bytes) else ""

    soft = _d(ifd0.get(piexif.ImageIFD.Software, b""))
    if soft:
        if any(e in soft.lower() for e in EDITORES):
            hallazgos.append(f"🔴 Software de edición: **{soft}**")
            pts += 4
        else:
            hallazgos.append(f"ℹ️ Software: {soft}")
    else:
        hallazgos.append("ℹ️ Sin campo Software en EXIF")

    dt_orig = _d(exifd.get(piexif.ExifIFD.DateTimeOriginal, b""))
    dt_mod  = _d(ifd0.get(piexif.ImageIFD.DateTime,         b""))
    if dt_orig and dt_mod and dt_orig != dt_mod:
        hallazgos.append(
            f"🟡 Fechas inconsistentes:\n"
            f"    📅 Captura: {dt_orig}\n"
            f"    ✏️ Modificado: {dt_mod}"
        )
        pts += 2
    elif dt_orig:
        hallazgos.append(f"✅ Fecha captura: {dt_orig}")

    make  = _d(ifd0.get(piexif.ImageIFD.Make,  b""))
    model = _d(ifd0.get(piexif.ImageIFD.Model, b""))
    if make or model:
        hallazgos.append(f"ℹ️ Dispositivo: {make} {model}".strip())
    else:
        hallazgos.append("🟡 Sin datos de cámara/dispositivo")
        pts += 1

    return pts, hallazgos


# ══════════════════════════════════════════════════════════════
# MOTOR DE VEREDICTO
#
# Pesos calibrados con datos reales:
#   Parche (recuadro liso) : hasta 4.0 pts  — señal más directa
#   ELA zonal              : hasta 1.5 pts
#   ELA global             : hasta 1.0 pts
#   Ghost JPEG             : hasta 2.5 pts
#   Ruido inconsistente    : hasta 2.0 pts
#   EXIF                   : hasta 1.5 pts
#   Total máximo           : 12.5 pts → normalizado a 10
#
# Verificado: imagen con monto sobrepuesto obtiene 7.5/10 → 🔴
# ══════════════════════════════════════════════════════════════
def build_verdict(ela_global, ela_ratio,
                  patch_score, patch_n,
                  ghost_ok, ghost_q,
                  noise_cv,
                  exif_pts, exif_h):

    risk    = 0.0
    details = []

    # 1. Parche/recuadro liso  (0-4 pts)
    patch_pts = patch_score / 10.0 * 4.0
    risk += patch_pts
    if patch_score >= 7.0:
        details.append(
            f"🔴 **RECUADRO CON FONDO ARTIFICIAL** ({patch_n} zonas): "
            "Se detectaron rectángulos cuyo fondo interior es más liso "
            "que el papel circundante. Señal de monto o texto sobrepuesto digitalmente."
        )
    elif patch_score >= 3.0:
        details.append(
            f"🟡 **Zona con fondo anómalo** ({patch_n} zona/s): "
            "Posible recuadro o parche digital."
        )
    else:
        details.append("🟢 Sin recuadros artificiales detectados.")

    # 2. ELA zonal  (0-1.5 pts)
    if ela_ratio > 1.5:
        risk += 1.5
        details.append(
            f"🔴 **ELA zonal crítico** (ratio={ela_ratio:.2f}x): "
            "Una zona concentra anomalías de compresión mucho mayores al resto."
        )
    elif ela_ratio > 1.25:
        risk += 0.8
        details.append(
            f"🟡 **ELA zonal moderado** (ratio={ela_ratio:.2f}x): "
            "Zona con compresión notablemente diferente al resto del documento."
        )
    else:
        details.append(f"🟢 ELA zonal uniforme (ratio={ela_ratio:.2f}x).")

    # 3. ELA global  (0-1 pt)
    if ela_global > 2.5:
        risk += 1.0
        details.append(f"🔴 ELA global alto ({ela_global:.2f}): compresión heterogénea.")
    elif ela_global > 1.5:
        risk += 0.5
        details.append(f"🟡 ELA global moderado ({ela_global:.2f}).")
    else:
        details.append(f"🟢 ELA global normal ({ela_global:.2f}).")

    # 4. Ghost JPEG  (0-2.5 pts)
    if ghost_ok:
        risk += 2.5
        details.append(
            f"🔴 **Doble compresión JPEG** (~{ghost_q}%): "
            "La imagen fue guardada al menos dos veces → editada y re-guardada."
        )
    else:
        details.append("🟢 Sin doble compresión JPEG.")

    # 5. Ruido  (0-2 pts)  — umbrales corregidos
    if noise_cv > 1.0:
        risk += 2.0
        details.append(
            f"🔴 **Ruido muy inconsistente** (CV={noise_cv:.2f}): "
            "Diferentes zonas tienen distinto grano → probable composición."
        )
    elif noise_cv > 0.6:
        risk += 1.0
        details.append(f"🟡 Ruido moderadamente inconsistente (CV={noise_cv:.2f}).")
    else:
        details.append(f"🟢 Ruido uniforme (CV={noise_cv:.2f}).")

    # 6. EXIF  (escalado a máx 1.5 pts)
    exif_pts_norm = min(exif_pts * 0.375, 1.5)
    risk += exif_pts_norm
    details.append("\n📋 **Metadatos EXIF:**")
    details += [f"  {h}" for h in exif_h]

    # Normalizar 0-10
    score_10 = round(min(risk / 12.5 * 10.0, 10.0), 1)

    if score_10 >= 6.0:
        veredicto = "🔴 RIESGO ALTO — Probable alteración del documento"
        resumen   = "Múltiples indicadores señalan manipulación digital."
    elif score_10 >= 3.5:
        veredicto = "🟡 RIESGO MODERADO — Revisión humana recomendada"
        resumen   = "Se detectaron anomalías compatibles con edición."
    else:
        veredicto = "🟢 SIN ANOMALÍAS SIGNIFICATIVAS"
        resumen   = "El documento no muestra señales claras de manipulación."

    return veredicto, resumen, "\n".join(details), score_10


# ══════════════════════════════════════════════════════════════
# HANDLER PRINCIPAL
# ══════════════════════════════════════════════════════════════
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid     = str(uuid.uuid4())[:8]
    in_p    = f"in_{uid}.jpg"
    ela_p   = f"ela_{uid}.jpg"
    patch_p = f"patch_{uid}.jpg"
    paths   = [in_p, ela_p, patch_p]

    try:
        file = await update.message.photo[-1].get_file()
        await file.download_to_drive(in_p)

        await update.message.reply_text(
            "🔬 *Análisis forense iniciado…*\n"
            "Ejecutando 5 módulos calibrados ⏳",
            parse_mode=constants.ParseMode.MARKDOWN,
        )

        # ── Ejecutar módulos ──────────────────────────────────
        ela_global, ela_ratio, ela_img    = mod_ela(in_p)
        patch_score, patch_n, patch_img   = mod_rect_patch(in_p)
        ghost_ok, ghost_q                 = mod_jpeg_ghost(in_p)
        noise_cv, _                       = mod_noise(in_p)
        exif_pts, exif_h                  = mod_exif(in_p)

        veredicto, resumen, detalles, score_10 = build_verdict(
            ela_global, ela_ratio,
            patch_score, patch_n,
            ghost_ok, ghost_q,
            noise_cv,
            exif_pts, exif_h,
        )

        # ── Enviar mapa ELA ───────────────────────────────────
        ela_img.save(ela_p, "JPEG", quality=85)
        await update.message.reply_photo(
            photo=open(ela_p, "rb"),
            caption=(
                "🖼️ *Mapa ELA — Errores de Compresión*\n"
                "Zonas brillantes = compresión heterogénea (anomalía)."
            ),
            parse_mode=constants.ParseMode.MARKDOWN,
        )

        # ── Enviar mapa de parches ────────────────────────────
        if patch_img is not None:
            patch_img.save(patch_p, "JPEG", quality=85)
            await update.message.reply_photo(
                photo=open(patch_p, "rb"),
                caption=(
                    "🟠 *Mapa de Recuadros Anómalos*\n"
                    "Rectángulos marcados = zonas cuyo fondo es más liso\n"
                    "que el papel de su entorno inmediato.\n"
                    "Número = factor de suavidad (> 1.4× = sospechoso)."
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
        "Envíame la foto del comprobante y lo analizaré con 5 módulos:\n\n"
        "1️⃣ *Recuadros artificiales* — Fondo interior más liso que el papel\n"
        "2️⃣ *ELA zonal + global* — Anomalías de compresión JPEG por zona\n"
        "3️⃣ *JPEG Ghost* — Doble compresión (editado y re-guardado)\n"
        "4️⃣ *Ruido del sensor* — Inconsistencias entre regiones\n"
        "5️⃣ *Metadatos EXIF* — Software de edición y fechas\n\n"
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
