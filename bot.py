def analyze_image(image_path):
    # --- 1. ELA TRADICIONAL ---
    quality = 90
    original = Image.open(image_path).convert('RGB')
    temp_resave = "temp_resave.jpg"
    original.save(temp_resave, 'JPEG', quality=quality)
    temporary = Image.open(temp_resave)
    ela_image = ImageChops.difference(original, temporary)
    stat = ImageStat.Stat(ela_image)
    ela_score = sum(stat.mean)

    # --- 2. DETECTOR DE PARCHES (NUEVO) ---
    # Convertimos a escala de grises para buscar inconsistencias de brillo
    img_cv = cv2.imread(image_path)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Buscamos variaciones locales de desviación estándar 
    # (Los parches de Power Point suelen ser muy "lisos" comparados con el ruido del papel real)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    diff_textura = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    patch_score = np.var(diff_textura) # Si hay saltos bruscos de textura, este número sube

    # --- 3. BORDES ---
    edges = cv2.Canny(gray, 50, 150)
    edge_score = np.mean(edges)

    os.remove(temp_resave)
    return ela_image, ela_score, edge_score, patch_score

def generate_dictamen(ela_score, edge_score, patch_score):
    risk_points = 0
    detalles = []

    # Lógica de detección de parches (Power Point/Paint)
    if patch_score > 5000: # Umbral para detectar cambios bruscos de textura/fondo
        risk_points += 3
        detalles.append("- 🔴 **PARCHE DETECTADO**: Se detectó un cambio de textura en el fondo (típico de montos sobrepuestos).")

    if edge_score > 12.0:
        risk_points += 2
        detalles.append("- 🔴 **BORDES ARTIFICIALES**: Letras demasiado nítidas para ser una foto real.")

    if ela_score > 10.0:
        risk_points += 1
        detalles.append("- 🟡 Inconsistencia digital leve en los píxeles.")

    if risk_points >= 3:
        return "🔴 **RIESGO CRÍTICO / POSIBLE FRAUDE**", "\n".join(detalles)
    elif risk_points >= 1:
        return "🟡 **RIESGO MODERADO**", "\n".join(detalles)
    else:
        return "🟢 **CONFIABLE**", "- No se detectan anomalías evidentes."
