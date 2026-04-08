import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
from PIL import Image, ImageChops, ImageEnhance

# Configuración básica
TOKEN = "8699029540:AAE9TGMSC5fvW2Fldhuc_keYQAYxM_ooW_s"

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def run_ela(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    temp_resave = "temp_resave.jpg"
    original.save(temp_resave, 'JPEG', quality=quality)
    temporary = Image.open(temp_resave)
    
    ela_image = ImageChops.difference(original, temporary)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    os.remove(temp_resave)
    return ela_image

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.photo[-1].get_file()
    input_path = "input_img.jpg"
    output_path = "ela_result.jpg"
    
    await file.download_to_drive(input_path)
    await update.message.reply_text("🔍 Analizando niveles de error (ELA)... Por favor espera.")
    
    # Ejecutar análisis
    ela_res = run_ela(input_path)
    ela_res.save(output_path)
    
    # Enviar resultado
    await update.message.reply_photo(photo=open(output_path, 'rb'), caption="✅ Análisis ELA completado.\n\nSi ves zonas que brillan mucho más que el resto (especialmente en textos o montos), es señal de manipulación digital.")
    
    os.remove(input_path)
    os.remove(output_path)

if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()
