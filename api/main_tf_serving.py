from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# السماح بالاتصال من الواجهة الأمامية
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل النموذج المحلي
MODEL = tf.keras.models.load_model(
    r"D:\python for data sinceand ml\Automatic Diagnosis of sunflower_leaf Diseases1\models\model_backup.h5"
)

# أسماء التصنيفات
CLASS_NAME = ['Downy mildew', 'Fresh Leaf', 'Gray mold', 'Leaf scars']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# قراءة الصورة وتطبيعها
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))  # غير الحجم حسب ما دربت النموذج
    return np.array(image) / 255.0     # تطبيع

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(img_batch)[0]
    predicted_class = CLASS_NAME[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
