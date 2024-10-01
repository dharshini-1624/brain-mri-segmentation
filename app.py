import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = load_model('path_to_your_best_model.h5', custom_objects={'dice_score': lambda y_true, y_pred: None})

def preprocess_image(image: Image.Image):
    image = image.convert('L')  
    image = image.resize((256, 256))  
    image_array = np.array(image) / 255.0  
    return np.expand_dims(image_array, axis=(0, -1))  
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        
        processed_image = preprocess_image(image)

        
        prediction = model.predict(processed_image)
        prediction = (prediction > 0.5).astype(np.uint8) 

        
        result_image = Image.fromarray(prediction[0, :, :, 0] * 255)
        result_image_io = io.BytesIO()
        result_image.save(result_image_io, format='PNG')
        result_image_io.seek(0)

        return JSONResponse(content={"filename": file.filename, "segmentation": result_image_io.getvalue()})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
