from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from audio_feature import extract_voice_features_from_mp3
from prediction_model import predict_gender_from_voice_features , predict_age

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    
    features = await extract_voice_features_from_mp3(file)


    result = predict_gender_from_voice_features(features)

    result_age = predict_age(features)

    return JSONResponse(content={
        "gender": result["gender"],
        "age": result_age["predicted_age_group"],
        "certainty": result['confidence']
    })
