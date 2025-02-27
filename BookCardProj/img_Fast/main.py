from fastapi import FastAPI
import requests
import base64
from pydantic import BaseModel
from typing import List

app = FastAPI()

# 요청 모델 정의 (유저가 프롬프트를 입력하면 해당 구조를 받음)
class ImageRequest(BaseModel):
    prompt: str
    steps: int = 50
    cfg_scale: float = 7.5
    width: int = 512
    height: int = 512

# Stable Diffusion API 엔드포인트
SD_API_URL = "https://f764cec27bc93d87dc.gradio.live/sdapi/v1/txt2img"


@app.post("/generate-image")
def generate_image():
    payload = {
        "prompt": "A mysterious young prince with golden hair, standing in a vast desert under a starry sky, wearing a long green coat and a scarf blowing in the wind. Cinematic lighting, dreamlike atmosphere, soft pastel tones.",
        "negative_prompt": "blurry, low detail, unrealistic proportions, exaggerated facial features",
        "steps": 50,
        "cfg_scale": 7.5,
        "width": 512,
        "height": 512
    }

    # Stable Diffusion API 요청
    response = requests.post(SD_API_URL, json=payload)


    if response.status_code == 200:
        response_data = response.json()  # Stable Diffusion API 응답(JSON)
        return {"images": response_data["images"]}  # JSON 형태 그대로 반환
    else:
        return {"error": f"Request failed with status {response.status_code}"}