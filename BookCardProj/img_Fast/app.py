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
SD_API_URL = "https://41031770ff98aa5fbb.gradio.live/sdapi/v1/txt2img"


@app.post("/generate-image/")
async def generate_image(request: ImageRequest):
    # API 요청을 위한 페이로드 구성
    payload = {
        "prompt": request.prompt,
        "steps": request.steps,
        "cfg_scale": request.cfg_scale,
        "width": request.width,
        "height": request.height
    }

    # Stable Diffusion API 요청
    response = requests.post(SD_API_URL, json=payload)

    if response.status_code == 200:
        response_data = response.json()  # Stable Diffusion API 응답(JSON)
        return {"images": response_data["images"]}  # JSON 형태 그대로 반환
    else:
        return {"error": f"Request failed with status {response.status_code}"}
