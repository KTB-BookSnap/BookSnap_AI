from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

import app_model

import json

app = FastAPI()
model = app_model.AppModel()

# User로부터 받는 줄거리 text
class TextRequest(BaseModel):
    text: str

# Image를 출력하기 위해 요청하는 프롬프트
class ImageRequest(BaseModel):
    prompt: str
    steps: int = 50
    cfg_scale: float = 7.5
    width: int = 512
    height: int = 512

# Stable Diffusion API 엔드포인트
SD_API_URL = "https://3537212002fdb056f8.gradio.live/sdapi/v1/txt2img"


#https://...../input_story
@app.post("/input_story")
def input_story(story: TextRequest):
    card_news = model.get_response(story.text)


    # 코드 블록 내의 JSON 부분 추출
    start_index = card_news.content.find("```json")
    end_index = card_news.content.find("```", start_index + 1)

    if start_index != -1 and end_index != -1:
        json_str = card_news.content[start_index + len("```json"):end_index].strip()
        try:
            card_news_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON 디코딩 에러: {e}")
            card_news_json =False
    else:
        print("JSON 코드 블록을 찾을 수 없습니다.")

    
    result = {}

    ###image making test
    for i in range(1, len(card_news_json)): 
        result[f"card_{i}"] = {'text': card_news_json[f"card_{i}"]["상황설명"]}
        result[f"card_{i}"]["Stable Diffusion"] = card_news_json[f"card_{i}"]["Stable Diffusion"]

        sd_script = result[f"card_{i}"]["Stable Diffusion"]

        default_prompt = ", highres, high quality, perfect body, perfect face"
        default_negative_prompt = ", deformed hands, extra limbs, distorted face, unrealistic anatomy, fused body parts, uncanny valley, ugly, tiling, watermark, signature"

        payload = {
            "prompt": sd_script['prompt']+default_prompt,
            "negative_prompt": sd_script['negative_prompt']+default_negative_prompt,
            "steps": 50,
            "cfg_scale": 7.5,
            "width": 512,
            "height": 512,
            "seed" : 42
        }



    return {"result": result}