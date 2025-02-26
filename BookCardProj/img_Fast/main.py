
import requests
import base64
from PIL import Image
from io import BytesIO

# 공유된 링크를 기반으로 API 엔드포인트 설정
api_url = 'https://41031770ff98aa5fbb.gradio.live/sdapi/v1/txt2img'

# 요청 페이로드 설정
payload = {
  "prompt": "A fantasy landscape with mountains and rivers",
  "steps": 50,
  "cfg_scale": 7.5,
  "width": 512,
  "height": 512
}

# API 요청 보내기
response = requests.post(api_url, json=payload)

# 응답에서 이미지 추출 및 디코딩
if response.status_code == 200:
  response_data = response.json()
  for i, img_data in enumerate(response_data['images']):
      image = Image.open(BytesIO(base64.b64decode(img_data.split(",",1)[0])))
      image.save(f"generated_image_{i}.png")
else:
  print(f"Error: {response.status_code}, {response.text}")