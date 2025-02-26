
import requests
import base64
from PIL import Image
from io import BytesIO

# 공유된 링크를 기반으로 API 엔드포인트 설정
api_url = 'https://f764cec27bc93d87dc.gradio.live/sdapi/v1/txt2img'

# 요청 페이로드 설정
payload = {
  "prompt": "A mysterious young prince with golden hair, standing in a vast desert under a starry sky, wearing a long green coat and a scarf blowing in the wind. Cinematic lighting, dreamlike atmosphere, soft pastel tones.",
  "negative_prompt": "blurry, low detail, unrealistic proportions, exaggerated facial features",
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