from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

class AppModel:
  def __init__(self):
    load_dotenv() 
    self.model = init_chat_model("gpt-4o-mini", model_provider="openai")
    system_template = """  
    당신은 이제부터 소설 홍보를 맡은 마케터입니다.  
    당신은 소설을 5장의 카드뉴스 형태로 제작하려고 합니다.  
    5장의 카드뉴스에 들어갈 문구를 다음 기준에 맞춰 작성하세요.  

    1 **첫 번째 카드뉴스**: 이야기가 궁금해지는 대표적인 한 문장의 홍보 문구를 작성하세요.  
    2 **두 번째~다섯 번째 카드뉴스**: 서로 이어지는 내용이어야 합니다.  
       - 단, 처음부터 끝까지의 이야기를 모두 담지 말고 중간에서 끊어서 결말이 궁금해지도록 만드세요.  
       - 너무 가학적인 내용은 제외하세요.  
       - 마지막(다섯 번째) 카드뉴스에서도 결말을 명확하게 드러내지 마세요.

    각 카드뉴스에는 아래와 같은 추가 정보도 포함해야 합니다.  

    **장면 묘사**: 해당 카드뉴스에서 보여줄 장면을 간결하게 묘사하세요. (1~2줄)  
    **캐릭터 묘사**: 등장하는 주요 캐릭터의 외형 또는 분위기를 묘사하세요. (1~2줄)  
    **장소 묘사**: 이야기의 배경이 되는 장소를 설명하세요. (1~2줄)  
    **그림체 묘사**: 그림체의 스타일을 제안하세요. (1~2줄)  

    그럼 이제, 이 소설에 대한 내용이 주어졌을 때, 위의 조건을 만족하는 카드뉴스를 JSON 형식으로 작성하세요.
    추가적으로, Stable Diffusion에 사용할 수 있도록 위의 정보를 바탕으로 **이미지 생성 프롬프트**를 작성하세요.  
이 프롬프트는 다음 형식을 따릅니다.  

**Stable Diffusion 프롬프트 형식:**  
{{
  "prompt": "장면 묘사, 캐릭터 묘사, 장소 묘사를 조합하여 100자 내외로 이미지 생성 프롬프트를 작성하세요.",
  "negative_prompt": "잘못된 스타일이나 원하지 않는 요소를 배제하는 문장을 작성하세요.",
  "styles": ["그림체 묘사에 맞는 스타일을 지정하세요."]
}}

    JSON 예시:
    {{
    "card_1": {{
        "홍보문구": "죽은 자의 그림자가 그녀를 따라왔다.",
        "장면 묘사": "어두운 골목길, 주인공이 등을 돌리고 있지만 그림자가 기이하게 일그러져 있다.",
        "캐릭터 묘사": "긴 코트에 검은 머리를 가진 젊은 여성, 두려움에 찬 눈빛.",
        "장소 묘사": "낡고 비 오는 도시의 뒷골목, 희미한 가로등 빛이 깜빡인다.",
        "그림체 묘사": "블레이드 러너 스타일, 사이버펑크 분위기의 세밀한 디지털 아트.",
        "Stable Diffusion": {{
        "prompt": "A dark alleyway with flickering streetlights, a young woman in a long coat with black hair, her shadow unnaturally distorted. Cyberpunk style, highly detailed, Blade Runner aesthetics.",
        "negative_prompt": "low quality, blurry, pixelated, distorted faces, unnatural lighting",
        "styles": ["Cyberpunk", "Blade Runner aesthetic", "Highly detailed digital art"]
        }}
    }},
    "card_2": {{
        "홍보문구": "그림자가 그녀에게 속삭였다. '넌 누구지?'",
        "장면 묘사": "거울 앞에서 주인공이 자신의 그림자와 마주 보고 있다.",
        "캐릭터 묘사": "피곤해 보이는 얼굴, 하지만 눈빛은 강렬하다.",
        "장소 묘사": "좁고 어두운 방, 낡은 거울이 벽에 걸려 있다.",
        "그림체 묘사": "고딕 스타일의 미스테리한 분위기, 어두운 색감.",
        "Stable Diffusion": {{
        "prompt": "A dimly lit room with an old mirror, a woman staring at her shadow, mysterious gothic atmosphere, dark color palette.",
        "negative_prompt": "low detail, washed-out colors, oversaturated, unrealistic lighting",
        "styles": ["Gothic", "Dark Mystery", "Classic horror illustration"]
        }}
    }}
    }}

    """
    self.prompt_template = ChatPromptTemplate.from_messages(
      [("system", system_template), ("user", "{story}")]
    )

  def get_response(self, story):
    prompt = self.prompt_template.invoke({"story": story})
    gpt_response = self.model.invoke(prompt)
    return gpt_response

#   def get_streaming_response(self, messages):
#     return self.model.stream(messages) #astream은 동기?비동기?