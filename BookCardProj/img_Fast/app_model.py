from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

    # - 단, 처음부터 끝까지의 이야기를 모두 담지 말고 중간에서 끊어서 결말이 궁금해지도록 만드세요.
    # - 기승전결 중, 기승 까지만 묘사해야합니다.  
    # - 마지막(다섯 번째) 카드뉴스에서도 결말을 명확하게 드러내지 마세요.
    


#     - 6장의 카드뉴스는 서로 이어지는 내용이어야 합니다.  

#     각 카드뉴스에는 아래와 같은 추가 정보도 포함해야 합니다.  

#     **장면 묘사**: 해당 카드뉴스에서 보여줄 장면을 간결하게 묘사하세요. (1~2줄)  
#     **캐릭터 묘사**: 등장하는 주요 캐릭터의 외형 또는 분위기를 묘사하세요. (1~2줄)  
#     **장소 묘사**: 이야기의 배경이 되는 장소를 설명하세요. (1~2줄)  
#     **그림체 묘사**: 그림체의 스타일을 제안하세요. (1~2줄)  

#     그럼 이제, 이 소설에 대한 내용이 주어졌을 때, 위의 조건을 만족하는 카드뉴스를 JSON 형식으로 작성하세요.
#     추가적으로, Stable Diffusion에 사용할 수 있도록 위의 정보를 바탕으로 **이미지 생성 프롬프트**를 작성하세요.  
# 이 프롬프트는 다음 형식을 따릅니다.  


class AppModel:
  def __init__(self):
    load_dotenv() 
    self.model = init_chat_model("gpt-4o", model_provider="openai")
    system_template = """  
    당신은 소설의 줄거리를 요약해서 6장의 이미지를 stable diffusion으로 생성하려고 합니다.  
    그리고 소설의 줄거리를 기반으로 각 이미지의 상황을 설명하는 문구도 같이 작성해 주세요.
    
    다음 형식에 맞춰 stable diffusion에 들어갈 프롬프트를 작성해주세요.
    **Stable Diffusion 프롬프트 형식:**  
    {{
      "prompt": "소설의 줄거리와 각 이미지의 상황을 설명하는 한 문장의 이미지 생성 프롬프트를 영어로 작성하세요.
      이미지에는 한 명의 인물만 등장 하고, 그 인물의 한 가지 행동만을 담아주세요.
      인물의 이름을 쓸 때는 이름과 함께 괄호로 정확한 나이와 성별을 표기하세요.
      이미지의 배경에 대한 묘사를 작성하세요.
      프롬프트는 최대한 단어로 작성하고, 단어들을 쉼표로 나열하세요.
      이미지는 다음의 질문에 대답할 수 있어야합니다.
      - 무슨일이 일어나고 있나요?
      - 주체는 무엇을 하고 있나요?
      - 주체는 어떻게 하고 있나요?
      - 주체 주변에서 무슨일이 일어나고 있나요?
      이미지를 묘사하는데 중요한 키위드를 앞에 배치하세요.",
      "negative_prompt": "잘못된 스타일이나 원하지 않는 요소를 배제하는 단어들을 쉼표로 나열하세요.",
    }}

    JSON 예시:
    {{
    "card_1": {{
        "상황설명": "...",
        "Stable Diffusion": {{
        "prompt": "...",
        "negative_prompt": "...",
        }}
    }},
    "card_2": {{
        "상황설명": "...",
        "Stable Diffusion": {{
        "prompt": "...",
        "negative_prompt": "...",
        }}
    }},
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