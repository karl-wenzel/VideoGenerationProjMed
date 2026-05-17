import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ACADEMIC_CLOUD_CHATAI_API_KEY"),
    base_url="https://saia.gwdg.de/v1"
)

story_schema = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "description": {"type": "string"},
                    "clothes": {"type": "string"},
                    "hair_color": {"type": "string"},
                    "accessories": {"type": "string"}
                },
                "required": [
                    "name",
                    "age",
                    "description",
                    "clothes",
                    "hair_color",
                    "accessories"
                ],
                "additionalProperties": False
            }
        },
        "scenes": {
            "type": "array",
            "minItems": 4,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "first_sentence": {"type": "string"},
                    "last_sentence": {"type": "string"}
                },
                "required": [
                    "summary",
                    "first_sentence",
                    "last_sentence"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["characters", "scenes"],
    "additionalProperties": False
}

user_query = "A bedtime story about a shy fox named mika with bown hair and a brave firefly who help the moon find its glow again."

system_prompt = """
You are a children's story generator for slideshow creation.

Your task is to take the user's story request and produce a structured story plan for a children's slideshow.

Requirements:
- Return valid JSON only.
- The JSON must follow the provided schema exactly.
- Create 2 to 3 main characters only.
- Create 4 to 8 scenes.
- The story should be age-appropriate for children.
- Keep the tone warm, imaginative, simple, and visual.
- Each scene must include:
  - a short summary,
  - a strong first sentence,
  - a satisfying last sentence.
- Character descriptions should be visually descriptive so they can be illustrated consistently.
- Avoid violence, horror, or mature themes.
- Use simple names unless the user asks otherwise.
- If the user gives specific characters, setting, or style, use them.
- If the user is vague, invent suitable details.

Important:
- Do not include markdown.
- Do not include explanations.
- Do not include any text outside the JSON object.
""".strip()

response = client.chat.completions.create(
    model="openai-gpt-oss-120b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "children_story_slideshow",
            "strict": True,
            "schema": story_schema
        }
    },
    temperature=0.8
)

content = response.choices[0].message.content
story_data = json.loads(content)

base_dir = Path(__file__).resolve().parent
output_path = base_dir.parent / "tmp" / "story.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as f:
    json.dump(story_data, f, indent=2, ensure_ascii=False)

print(f"Saved to: {output_path}")
