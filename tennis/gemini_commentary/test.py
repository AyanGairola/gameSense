import google.generativeai as genai

genai.configure(api_key="AIzaSyCFGXMS4OMDNCLK8gaYq-K5ky8I9U9ELEM")

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a story about a magic backpack", generation_config=genai.types.GenerationConfig(
    max_output_tokens=52,
    temperature=1.0
))
print(response.text)