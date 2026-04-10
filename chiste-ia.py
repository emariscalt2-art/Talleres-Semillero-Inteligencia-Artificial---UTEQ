Create chiste-ia.py
pip install "torch>=2.3.0" "transformers==4.44.2" "huggingface_hub==0.25.2" "accelerate==0.33.0" "safetensors==0.4.4"
from huggingface_hub import login

token = "TOKEN_DE_TU_COMPAÑERO"
login(token)
modelo = "flax-community/gpt-2-spanish"
from transformers import pipeline

modelo_chistes = pipeline(
    "text-generation", 
    model=modelo, 
    device="cpu"
)
prompt = """Pregunta: ¿Qué le dice un cable a otro? 
Respuesta: ¡Sígueme para más corriente!

Pregunta: ¿Qué le dice una IA a un programador?
Respuesta:"""
resultado = modelo_chistes(
    prompt,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)[0]["generated_text"]

print("🎭 Prompt de Estudiante:")
print(prompt)

print("\n🤣 Chiste generado:")
print(resultado)
