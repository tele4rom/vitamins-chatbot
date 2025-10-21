from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
import os
from typing import List, Optional

app = FastAPI()

# CORS для работы с вашим сайтом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите домен вашего сайта
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация клиентов
import pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1-aws")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Подключение к индексу
index = pinecone.Index("vitamins-catalog-v2")

# Модели данных
class ChatMessage(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    products: Optional[List[dict]] = []

# System prompt для AI
SYSTEM_PROMPT = """Ти - розумний помічник інтернет-магазину вітамінів та БАДів в Україні.

ВАЖЛИВІ ПРАВИЛА:
1. Ти НЕ є медичним працівником і НЕ надаєш медичні консультації
2. Завжди рекомендуй проконсультуватися з лікарем перед прийомом будь-яких добавок
3. Відповідай ввічливо, дружньо та професійно
4. Якщо не знаєш відповіді - чесно скажи про це
5. Використовуй інформацію з бази даних товарів для рекомендацій
6. Можеш спілкуватися українською та російською мовами

ТВОЇ ФУНКЦІЇ:
- Відповідати на питання про товари (склад, ціна, наявність)
- Рекомендувати товари на основі потреб користувача
- Допомагати з навігацією по сайту
- Відповідати на загальні питання про добавки

ДИСКЛЕЙМЕР (додавай при необхідності):
"⚠️ Ця інформація не є медичною рекомендацією. Перед прийомом будь-яких добавок обов'язково проконсультуйтеся з лікарем."

Якщо користувач запитує про конкретні товари або просить рекомендації - використовуй контекст з бази даних."""

def get_embedding(text: str) -> List[float]:
    """Створює embedding для тексту"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_products(query: str, top_k: int = 5) -> List[dict]:
    """Шукає релевантні товари в Pinecone"""
    query_embedding = get_embedding(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    products = []
    for match in results.matches:
        if match.score > 0.7:  # Фільтр по релевантності
            product = match.metadata
            product['score'] = match.score
            products.append(product)
    
    return products

def format_products_context(products: List[dict]) -> str:
    """Форматує товари для контексту AI"""
    if not products:
        return "Товари не знайдені."
    
    context = "ЗНАЙДЕНІ ТОВАРИ:\n\n"
    for i, product in enumerate(products, 1):
        context += f"{i}. {product.get('name', 'Без назви')}\n"
        context += f"   Бренд: {product.get('brand', 'Не вказано')}\n"
        context += f"   Ціна: {product.get('price', 'Не вказано')} грн\n"
        context += f"   Наявність: {'В наявності' if product.get('status') else 'Немає в наявності'}\n"
        context += f"   Опис: {product.get('description', 'Немає опису')[:200]}...\n"
        context += f"   SKU: {product.get('sku', 'N/A')}\n\n"
    
    return context

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):
    """Обробка повідомлень від користувача"""
    try:
        user_message = request.message
        conversation_history = request.conversation_history or []
        
        # Пошук релевантних товарів
        products = search_products(user_message, top_k=5)
        products_context = format_products_context(products)
        
        # Формування повідомлень для OpenAI
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": products_context}
        ]
        
        # Додаємо історію розмови (останні 5 повідомлень)
        messages.extend(conversation_history[-10:])
        messages.append({"role": "user", "content": user_message})
        
        # Запит до OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        ai_response = response.choices[0].message.content
        
        return ChatResponse(
            response=ai_response,
            products=[{
                "name": p.get("name"),
                "price": p.get("price"),
                "brand": p.get("brand"),
                "sku": p.get("sku"),
                "product_id": p.get("product_id"),
                "status": p.get("status")
            } for p in products[:3]]  # Повертаємо топ-3 товари
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Перевірка роботи API"""
    return {"status": "ok", "message": "Chatbot API is running"}

@app.get("/")
async def root():
    """Головна сторінка"""
    return {
        "message": "Vitamins Chatbot API",
        "version": "1.0",
        "endpoints": {
            "/chat": "POST - Send message to chatbot",
            "/health": "GET - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)