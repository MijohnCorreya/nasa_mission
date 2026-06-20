from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str,
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = """You are an expert assistant specializing in NASA space missions. 
You have deep knowledge of Apollo missions, the Challenger disaster, and other NASA programs.

When answering questions:
- Use the provided context from NASA documents as your primary source of information
- Be accurate, factual, and cite specific details from the context when relevant
- If the context does not contain enough information to answer fully, say so clearly
- Keep responses clear, informative, and appropriately detailed
- Maintain a professional yet approachable tone"""

    # Set context in messages — system message includes the retrieved document context
    messages = [
        {
            "role": "system",
            "content": f"{system_prompt}\n\n{context}" if context else system_prompt
        }
    ]

    # Add chat history (exclude the current user message, already appended below)
    for entry in conversation_history:
        if entry.get("role") in ("user", "assistant"):
            messages.append({
                "role": entry["role"],
                "content": entry["content"]
            })

    # Append the current user message
    messages.append({"role": "user", "content": user_message})

    # Create OpenAI client
    client = OpenAI(
    api_key=openai_key,
    base_url="https://openai.vocareum.com/v1"
)


    # Send request to OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    # Return response
    return response.choices[0].message.content