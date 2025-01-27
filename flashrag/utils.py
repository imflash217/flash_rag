def get_prompt(user_prompt: str):
    prompt = f"""
{user_prompt}
Answer:
"""
    return prompt.strip()


def get_llm():
    return None
