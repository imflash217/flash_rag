def get_prompt(user_prompt: str, retrieved_responses: list):
    # TODO!
    prompt = f"""
{retrieved_responses}
---------
{user_prompt}
Answer:
"""
    return prompt.strip()
