from langchain_core.prompts import ChatPromptTemplate

def adjust_system_prompt(system_prompt: str):
    template = (
        f"{system_prompt}\n\n"
        "Використовуючи наданий контекст:\n"
        "{context}\n\n"
        "Відповідай на питання користувача:\n"
        "{question}"
    )
    return ChatPromptTemplate.from_template(template)