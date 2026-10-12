from langchain_core.prompts import (ChatPromptTemplate, MessagePromptTemplate,
                                    MessagesPlaceholder, PromptTemplate)
from langchain_groq import ChatGroq

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user",
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a social media assistant tasked with writing engaging, viral tweets. ",
            "Generate the best possible tweet for the user's request.",
            "If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatGroq(model="llama-3.3-70b-versatile")
generation_chain = llm | generation_prompt
reflection_chain = llm | reflection_prompt