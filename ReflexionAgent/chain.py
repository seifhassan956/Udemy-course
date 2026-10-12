import datetime

from dotenv import load_dotenv

from ReflexionAgent.schemas import AnswerQuestion, RevisedAnswer

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (JsonOutputToolsParser,
                                                        PydanticToolsParser)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.

    Current time: {time}

    1. {first_instruction}
    2. Reflect and critique your answer. Be severe to maximize improvement.
    3. Recommend search queries to research information and improve your answer."""
            ),
        ]
).partial(time=lambda: datetime.datetime.now().isoformat())

llm = ChatGroq(model="llama-3.3-70b-versatile")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

first_responder_prompt = actor_prompt_template.partial(first_instruction="Answer the question to the best of your ability.")
first_reflector_prompt = actor_prompt_template | llm.bind_tools(tools=[AnswerQuestion],tool_choice = "AnswerQuestion") 

revise_instructions = """Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer.
    - You MUST include numerical citations in your revised answer to ensure it can be verified.
    - Add a "References" section to the bottom of your answer (which does not count towards the word limit):
        - [1] https://example.com
        - [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make SURE ..."""

human_message = HumanMessage(
    content="Write about AI-Powered SOC / autonomous soc  problem domain,"
    " list startups that do that and raised capital."
)

first_responder = (
    first_responder_prompt
    | llm.bind_tools(
        tools=[AnswerQuestion],
        tool_choice="AnswerQuestion"
    )
    | PydanticToolsParser(tools=[AnswerQuestion])
)

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[RevisedAnswer], tool_choice="RevisedAnswer") | parser_pydantic