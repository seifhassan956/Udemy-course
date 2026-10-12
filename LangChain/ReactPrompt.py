import inspect
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

from langchain.tools import tool

load_dotenv()
MAX_ITERATIONS = 10


@tool
def getProductPrice(product: str) -> str:
    """
        Get the catalog price of a product.

        Use this tool when you need to find the price of a product.
        Do not guess or calculate the product price yourself.
    """
    print(f"Getting price for: {product}")
    prices = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
    }
    key = product.strip().lower()
    if key in prices:
        return f"The price of {key} is ${prices[key]:.2f}."
    return f"Product {product} not found."


@tool
def applyDiscount(price: float, discount_tier: str) -> str:
    """
        Calculate the discounted price for a product.

        Use this tool after getting the product price.
        Valid discount tiers are:
        - regular: 10% discount
        - premium: 20% discount
        - vip: 30% discount

        Use the exact price returned by getProductPrice.
    """
    discounts = {
        "regular": 0.1,
        "premium": 0.2,
        "vip": 0.3,
    }
    tier = discount_tier.strip().lower()
    if tier in discounts:
        discounted_price = price * (1 - discounts[tier])
        return f"The discounted price is ${discounted_price:.2f}."
    return f"Invalid discount tier: {discount_tier}."


def get_tool_descriptions(tool_list):
    descriptions = []
    for t in tool_list:
        signature = inspect.signature(t.func)
        descriptions.append(f"{t.name}{signature} - {t.description}")
    return "\n".join(descriptions)


tools = [getProductPrice, applyDiscount]
tools_dict = {t.name: t for t in tools}
tool_descriptions = get_tool_descriptions(tools)

react_prompt = f"""You are a shopping assistant. You do NOT know any product prices; you must use tools.

    Tools:
    {tool_descriptions}

    Use EXACTLY this format:

    Thought: <your reasoning>
    Action: <tool name, exactly as listed>
    Action Input: <comma-separated key=value pairs>

    Then STOP and wait. The system will reply with:
    Observation: <tool result>

    Rules:
    - Output only ONE Thought/Action/Action Input per response, then stop.
    - Never write an Observation yourself.
    - Never answer from your own knowledge.
    - After each Observation, continue with the NEXT step. Never repeat an action you already did.
    - Only after you have all the Observations you need, write:
    Thought: I now know the final answer
    Final Answer: <answer>

    Example:
    Question: What is the price of headphones with the vip discount?
    Thought: I need the catalog price first.
    Action: getProductPrice
    Action Input: product=headphones
    Observation: The price of headphones is $199.99.
    Thought: Now I apply the discount.
    Action: applyDiscount
    Action Input: price=199.99, discount_tier=vip
    Observation: The discounted price is $139.99.
    Thought: I now know the final answer
    Final Answer: The headphones cost $139.99 with the vip discount.

    Question: {{question}}
"""

question = "What is the price of a laptop after applying the premium discount?"
messages = [HumanMessage(content=react_prompt.format(question=question))]

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

for iteration in range(1, MAX_ITERATIONS + 1):
    print(f"\n--- Iteration {iteration} ---")

    response = llm.invoke(messages, stop=["\nObservation"])
    output = response.content
    print(f"LLM Output:\n{output}")

    action_match = re.search(r"^Action:\s*(.+)$", output, re.MULTILINE)
    action_input_match = re.search(r"^Action Input:\s*(.+)$", output, re.MULTILINE)

    if action_match and action_input_match:
        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"[Tool Selected] {tool_name}")
        print(f"[Tool Input] {tool_input_raw}")

        args = {}
        for part in tool_input_raw.split(","):
            if "=" in part:
                key, value = part.split("=", 1)
                args[key.strip()] = value.strip().strip("'\"")

        if tool_name not in tools_dict:
            observation = (
                f"Error: Tool '{tool_name}' not found. "
                f"Available tools: {list(tools_dict.keys())}"
            )
        else:
            try:
                observation = str(tools_dict[tool_name].invoke(args))
            except Exception as e:
                observation = f"Error executing tool: {e}"

        print(f"[Tool Result] {observation}")

        messages.append(AIMessage(content=output))
        messages.append(
            HumanMessage(
                content=f"Observation: {observation}\n"
                "Continue with the next step. Do not repeat previous actions."
            )
        )
        continue

    final_answer_match = re.search(r"Final Answer:\s*(.+)", output, re.DOTALL)

    if final_answer_match:
        print("\n" + "=" * 60)
        print(f"Final Answer: {final_answer_match.group(1).strip()}")
        break

    print("ERROR: Could not parse Action/Action Input or Final Answer")
    break