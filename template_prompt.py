from dotenv import load_dotenv
# Cohere
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
# Groq
from langchain_groq import ChatGroq
# Hugging Face
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# Mistral
from langchain_mistralai import ChatMistralAI
# Ollama
from langchain_ollama import ChatOllama
# OpenRouter
from langchain_openai import ChatOpenAI

load_dotenv()


# ============================================================
# 1. Groq
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
)


# ============================================================
# 2. Google Gemini
# ============================================================

# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.8-flash",
# )


# ============================================================
# 3. Mistral
# ============================================================

# llm = ChatMistralAI(
#     model="mistral-small-latest",
# )


# ============================================================
# 4. Cohere
# ============================================================

# llm = ChatCohere(
#     model="command-a-03-2025",
# )


# ============================================================
# 5. Hugging Face
# ============================================================

# hf_llm = HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-120b",
#     task="text-generation",
# )
#
# llm = ChatHuggingFace(
#     llm=hf_llm
# )


# ============================================================
# 6. Ollama - Local
# ============================================================

# llm = ChatOllama(
#     model="qwen3",
# )


# ============================================================
# 7. OpenRouter
# ============================================================

# llm = ChatOpenAI(
#     model="YOUR_FREE_MODEL",
#     base_url="https://openrouter.ai/api/v1",
#     api_key=None,
# )

info = """
    Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman, industrialist, and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025, and briefly became the only trillionaire (in terms of US dollars) in June 2026; as of September 2026, Forbes estimates his net worth to be US$964 billion.

Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded Zip2, a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.

In 2002, Musk founded and became CEO and chief engineer of SpaceX, a space technology company; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, Musk co-founded OpenAI to advance artificial intelligence (AI) research, but later left; his growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired Twitter, a social networking service, and made significant changes, including rebranding it as X in 2023. His other businesses include Neuralink, a neurotechnology company that he co-founded in 2016, and the Boring Company, a tunneling company that he founded in 2017. In November 2025, Tesla approved a pay package worth $1 trillion for Musk, which he is to receive over 10 years if certain milestones are met, such as achieving a market capitalization of $8.5 trillion. Musk became the first US-dollar trillionaire upon the initial public offering of SpaceX in June 2026, though a SpaceX stock collapse brought him under the threshold by the next month.
"""

summary_template = """
    Given the following information, summarize it in a few sentences:

    1. Short summary
    2. Two intersecting facts about them

    Information:
    {info}
"""

summary_prompt_template = PromptTemplate(input_variables=["info"] , template = summary_template)

chain = summary_prompt_template | llm

response = chain.invoke({"info": info})
# response = llm.invoke("Explain RAG simply")
print(response.text)
