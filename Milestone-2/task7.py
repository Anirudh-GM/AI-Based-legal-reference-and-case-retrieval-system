import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Initialize GPT-3.5-turbo
chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Loop to take user input
while True:
    query = input("\nEnter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        print("Exiting...")
        break

    response = chat_model.invoke([HumanMessage(content=query)])
    print("\nChatGPT Response:\n")
    print(response.content)