from dotenv import load_dotenv
load_dotenv()

from langchain  import __version__ as core_version
from langgraph.version import __version__ as graph_version
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

print(f"langchain-core version: {core_version}")
print(f"langgraph-core version: {graph_version}")


def main():
    # Test Gemini
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    response_gemini = gemini_llm.invoke("Say 'setup complete!' in one word")
    print("Gemini response:", response_gemini.content)

    # Test Anthropic
    # anthropic_llm = ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0)
    # response_anthropic = anthropic_llm.invoke("Say 'setup complete!' in one word")
    # print("Anthropic response:", response_anthropic.content)

    print("Setup Complete")

if __name__ == "__main__":
    main()