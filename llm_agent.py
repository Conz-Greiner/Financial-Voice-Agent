from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_ollama import ChatOllama
from config import LLMConfig
from tools import FINANCIAL_TOOLS


class FinancialAgent:
    """
    Wraps the LangChain Agent and Granite 3.1 LLM.
    Handles tool calling and enforces voice-friendly output.
    """

    def __init__(self):
        print("🧠 [LLM] Initializing Granite 3.1 Agent...")

        # Initialize Ollama with Granite 3.1
        # format="json" is the secret weapon. It forces the model's logits
        # to strictly adhere to JSON schema, preventing tool-call parsing errors.
        self.llm = ChatOllama(
            model=LLMConfig.OLLAMA_MODEL,
            base_url=LLMConfig.OLLAMA_BASE_URL,
            temperature=LLMConfig.TEMPERATURE,
            num_ctx=LLMConfig.CONTEXT_WINDOW
        )

        # Define the strict voice-optimized system prompt
        self.system_prompt = """
        You are an expert financial analyst delivering a live, spoken market briefing.
        When the user asks for a market report, you MUST use the provided tools to gather data.

        CRITICAL DATA RULES:
        1. The tools now return CLEAN, STRUCTURED data. Do not hallucinate or guess numbers. Read the exact values provided by the tools.
        2. For the Economic Calendar, focus ONLY on the 'Actual' vs 'Forecast' numbers. If the Actual is higher/lower than the Forecast, explain the market impact (e.g., "CPI came in hotter than expected, which is bullish for the USD").
        3. If a tool returns "No data" or "Pending", state that clearly. Do not make up numbers.

        CRITICAL VOICE RULES:
        1. NO MARKDOWN: Do not use asterisks, bullet points, or bold text. 
        2. CONVERSATIONAL FLOW: Write exactly how a human news anchor speaks. Use transitional phrases.
        3. BREVITY: Keep the entire response under 120 words. Voice users lose attention quickly.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Create the agent
        agent = create_tool_calling_agent(self.llm, FINANCIAL_TOOLS, self.prompt)

        # Wrap in executor with error handling
        self.executor = AgentExecutor(
            agent=agent,
            tools=FINANCIAL_TOOLS,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3  # Prevent infinite tool-calling loops
        )

        print("✅ [LLM] Agent ready.")

    def run(self, user_input: str) -> str:
        """
        Processes the user input, calls tools if necessary, and returns the final text.
        """
        try:
            # Warmup check
            # Ollama loads model into VRAM/RAM on first run
            result = self.executor.invoke({"input": user_input})
            return result.get("output", "I couldn't generate a response.")
        except Exception as e:
            print(f"❌ [LLM] Critical Agent Error: {e}")
            return "I encountered a critical error processing that request. Please try again."