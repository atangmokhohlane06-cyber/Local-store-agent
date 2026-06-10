import os
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_ollama import ChatOllama
from store import TOOLS

# 1. Initialize the local inference client via Ollama
llm = ChatOllama(model="qwen2.5:0.5b", temperature=0)

# 2. Hardcode the official open-source ReAct system instructions.
# This eliminates the dependency on the 'hub' download entirely!
react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

# Convert the string into a structured LangChain prompt object
prompt = PromptTemplate.from_template(react_template)

# 3. Construct the logic state graph connecting the model and our tools list
agent = create_react_agent(llm, TOOLS, prompt)

# 4. Wrap the agent pattern into an executor framework to run loops
executor = AgentExecutor(
    agent=agent,
    tools=TOOLS,
    verbose=True,  # Prints Thought/Action chains into your terminal window
    max_iterations=5,  # Protects local RAM by stopping endless execution loops
    handle_parsing_errors=True,  # Handles output text mismatches safely
)

if __name__ == "__main__":
    print("🤖 Local Store Agent Active. Type your question or 'exit' to quit.\n")
    while True:
        try:
            q = input("You: ")
            if q.strip().lower() in ["exit", "quit"]:
                break

            out = executor.invoke({"input": q})
            print(f"\nBot: {out['output']}\n")

        except KeyboardInterrupt:
            print("\nShutting down safely.")
            break
