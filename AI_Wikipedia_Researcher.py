from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

# Initialize the OpenAI language model with an API key and set the temperature parameter
# Temperature 0.0 means the model's outputs will be more deterministic
llm = OpenAI(openai_api_key='todo add your own key', temperature=0.0)

# Load specific tools for the agent to use. In this case, Wikipedia and LLM-based math.
# The 'llm' argument ensures the tools use the initialized language model.
tools = load_tools(
    ['wikipedia', 'llm-math'], 
    llm=llm
)

# Initialize the agent with the loaded tools and language model.
# AgentType.ZERO_SHOT_REACT_DESCRIPTION specifies the agent type.
# Setting verbose=True enables detailed logging of the agent's operations.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

prompt = input('The Wikipedia research task: ')

# Run the agent with the provided prompt.
# The agent will use the loaded tools and language model to perform the task.
agent.run(prompt)