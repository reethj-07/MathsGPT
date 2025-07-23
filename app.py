# import os
# os.environ.pop("SSL_CERT_FILE", None)
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.chains import LLMMathChain, LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_community.utilities import WikipediaAPIWrapper
# from langchain.agents.agent_types import AgentType
# from langchain.agents import Tool, initialize_agent
# from langchain.callbacks import StreamlitCallbackHandler

# ## Set upi the Stramlit app
# st.set_page_config(page_title="Text To MAth Problem Solver And Data Serach Assistant",page_icon="🧮")
# st.title("Text To Math Problem Solver Using Google Gemma 2")

# groq_api_key=st.sidebar.text_input(label="Groq API Key",type="password")


# if not groq_api_key:
#     st.info("Please add your Groq APPI key to continue")
#     st.stop()

# llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)


# ## Initializing the tools
# wikipedia_wrapper=WikipediaAPIWrapper()
# wikipedia_tool=Tool(
#     name="Wikipedia",
#     func=wikipedia_wrapper.run,
#     description="A tool for searching the Internet to find the vatious information on the topics mentioned"

# )

# ## Initializa the MAth tool

# math_chain=LLMMathChain.from_llm(llm=llm)
# calculator=Tool(
#     name="Calculator",
#     func=math_chain.run,
#     description="A tools for answering math related questions. Only input mathematical expression need to be provided"
# )

# prompt="""
# You are a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
# and display it point wise for the question below
# Question:{question}
# Answer:
# """

# prompt_template=PromptTemplate(
#     input_variables=["question"],
#     template=prompt
# )

# ## Combine all the tools into chain
# chain=LLMChain(llm=llm,prompt=prompt_template)

# reasoning_tool=Tool(
#     name="Reasoning tool",
#     func=chain.run,
#     description="A tool for answering logic-based and reasoning questions."
# )

# ## initialize the agents

# assistant_agent=initialize_agent(
#     tools=[wikipedia_tool,calculator,reasoning_tool],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=False,
#     handle_parsing_errors=True
# )

# if "messages" not in st.session_state:
#     st.session_state["messages"]=[
#         {"role":"assistant","content":"Hi, I'm a MAth chatbot who can answer all your maths questions"}
#     ]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg['content'])

# ## LEts start the interaction
# question=st.text_area("Enter your question:",)

# if st.button("find my answer"):
#     if question:
#         with st.spinner("Generate response.."):
#             st.session_state.messages.append({"role":"user","content":question})
#             st.chat_message("user").write(question)

#             st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
#             response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb]
#                                          )
#             st.session_state.messages.append({'role':'assistant',"content":response})
#             st.write('### Response:')
#             st.success(response)

#     else:
#         st.warning("Please enter the question")

import os
import certifi
import streamlit as st
import numpy as np
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain_experimental.tools import PythonREPLTool

#IMPORTANT SSL FIX
os.environ["SSL_CERT_FILE"] = certifi.where()


#CUSTOM TOOL 
def solve_relative_speed_problem(input_str: str) -> str:
    """
    Solves relative speed problems where two objects move towards each other.
    The input must be a comma-separated string of three numbers:
    total_distance, speed1, speed2.
    For example: '440, 60, 50'
    """
    try:
        parts = [float(p.strip()) for p in input_str.split(',')]
        if len(parts) != 3:
            return "Error: Please provide exactly three numbers: total_distance, speed1, speed2."

        total_distance, speed1, speed2 = parts
        
        if (speed1 + speed2) == 0:
            return "Error: The combined speed cannot be zero."

        time_to_meet = total_distance / (speed1 + speed2)
        distance_from_a = speed1 * time_to_meet
        
        return (f"The two objects will meet after {time_to_meet:.2f} hours. "
                f"They will meet at a distance of {distance_from_a:.2f} km from the starting point of the first object (traveling at {speed1} km/h).")
    except Exception as e:
        return f"An error occurred: {e}. Please ensure the input is a comma-separated string of three numbers."
# ------------------------------------


## Set up the Streamlit app
st.set_page_config(page_title="MathsGPT Problem Solver", page_icon="🧮")
st.title("MathsGPT: Problem Solver & Search Assistant")

# Sidebar for API Key input
with st.sidebar:
    groq_api_key = st.text_input(label="Groq API Key", type="password", key="groq_api_key")
    st.info("This app uses specialized tools to solve problems. If a problem causes a loop, try rephrasing it.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, I'm MathsGPT. How can I help you solve a problem today?"}]

# Display past messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Main app logic
if not groq_api_key:
    st.info("Please add your Groq API key in the sidebar to continue.")
    st.stop()

# Initialize the LLM
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key, temperature=0)

#TOOL DEFINITIONS WITH IMPROVED DESCRIPTIONS

# 1. Specialized Relative Speed Tool
relative_speed_tool = Tool(
    name="RelativeSpeedProblemSolver",
    func=solve_relative_speed_problem,
    description="Use this tool for 'meeting point' or 'relative speed' problems. It is the perfect choice for questions about two trains, cars, or objects traveling towards each other from a known distance. The input must be a single string containing three comma-separated numbers: total_distance, speed_of_object_1, speed_of_object_2."
)

# 2. General Python Calculator
python_repl_tool = PythonREPLTool()
python_calculator_tool = Tool(
    name="PythonCalculator",
    func=python_repl_tool.run,
    description="A powerful general-purpose calculator that can execute Python code. Use this for complex math problems that do NOT have a more specialized tool available. Always check if a specialized tool like RelativeSpeedProblemSolver is a better fit before using this."
)

# 3. Wikipedia Tool
# --- THIS LINE WAS MISSING - IT IS NOW FIXED ---
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A useful tool for searching Wikipedia for general knowledge and definitions."
)


# --- AGENT INITIALIZATION ---
# Give the agent the list of all available tools
assistant_agent = initialize_agent(
    tools=[relative_speed_tool, python_calculator_tool, wikipedia_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

# Handle user input
if question := st.chat_input("Enter your question here..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        response = assistant_agent.run(question, callbacks=[st_cb])
        
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)