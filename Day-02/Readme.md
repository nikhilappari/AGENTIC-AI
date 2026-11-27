Day 3 – Conversation Chains, Chat History, Testing, and Agent Introduction
Objective
Rebuild the Sweet Crumbs bakery assistant using LangChain prompt templates and an output parser. Add conversation memory, run simple CSV-based response checks, and explore LangChain's agent-creation interface.

Tasks
Install LangChain and the Groq integration.
Configure a Groq chat model.
Create a structured prompt with a history placeholder.
Build a LangChain chain using prompt, model, and output parser components.
Preserve conversation history between questions.
Create a CSV-driven test function.
Experiment with creating an agent without tools.
Implementation
1. Install packages and initialize the model
!pip install -q langchain langchain-groq

import os
from google.colab import userdata

os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

from langchain_groq import ChatGroq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3
)
The API key is loaded from Colab Secrets and made available to LangChain through the environment variable.

2. Create the prompt template
The notebook imports:

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
The system prompt defines the bakery assistant's role, menu prices, 24-hour order notice, eggless options, delivery-time restrictions, fallback response, and short friendly response format.

The prompt template contains three parts:

The system instructions.
MessagesPlaceholder("history"), which inserts earlier conversation messages.
The current customer question, supplied through {question}.
3. Build the LangChain chain
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()
The pipe operator connects the prompt template, language model, and string output parser. The result is a reusable chain that accepts input variables and returns a plain-text answer.

4. Maintain conversation history
history = []

def ask(question: str) -> str:
    answer = chain.invoke({
        "history": history,
        "question": question
    })
    history.extend([
        ("human", question),
        ("ai", answer)
    ])
    return answer
The ask() function passes the existing history and the new question into the chain. After receiving the answer, it adds both the user question and assistant answer to the history list.

The notebook tests this by introducing the customer as Ravi, asking whether eggless cakes are available, and then asking the assistant for the customer's name. The second question checks whether the conversation context is retained.

5. Run CSV-based response tests
The notebook defines:

def run_tests(path: str = "/content/test.csv") -> None:
    rows = list(csv.DictReader(open(path)))
    passed = 0
    for row in rows:
        answer = chain.invoke({
            "history": [],
            "question": row["question"]
        })
        ok = row["must_include"].lower() in answer.lower()
        passed += ok
        print("PASS" if ok else "FAIL", "|", row["question"], "→", answer[:70])
    print(f"\n{passed}/{len(rows)} passed ({passed / len(rows):.0%})")
Each CSV row is expected to contain question and must_include columns. The function sends each question to the chain with an empty history and checks whether the expected text appears in the answer, without considering letter case. It prints each test result and a final pass count.

The notebook calls run_tests() using its default path, /content/test.csv; that CSV file must be present in the Colab environment for the test to run.

6. Explore agent creation
The final cell imports create_agent and attempts to create an agent with the bakery model and system prompt, passing an empty tools list. The comment indicates that tools will be added later.

This is an introductory experiment only. The cell currently has inconsistent variable names: it assigns the agent to agent_order but invokes agent, and assigns the result to result_order but prints result. Those names need to be made consistent before the cell will run successfully.

Result
The main Day 3 implementation creates a reusable LangChain conversation chain with chat history and a simple CSV-based response evaluation function. The final cell introduces the next step: an agent that can later be extended with tools.

Notes
Add GROQ_API_KEY to Colab Secrets.
Provide /content/test.csv with question and must_include columns before calling run_tests().
The CSV test checks for a required substring; it is a simple smoke test, not a full evaluation of factual accuracy or response quality.
Conversation history is held in a Python list and is not persisted after the runtime ends.
Fix the variable-name mismatch in the final agent cell before running it.
