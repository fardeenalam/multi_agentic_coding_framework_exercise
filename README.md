# Multi-Agent Coding System

This project implements a multi-agent software development system using LangChain and LangGraph.
The system takes a natural language software requirement and processes it end to end using multiple specialized agents to generate code, review it, document it, generate tests, and prepare basic deployment artifacts.

A Streamlit UI is provided to interact with and demonstrate the system.

---

## How to Run the Project

Follow these steps exactly.

### 1. Clone the repository

    git clone https://github.com/fardeenalam/multi_agentic_coding_framework_exercise
    cd MULTI_AGENT_CODER

---

### 2. Install dependencies

It is recommended to use a virtual environment, but it is not mandatory.

    pip install -r requirements.txt

---

### 3. Set up environment variables

You must provide an OpenAI API key.

Create a file named `.env` in the project root with the following content:

    OPENAI_API_KEY=your_openai_api_key_here

Do not commit this file.

---

### 4. Run the Streamlit application

    streamlit run streamlit_app.py

---

### 5. Use the system

1. A browser window will open with the Streamlit UI.
2. Enter a software requirement in plain English.
3. Click the Run button.
4. The system will execute the multi-agent workflow.
5. Generated outputs will be displayed in the UI:
   - Refined requirements
   - Generated Python code
   - Code review result
   - Documentation
   - Test cases
   - Deployment configuration
6. The terminal where Streamlit is running will show streaming logs from the agents as the workflow progresses.

---

## Sample Outputs

The repository includes a `testresults/` folder containing sample outputs from multiple runs of the system.
These are provided only for reference and to demonstrate expected behavior.

---

## Project Overview

This project demonstrates a multi-agent software development workflow built using LangChain and LangGraph.

At a high level:
- A user provides a natural language software requirement.
- The requirement is processed iteratively by specialized agents.
- Each agent has a clearly defined responsibility.
- Agents collaborate through a graph-based workflow.
- The system produces production-ready artifacts without human intervention.

The Streamlit UI acts only as a presentation layer to interact with the agentic backend.

---

## Technologies Used

- Python
- LangChain
- LangGraph
- OpenAI GPT models
- Streamlit

---

## Agents Overview

**Requirement Analysis Agent**\
Transforms the user’s informal software request into a precise, structured, implementation-ready requirement. It defines system purpose, supported operations, inputs, outputs, error handling, assumptions, and non-goals.

**Coding Agent**\
Generates functional Python code strictly based on the refined requirement. The output is a single self-contained Python module with clear structure, validation, and error handling.

**Code Review Agent**\
Reviews the generated code against the refined requirement. It verifies correctness, completeness, and alignment with the specification. The agent either approves the code or provides actionable feedback for iteration.

**Documentation Agent**\
Produces structured technical documentation for the generated code. It explains the design, public API, data flow, usage examples, and limitations without modifying the code.

**Test Case Generation Agent**\
Generates executable unit tests and basic integration tests using pytest. The tests validate normal behavior, error conditions, and end-to-end flows as defined in the requirement.

**Deployment Configuration Agent**\
Generates minimal deployment artifacts required to run the generated code locally. This includes a requirements.txt file if needed and a run.sh script to execute the application.

**Streamlit UI (Presentation Layer)**\
Provides a simple frontend to interact with the multi-agent system. It allows users to input requirements and view all generated outputs while streaming agent progress logs in the console.

---

## Notes

- All generated code is in-memory and non-persistent.
- Each run produces a single Python module as output.
- The system prioritizes clarity, correctness, and deterministic behavior over over-engineering.
