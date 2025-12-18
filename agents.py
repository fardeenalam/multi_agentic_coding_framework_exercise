from langgraph.graph import StateGraph, END
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from typing import TypedDict, Annotated
import operator
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

from prompts import REQUIREMENT_ANALYSIS_PROMPT, CODING_AGENT_PROMPT, REVIEW_AGENT_PROMPT_BALANCED, DOCUMENTATION_AGENT_PROMPT, TEST_AGENT_PROMPT, DEPLOYMENT_AGENT_PROMPT

load_dotenv()

llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


class DevState(TypedDict):
    user_input: str
    refined_requirement: str

    code: str
    review_feedback: str
    code_approved: bool
    review_attempts: int

    documentation: str
    test_cases: str
    deployment_files: dict

    messages: Annotated[list, operator.add]
    next_agent: str


# Requirement Analysis Agent
class RequirementOutput(BaseModel):
    refined_requirement: str = Field(description="Clear, structured, implementation ready software requirement")


def requirement_agent(state: DevState) -> DevState:
    """Requirement Analysis Agent - Takes input in natural language and refines it into a
       structured software requirement."""
    
    structured_llm = llm.with_structured_output(RequirementOutput)

    prompt = REQUIREMENT_ANALYSIS_PROMPT.format(user_input=state["user_input"])

    result: RequirementOutput = structured_llm.invoke(prompt)

    return {
        **state,
        "refined_requirement": result.refined_requirement,
        "review_attempts": 0,
        "messages": [AIMessage(content="Requirement Agent: requirement refined")],
        "next_agent": "coding_agent"
    }


# Coding Agent
def coding_agent(state: DevState) -> DevState:
    """Coding Agent - Converts the refined requirements into functional Python code."""

    review_feedback_block = ""
    if state.get("review_feedback"):
        review_feedback_block = (
            "Review Feedback:\n" + state["review_feedback"]
        )

    prompt = CODING_AGENT_PROMPT.format(
        refined_requirement=state["refined_requirement"],
        review_feedback_block=review_feedback_block
    )

    code = llm.invoke(prompt).content

    return {
        **state,
        "code": code,
        "messages": [AIMessage(content="Coding Agent: code generated")],
        "next_agent": "review_agent"
    }


# Code Review Agent
class ReviewOutput(BaseModel):
    approved: bool = Field(description="Whether the code satisfies the requirement")
    feedback: str = Field(description="Feedback if code is not approved")


def review_agent(state: DevState) -> DevState:
    """Code Review Agent - Reviews the generated code for correctness, efficiency, and security.
    If improvements are needed, it provides feedback for re-iteration."""

    structured_llm = llm.with_structured_output(ReviewOutput)

    prompt = REVIEW_AGENT_PROMPT_BALANCED.format(
        refined_requirement=state["refined_requirement"],
        code=state["code"]
    )

    result: ReviewOutput = structured_llm.invoke(prompt)

    if result.approved:
        return {
            **state,
            "code_approved": True,
            "review_feedback": result.feedback,
            "messages": [AIMessage(content="Review Agent: code approved")],
            "next_agent": "documentation_agent"
        }
    
    attempts = state["review_attempts"] + 1

    if attempts >= 3:
        return {
            **state,
            "code_approved": False,
            "review_feedback": result.feedback,
            "review_attempts": attempts,
            "messages": [AIMessage(content="Review Agent: max attempts reached, revisiting requirement")],
            "next_agent": "requirement_agent"
        }
    
    return {
        **state,
        "code_approved": False,
        "review_feedback": result.feedback,
        "review_attempts": attempts,
        "messages": [AIMessage(content=f"Review Agent: issues found (attempt {attempts}), retrying coding")],
        "next_agent": "coding_agent"
    }


# Documentation Agent
def documentation_agent(state: DevState) -> DevState:
    """Documentation Agent - Generates proper documentation for the developed code."""

    prompt = DOCUMENTATION_AGENT_PROMPT.format(
        refined_requirement = state["refined_requirement"],
        code = state["code"]    
    )

    documentation = llm.invoke(prompt).content

    return {
        **state,
        "documentation": documentation,
        "messages": [AIMessage(content="Documentation Agent: documentation generated")],
        "next_agent": "test_agent"
    }


# Test Agent
def test_agent(state: DevState) -> DevState:
    """Test Case Generation Agent - Creates unit tests and integration test cases for the
    developed code."""

    prompt = TEST_AGENT_PROMPT.format(
        refined_requirement = state["refined_requirement"],
        code = state["code"]    
    )

    tests = llm.invoke(prompt).content

    return {
        **state,
        "test_cases": tests,
        "messages": [AIMessage(content="Test Agent: test cases generated")],
        "next_agent": "deployment_agent"
    }


# Deployment Configuration Agent
class DeploymentOutput(BaseModel):
    requirements_txt: str = Field(
        description="Contents of requirements.txt. Empty string if no dependencies."
    )
    run_sh: str = Field(
        description="Shell script to install dependencies and run the application."
    )


def deployment_agent(state: DevState) -> DevState:
    """Deployment Configuration Agent - Generates a deployment script to deploy the developed
    code."""
    
    structured_llm = llm.with_structured_output(DeploymentOutput)

    prompt = DEPLOYMENT_AGENT_PROMPT.format(
        refined_requirement = state["refined_requirement"],
        code = state["code"]    
    )

    result: DeploymentOutput = structured_llm.invoke(prompt)

    deployment_files = {
        "requirements.txt": result.requirements_txt.strip(),
        "run.sh": result.run_sh.strip()
    }

    return {
        **state,
        "deployment_files": deployment_files,
        "messages": [AIMessage(content="Deployment Agent: structured deployment files generated")],
        "next_agent": "end"
    }


def route_agent(state: DevState):
    return state.get("next_agent", "end")


# Graph creation
workflow = StateGraph(DevState)

# Adding all agent nodes in the graph
workflow.add_node("requirement_agent", requirement_agent)
workflow.add_node("coding_agent", coding_agent)
workflow.add_node("review_agent", review_agent)
workflow.add_node("documentation_agent", documentation_agent)
workflow.add_node("test_agent", test_agent)
workflow.add_node("deployment_agent", deployment_agent)

# Setting entry point
workflow.set_entry_point("requirement_agent")

# Adding conditional edges
workflow.add_conditional_edges(
    "requirement_agent",
    route_agent,
    {
        "coding_agent": "coding_agent"
    }
)

workflow.add_conditional_edges(
    "coding_agent",
    route_agent,
    {
        "review_agent": "review_agent"
    }
)

workflow.add_conditional_edges(
    "review_agent",
    route_agent,
    {
        "coding_agent": "coding_agent",
        "requirement_agent": "requirement_agent",
        "documentation_agent": "documentation_agent"
    }
)

workflow.add_conditional_edges(
    "documentation_agent",
    route_agent,
    {
        "test_agent": "test_agent"
    }
)

workflow.add_conditional_edges(
    "test_agent",
    route_agent,
    {
        "deployment_agent": "deployment_agent"
    }
)

workflow.add_conditional_edges(
    "deployment_agent",
    route_agent,
    {
        "end": END
    }
)

# Compiling the graph
app = workflow.compile()


def run_dev_flow(user_requirement: str):
    """
    Execute the complete 7-agent development workflow with detailed progress tracking.
    Outputs are formatted for professional review and documentation.
    """
    initial_state = {
        "user_input": user_requirement,
        "refined_requirement": "",
        "code": "",
        "review_feedback": "",
        "code_approved": False,
        "review_attempts": 0,
        "documentation": "",
        "test_cases": "",
        "deployment_files": {},
        "messages": [],
        "next_agent": ""
    }

    print("\n" + "=" * 100)
    print("MULTI-AGENT SOFTWARE DEVELOPMENT WORKFLOW")
    print("=" * 100)
    print(f"Input Requirement: {user_requirement[:150]}")
    if len(user_requirement) > 150:
        print(f"                   {user_requirement[150:300]}...")
    print("=" * 100 + "\n")

    final_state = None

    # Track agent execution
    for event in app.stream(initial_state):
        for node_name, node_state in event.items():
            final_state = node_state
            
            print(f"[Agent: {node_name.replace('_', ' ').title()}]")
            
            # Print agent message
            if "messages" in node_state and node_state["messages"]:
                latest = node_state["messages"][-1]
                status = latest.content.split(": ")[-1] if ": " in latest.content else latest.content
                print(f"  Status: {status}")
            
            # Agent-specific outputs
            if node_name == "requirement_agent" and node_state.get("refined_requirement"):
                lines = node_state["refined_requirement"].split("\n")
                print(f"  Output: {len(lines)} lines of structured requirements")
            
            elif node_name == "coding_agent" and node_state.get("code"):
                code_lines = len(node_state["code"].split("\n"))
                print(f"  Output: {code_lines} lines of Python code")
            
            elif node_name == "review_agent":
                attempts = node_state.get("review_attempts", 0)
                if node_state.get("code_approved"):
                    print(f"  Decision: APPROVED (after {attempts} iteration(s))")
                else:
                    print(f"  Decision: REJECTED (iteration {attempts} of 3)")
                    if node_state.get("next_agent") == "requirement_agent":
                        print(f"  Action: Returning to requirement analysis")
            
            elif node_name == "documentation_agent" and node_state.get("documentation"):
                doc_lines = len(node_state["documentation"].split("\n"))
                print(f"  Output: {doc_lines} lines of documentation")
            
            elif node_name == "test_agent" and node_state.get("test_cases"):
                test_lines = len(node_state["test_cases"].split("\n"))
                print(f"  Output: {test_lines} lines of test code")
            
            elif node_name == "deployment_agent" and node_state.get("deployment_files"):
                files = node_state["deployment_files"]
                req_len = len(files.get('requirements.txt', ''))
                sh_len = len(files.get('run.sh', ''))
                print(f"  Output: requirements.txt ({req_len} bytes), run.sh ({sh_len} bytes)")
            
            print()

    print("=" * 100)
    print("WORKFLOW EXECUTION COMPLETE")
    print("=" * 100)

    if final_state:
        # Summary Section
        print("\n" + "-" * 100)
        print("EXECUTION SUMMARY")
        print("-" * 100)
        print(f"Review Iterations:    {final_state.get('review_attempts', 0)}")
        print(f"Code Status:          {'Approved' if final_state.get('code_approved') else 'Not Approved'}")
        print(f"Code Size:            {len(final_state.get('code', '').split(chr(10)))} lines")
        print(f"Documentation Size:   {len(final_state.get('documentation', '').split(chr(10)))} lines")
        print(f"Test Suite Size:      {len(final_state.get('test_cases', '').split(chr(10)))} lines")
        
        # Refined Requirements
        print("\n\n" + "=" * 100)
        print("SECTION 1: REFINED REQUIREMENTS")
        print("=" * 100)
        print(final_state.get("refined_requirement", "Not generated"))
        
        # Generated Code
        print("\n\n" + "=" * 100)
        print("SECTION 2: GENERATED CODE")
        print("=" * 100)
        code = final_state.get("code", "")
        if code:
            print(code)
        else:
            print("No code was generated")
        
        # Review Feedback
        if final_state.get("review_feedback"):
            print("\n\n" + "=" * 100)
            print("SECTION 3: CODE REVIEW FEEDBACK")
            print("=" * 100)
            print(final_state["review_feedback"])
        
        # Documentation
        print("\n\n" + "=" * 100)
        print("SECTION 4: DOCUMENTATION")
        print("=" * 100)
        doc = final_state.get("documentation", "")
        if doc:
            print(doc)
        else:
            print("No documentation was generated")
        
        # Test Cases
        print("\n\n" + "=" * 100)
        print("SECTION 5: TEST CASES")
        print("=" * 100)
        tests = final_state.get("test_cases", "")
        if tests:
            print(tests)
        else:
            print("No test cases were generated")
        
        # Deployment Configuration
        print("\n\n" + "=" * 100)
        print("SECTION 6: DEPLOYMENT CONFIGURATION")
        print("=" * 100)
        if final_state.get("deployment_files"):
            files = final_state["deployment_files"]
            
            print("\n--- File: requirements.txt ---")
            req = files.get("requirements.txt", "")
            if req.strip():
                print(req)
            else:
                print("# No external dependencies required")
            
            print("\n--- File: run.sh ---")
            sh = files.get("run.sh", "")
            if sh.strip():
                print(sh)
            else:
                print("# No deployment script generated")
        else:
            print("No deployment files were generated")
        
        print("\n\n" + "=" * 100)
        print("END OF REPORT")
        print("=" * 100 + "\n")
    
    return final_state


def save_output_to_file(state, filename="workflow_output.txt"):
    """
    Save the workflow output to a text file for documentation purposes.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("MULTI-AGENT SOFTWARE DEVELOPMENT WORKFLOW - OUTPUT REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        # Summary
        f.write("EXECUTION SUMMARY\n")
        f.write("-" * 100 + "\n")
        f.write(f"Review Iterations:    {state.get('review_attempts', 0)}\n")
        f.write(f"Code Status:          {'Approved' if state.get('code_approved') else 'Not Approved'}\n")
        f.write(f"Code Size:            {len(state.get('code', '').split(chr(10)))} lines\n")
        f.write(f"Documentation Size:   {len(state.get('documentation', '').split(chr(10)))} lines\n")
        f.write(f"Test Suite Size:      {len(state.get('test_cases', '').split(chr(10)))} lines\n\n")
        
        # Requirements
        f.write("\n" + "=" * 100 + "\n")
        f.write("SECTION 1: REFINED REQUIREMENTS\n")
        f.write("=" * 100 + "\n")
        f.write(state.get("refined_requirement", "Not generated") + "\n")
        
        # Code
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("SECTION 2: GENERATED CODE\n")
        f.write("=" * 100 + "\n")
        f.write(state.get("code", "No code was generated") + "\n")
        
        # Review
        if state.get("review_feedback"):
            f.write("\n\n" + "=" * 100 + "\n")
            f.write("SECTION 3: CODE REVIEW FEEDBACK\n")
            f.write("=" * 100 + "\n")
            f.write(state["review_feedback"] + "\n")
        
        # Documentation
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("SECTION 4: DOCUMENTATION\n")
        f.write("=" * 100 + "\n")
        f.write(state.get("documentation", "No documentation was generated") + "\n")
        
        # Tests
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("SECTION 5: TEST CASES\n")
        f.write("=" * 100 + "\n")
        f.write(state.get("test_cases", "No test cases were generated") + "\n")
        
        # Deployment
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("SECTION 6: DEPLOYMENT CONFIGURATION\n")
        f.write("=" * 100 + "\n")
        if state.get("deployment_files"):
            files = state["deployment_files"]
            f.write("\n--- File: requirements.txt ---\n")
            req = files.get("requirements.txt", "")
            f.write(req if req.strip() else "# No external dependencies required\n")
            f.write("\n--- File: run.sh ---\n")
            sh = files.get("run.sh", "")
            f.write(sh if sh.strip() else "# No deployment script generated\n")
        else:
            f.write("No deployment files were generated\n")
        
        f.write("\n\n" + "=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    print(f"\nOutput saved to: {filename}")


if __name__ == "__main__":
    requirement = """Build a Python program that manages a simple in-memory contact book.
The program should allow adding a contact, searching for a contact by name, listing all contacts, and deleting a contact by name.
Each contact should have a name, phone number, and email address.
Names must be unique.
The system should handle invalid inputs gracefully."""
    
    result = run_dev_flow(requirement)
    
    # Optionally save to file
    if result:
        save_output_to_file(result, "contact_book_workflow_output.txt")