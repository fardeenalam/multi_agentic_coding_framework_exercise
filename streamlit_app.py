import streamlit as st
from agents import run_dev_flow

st.set_page_config(
    page_title="Multi-Agent Coding Assistant",
    layout="wide"
)

st.title("Multi-Agent Coding Assistant")
st.write(
    "Enter a software requirement below to see how the multi-agent system "
    "analyzes, implements, reviews, documents, tests, and prepares deployment artifacts."
)

st.divider()

# Input section
st.subheader("Software Requirement")

user_requirement = st.text_area(
    label="Describe the software you want to build",
    height=200,
    placeholder="Example: Build a Python program that manages a simple in-memory todo list..."
)

run_button = st.button("Run Multi-Agent System", type="primary")

# Run workflow
if run_button:
    if not user_requirement.strip():
        st.warning("Please enter a software requirement before running the system.")
    else:
        with st.spinner("Running multi-agent workflow..."):
            final_state = run_dev_flow(user_requirement)

        st.success("Workflow completed successfully.")

        st.divider()

        # Refined Requirement
        st.subheader("Refined Requirement")
        st.markdown(final_state.get("refined_requirement", ""))

        # Review Summary
        st.subheader("Code Review Result")
        approved = final_state.get("code_approved", False)
        review_attempts = final_state.get("review_attempts", 0)

        if approved:
            st.success(f"Code approved after {review_attempts} review attempt(s).")
        else:
            st.error("Code was not approved.")
            st.write("Review Feedback:")
            st.write(final_state.get("review_feedback", ""))

        # Generated Code
        st.subheader("Generated Python Code")
        st.code(final_state.get("code", ""), language="python")

        # Documentation
        st.subheader("Documentation")
        st.markdown(final_state.get("documentation", ""))

        # Test Cases
        st.subheader("Generated Test Cases")
        st.code(final_state.get("test_cases", ""), language="python")

        # Deployment Files
        st.subheader("Deployment Configuration")

        deployment_files = final_state.get("deployment_files", {})

        if deployment_files:
            for filename, content in deployment_files.items():
                with st.expander(filename):
                    if filename.endswith(".sh"):
                        st.code(content, language="bash")
                    else:
                        st.code(content)
        else:
            st.write("No deployment files generated.")
