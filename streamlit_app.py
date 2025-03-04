import streamlit as st
import requests
import time

# FastAPI backend URL
API_URL = "http://localhost:8000"

def main():
    st.title("JobSync AI – Find Your Dream Job")
    st.write("Ask anything about job opportunities!")

    if "jobs" not in st.session_state:
        st.session_state.jobs = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("Search for Jobs")
    job_query = st.text_input("Enter your job query (e.g., 'Software Engineer jobs in Bangalore')", "")
    if st.button("Find Jobs"):
        if job_query:
            with st.spinner("Finding the best matches for you..."):
                time.sleep(2)  # Give FastAPI time to bind
                try:
                    response = requests.post(f"{API_URL}/jobs", json={"query": job_query})
                    response.raise_for_status()
                    result = response.json()
                    st.session_state.jobs = result["jobs"]
                    st.success("Here are the job recommendations based on your query:")
                    st.markdown(result["message"], unsafe_allow_html=True)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching jobs: {e}")
        else:
            st.warning("Please enter a job query.")

    st.subheader("Chat with JobSync AI")
    if st.session_state.jobs:
        chat_query = st.text_input("Ask a question about the jobs (e.g., 'What are the requirements for the Infosys job?')", "")
        if st.button("Send"):
            if chat_query:
                payload = {"query": chat_query, "previous_jobs": st.session_state.jobs}
                try:
                    response = requests.post(f"{API_URL}/chat", json=payload)
                    response.raise_for_status()
                    chat_response = response.json()["response"]
                    st.session_state.chat_history.append((chat_query, chat_response))
                    st.markdown("### Chat History")
                    for user_q, ai_r in st.session_state.chat_history:
                        st.write(f"**You**: {user_q}")
                        st.markdown(f"**JobSync AI**: {ai_r}", unsafe_allow_html=True)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error chatting: {e}")
            else:
                st.warning("Please enter a chat query.")
    else:
        st.info("Search for jobs first to start chatting!")

if __name__ == "__main__":
    main()