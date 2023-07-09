import os, streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain

# Streamlit app
st.subheader('Test Langchain Text Summary')

# Get OpenAI API key and source text input
openai_api_key = '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'
source_text = st.text_area("Source Text", height=200)

# If the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not source_text.strip():
        st.error(f"Please provide the missing fields.")
    else:
        try:
            with st.spinner('Please wait...'):
              # Split the source text
              text_splitter = CharacterTextSplitter()
              texts = text_splitter.split_text(source_text)

              # Create Document objects for the texts (max 3 pages)
              docs = [Document(page_content=t) for t in texts[:3]]

              # Initialize the OpenAI module, load and run the summarize chain
              llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
              chain = load_summarize_chain(llm, chain_type="map_reduce")
              summary = chain.run(docs)

              st.success(summary)
        except Exception as e:
            st.exception(f"An error occurred: {e}")
