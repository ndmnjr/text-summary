import os, streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate, LLMChain

# Streamlit app
st.subheader('Test Langchain Text Summary')

# Get OpenAI API key and source text input
openai_api_key = st.text_input("OpenAI API Key", type="password")
character_input = st.text_input("Specify the maximum number of characters ", type="default")
source_text = st.text_area("Source Text", height=200)

# If the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key.strip() or not source_text.strip():
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
              chain = None
              if not character_input:
                  chain = load_summarize_chain(llm, chain_type="map_reduce")
              else:
                  try:
                      prompt_template = """Write a concise summary of the following in less than {character_input} characters:
        
        
                        {source_text}
                        """
                      PROMPT = PromptTemplate(template=prompt_template, input_variables=["character_input","source_text"])
                      chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
                  except Exception as e:
                     st.write(e) 
              
              if chain is None:
                  st.write("chain is none")
                  chain = load_summarize_chain(llm, chain_type="map_reduce")
              summary = chain.run(docs)

              st.success(summary)
        except Exception as e:
            st.exception(f"An error occurred: {e}")
