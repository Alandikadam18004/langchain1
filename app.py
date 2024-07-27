import os
import cohere
import streamlit as st
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# Set Cohere API key
api_key = 'cv2iT1eJhSikUq7HcUeIjiCSKUQdOH1dsvE9HgTB'  # Initialize with your API key
cohere_client = cohere.Client(api_key)
os.environ['COHERE_API_KEY'] = 'cv2iT1eJhSikUq7HcUeIjiCSKUQdOH1dsvE9HgTB'

# Streamlit app framework
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here')

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Write me a YouTube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='Write me a YouTube video script based on this title TITLE: {title} while leveraging this Wikipedia research: {wikipedia_research}'
)

# Memory for conversation history
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Initialize language model
llm = Cohere(client=cohere_client, temperature=0.9)

# Create chains for generating titles and scripts
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

# Initialize Wikipedia API wrapper
wiki = WikipediaAPIWrapper()

# Process user input and generate results
if prompt:
    with st.spinner('Generating title...'):
        try:
            # Generate YouTube video title
            title = title_chain.run({"topic": prompt})
        except Exception as e:
            st.error(f"Error generating title: {e}")
            title = None

    if title:
        with st.spinner('Fetching Wikipedia research...'):
            try:
                # Fetch Wikipedia research data
                wiki_research = wiki.run(prompt)
            except Exception as e:
                st.error(f"Error fetching Wikipedia research: {e}")
                wiki_research = None

        if wiki_research:
            with st.spinner('Generating script...'):
                try:
                    # Generate YouTube video script
                    script = script_chain.run({"title": title, "wikipedia_research": wiki_research})
                except Exception as e:
                    st.error(f"Error generating script: {e}")
                    script = None

            # Display results
            if title:
                st.write("**Title:**", title)

            if script:
                st.write("**Script:**", script)

            # Expandable sections for history and research
            with st.expander('Title History'):
                st.info(title_memory.buffer)

            with st.expander('Script History'):
                st.info(script_memory.buffer)

            with st.expander('Wikipedia Research'):
                st.info(wiki_research)