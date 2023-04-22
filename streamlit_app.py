import os
from llama_index import download_loader, GPTSimpleVectorIndex, GPTListIndex, LLMPredictor, ServiceContext, ComposableGraph
from pathlib import Path
import streamlit as st
from langchain import OpenAI

def get_openai_api_key():
    openai_api_key = st.sidebar.text_input("OpenAI API Key")
    os.environ['OPENAI_API_KEY'] = openai_api_key

    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("Please enter your OpenAI API key in the sidebar to use LlamaIndex.")

    return openai_api_key

def create_index_set_resource(data_directory):
    # Load the 10-K data into memory
    UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)
    loader = UnstructuredReader()
    doc_set = {}
    all_docs = []
    years = [2022, 2021, 2020, 2019]
    for year in years:
        year_docs = loader.load_data(file=Path(f'{data_directory}/UBER/UBER_{year}.html'), split_documents=False)
        for d in year_docs:
            d.extra_info = {"year": year}
        doc_set[year] = year_docs
        all_docs.extend(year_docs)

    index_set = {}
    for year in years:
        cur_index = GPTSimpleVectorIndex.from_documents(doc_set[year])
        index_set[year] = cur_index

    return index_set

def risk_factors_query(index_set, year):
    if index_set:
        query_results(index_set, year, "risk factors")
    else:
        st.warning("Please enter your OpenAI API key in the sidebar to use LlamaIndex.")

        
def query_results(index_set, year, query_str):
    response = index_set[year].query(query_str, similarity_top_k=3)
    for r in response:
        st.write(r.text)

def composable_graph_query(data_directory, risk_query_str):
    graph = load_graph(data_directory)
    query_configs = [
        {
            "index_struct_type": "dict",
            "query_mode": "default",
            "query_kwargs": {
                "similarity_top_k": 1,
            }
        },
        {
            "index_struct_type": "list",
            "query_mode": "default",
            "query_kwargs": {
                "response_mode": "tree_summarize",
            }
        },
    ]
    response_summary = graph.query(risk_query_str, query_configs=query_configs)
    st.write(response_summary)
    st.write(response_summary.get_formatted_sources())

def global_query(index_set, risk_query_str):
    years = [2022, 2021, 2020, 2019]
    st.write("Response for all years:")
    for year in years:
        st.write(f"Response for year {year}:")
        query_results(index_set, year, risk_query_str)

cache = {}

def get_index_set(data_directory):
    if "index_set" not in cache:
        cache["index_set"] = load_data(data_directory)
    return cache["index_set"]

def app():
    st.set_page_config(
    page_title="🦙🔒🎯 LlamaLock: Target Your Search with Llama-like Accuracy!",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

    # Add some CSS to make the page title stand out
    st.markdown(
    """
    <style>
    .css-1aumxhk h1{
        font-size: 72px !important;
        color: green !important;
        animation: bounce 2s infinite;
    }
    @keyframes bounce {
        0% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    st.sidebar.title("🦙🔒 LlamaLock App")
    
    openai_api_key = get_openai_api_key()

    if openai_api_key:
        data_directory = st.sidebar.text_input("Data Directory", "./data")
        index_set = get_index_set(data_directory)

    years = [2022, 2021, 2020, 2019]
    year = st.sidebar.selectbox("Year", years)

    st.header("Risk Factors Query")
    risk_factors_query(index_set, year)

    st.header("Composable Graph Query")
    query_str = st.text_input("Composable graph query string:", "What are some of the biggest risk factors in each year?")
    if st.button("Execute Composable Graph Query"):
        composable_graph_query(data_directory, query_str)

    st.header("Global Query")
    query_str = st.text_input("Global query string:", "What are some of the biggest risk factors in each year?")
    if st.button("Execute Global Query"):
        global_query(index_set, query_str)

if __name__ == "__main__":
    app()

