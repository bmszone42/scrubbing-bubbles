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

        # create index and save to disk
        cur_index = GPTSimpleVectorIndex.from_documents(year_docs)
        cur_index.save_to_disk(f"index_{year}.json")

    # create index set from saved indices
    index_set = {}
    for year in years:
        cur_index = GPTSimpleVectorIndex.load_from_disk(f"index_{year}.json")
        index_set[year] = cur_index

    return index_set
        
def query_results(index_set, year, query_str):
    response = index_set[year].query(query_str, similarity_top_k=3)
    st.write(response)
    
def risk_factors_query(index_set, year):
    risk_query_str = (
        "Describe the current risk factors. If the year is provided in the information, "
        "provide that as well. If the context contains risk factors for multiple years, "
        "explicitly provide the following:\n"
        "- A description of the risk factors for each year\n"
        "- A summary of how these risk factors are changing across years"
    )
    query_results(index_set, year, risk_query_str)

    
def composable_graph_query(index_set, risk_query_str):
    years = [2022, 2021, 2020, 2019]
    graph = ComposableGraph(index_struct=index_set, docstore=doc_set)

    for year in years:
        graph.add_index(index_set[year], index_struct_type="dict")

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

    # Create a ComposableGraph from indices
    graph = ComposableGraph.from_indices(
        GPTListIndex,
        [index_set[y] for y in years],
        service_context=service_context
    )

    # Save the graph to disk
    graph.save_to_disk('10k_graph.json')

    # Load the graph from disk
    graph = ComposableGraph.load_from_disk('10k_graph.json', service_context=service_context)

    # Query the global index
    response = graph.query(risk_query_str, query_configs=[{
        "index_struct_type": "list",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 4
        }
    }])

    st.write(response)

cache = {}

def get_index_set(data_directory):
    if "index_set" not in cache:
        cache["index_set"] = create_index_set_resource(data_directory)
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

        # Set number of output tokens
        #llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512))
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=512, model_name="text-davinci-003"))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)


if __name__ == "__main__":
    app()

