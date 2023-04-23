# 🦙🔒🎯 LlamaLock: Target Your Search with Llama-like Accuracy!

This app allows you to search through UBER's 10-K reports from 2019 to 2022, providing summaries of risk factors for each year. It utilizes the Llama Index library, which includes the GPTSimpleVectorIndex and GPTListIndex for indexing the documents, and ComposableGraph for querying the indices.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Dependencies](#dependencies)
4. [License](#license)

## Installation

To install the required dependencies, run:

\`\`\`
pip install -r requirements.txt
\`\`\`

## Usage

1. Run the app with:

\`\`\`bash
streamlit run app.py
\`\`\`

2. Enter your OpenAI API Key in the sidebar.

3. Set the data directory (default: "./data").

4. Select a year from the sidebar.

5. View the Risk Factors Query results, Composable Graph Query results, and Global Query results.

## Dependencies

- streamlit
- openai
- llama_index
- langchain

## License

This project is licensed under the MIT License.
