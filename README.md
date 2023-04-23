LlamaLock App
Welcome to LlamaLock, a powerful tool for searching and analyzing 10-K financial reports. This app allows you to perform searches and queries on a set of financial reports from the ride-sharing company Uber.

Installation and Setup
To use this app, please ensure that you have the following installed on your machine:

Python 3.6 or higher
pip package manager
To install the required dependencies, run the following command in your terminal:

pip install -r requirements.txt

Usage
To launch the LlamaLock app, run the following command in your terminal:

streamlit run app.py

This will open the app in your web browser.

Once the app is running, you will be prompted to enter your OpenAI API key in the sidebar. Please enter your API key and click "Enter".

Next, you will need to specify the location of the data directory containing the Uber 10-K reports. You can do this by entering the path to the directory in the "Data Directory" field in the sidebar.

Once you have specified the data directory, you can use the app to perform searches and queries on the 10-K reports.

Risk Factors Query
The "Risk Factors Query" section allows you to search for descriptions of current risk factors. To perform a risk factors query, select the year of the report you wish to search and click "Search". The app will return a list of the top 3 most similar documents to your query.

Composable Graph Query
The "Composable Graph Query" section allows you to perform more complex searches and queries using the ComposableGraph class from the llama_index library. To perform a composable graph query, enter your query string in the "Composable graph query string" field and click "Execute Composable Graph Query". The app will return a summary of the query results, as well as a formatted list of the sources of the documents returned.

Global Query
The "Global Query" section allows you to search across all available years of the 10-K reports. To perform a global query, enter your query string in the "Global query string" field and click "Execute Global Query". The app will return a list of the top 4 most similar documents to your query.

Limitations
Please note that the LlamaLock app is designed to work with a specific set of 10-K reports from the ride-sharing company Uber. The app may not function correctly if used with other datasets.

Additionally, the LlamaLock app relies on the OpenAI GPT-3.5 language model and the llama_index library to perform its searches and queries. As such, the app's performance may be affected by changes to these libraries.

Support
If you encounter any issues while using the LlamaLock app, please feel free to open an issue on the project's GitHub page: https://github.com/llamalock/llamalock-app.

Acknowledgments
The LlamaLock app was created as part of the OpenAI API hackathon. We would like to thank OpenAI for providing access to their GPT-3.5 language model, as well as the creators of the llama_index library for their work on developing powerful search tools for natural language data.
