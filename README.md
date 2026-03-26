# Equipment Recommender Thesis Project

## Description

Repo for my Thesis Project regarding an AI system that recommends equipment depending on the problem description given by user. This equipment should eventually link the user to researchers that work at the Rijksuniversiteit Groningen to together see if there problem is solvable with current equipment available at the RUG.

## Running the code

We make use of `uv` for handling dependencies, virtual environments etc.. Addiotionally, this improves reproducibilty. 

Run the code using `uv run python main.py`

## Initial setup (baseline)

Two things we need for the baseline:
- Dense Retrieval / semantic search
- Retrieval Augmented Generation (RAG)

## Formatting

We use Ruff for formatting. Important commands:

- `uv run ruff format .` This formats your code automatically (making indents, puts long code on new lines etc.)
- `uv run ruff check .`  Finds issues that are not solveable with formatting, similar to ESlint
- `uv run ruff check . --fix` Tries to fix these issues by itself

## Usage of models

As I am unsure on what models to use for RAG and embeddings, I start with using OpenAI's API to create the embeddings, as well as for the LLMs. Ideally this should be run locally (e.g. using Ollama) and tailored for our specific use case. However, this is now for an initial setup. And I still have a lot of tokens left for OpenAI.

In a `.env` that is hidden in the `.gitignore` is  my personal API key as credential. If others want to use it please add the following in a `.env` file:

`OPENAI_KEY= 'xxxxxxxxxxxxxxxxxxxxx'`

Where the x's should be replaced with your key. Please refer to OpenAI's documentation for how to use it.

## Usage of LLMs for coding

All code is written by hand. LLM is only used as replacement to go through every documentation of libraries as this saves a lot of time. The code itself is written by hand for educational purposes as I believe before using AI for writing code, you need to be able to write code confidently yourself.

Additionally, it is used for advice for structure of the project code, and for debugging.
