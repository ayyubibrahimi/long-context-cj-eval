# Long Context Evaluation for Criminal Justice Documents

## Overview
This repository is dedicated to evaluating the performance of long context windows on criminal justice documents. The project aims to analyze how well different models and techniques handle extensive context in legal text processing, focusing on criminal justice documents.

## Requirements

- Python 3.x
- Required Python packages (can be installed using `pip install -r requirements.txt`)

## Usage

1. Clone the repository:
- git clone https://github.com/ayyubibrahimi/long-context-cj-eval.git
- cd long-context-cj-eval

2. Install the required Python packages:
- pip install -r requirements.txt

3. Prepare the input data:
- Place the transcript files in the `ocr/data/input/transcripts` directory.
- Place the report files in the `ocr/data/input/reports` directory.
- Place the groundtruth files in the `evaluation/data/input` directory.

4. Adjust the model selection:
- Open the `main.py` file.
- Uncomment the desired `process_query` import statement based on the model you want to use:
  - For the model that preprocesses documents using NER, uncomment `from model.ner.src import process_query`.
  - For the model that iterates over all pages, uncomment `from model.allPages.src import process_query`.
  - For the model that uses the full context window, uncomment `from model.allContext.src import process_query`.

5. Set the LLM output directory and model:
- In the `if __name__ == "__main__":` block, set the `llm_output_directory` variable to the appropriate directory based on your model selection:
  - For the NER model, use `r"model/data/ner-20-ages"`.
  - For the allPages model, use `r"model/data/all-pages"`.
  - For the allContext model, use `r"model/data/all-context"`.
- Set the `model` variable to the desired model name (e.g., `"claude-3-sonnet-20240229"`, `"gpt-3.5-turbo-0125"`).

The script will process the input files, generate LLM output, evaluate the performance, and save the evaluation results in the `evaluation/data/output` directory. The results file will be named based on the selected model and the LLM output directory.

## Previously Run Models

The script has been previously run on the following models:
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-3-opus-20240229`
- `gpt-4-0125-preview`
- `gpt-3.5-turbo-0125`

## New Models to Consider

You can also consider running the script on the following new models:
- `gpt-4-turbo`
- `gpt-4o`

## Results

The evaluation results will be saved in the `evaluation/data/output` directory. The results file will be named `results_{directory_suffix}_{model_name}.csv`, where:
- `{directory_suffix}` is the name of the LLM output directory (e.g., `ner`, `all-pages`, `all-context`).
- `{model_name}` is the name of the selected model.
