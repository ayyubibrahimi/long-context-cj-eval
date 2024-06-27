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

4. Configure the models:
- Open the `main.py` file.
- Adjust the model lists for each processing type as needed:
  ```python
  allContextModels = ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229"]
  allPagesModels = ["claude-3-haiku-20240307", "mistralai/Mixtral-8x22B"]
  nerModels = ["claude-3-haiku-20240307", "mistralai/Mixtral-8x22B"]
  ```

5. Run the script:

The script will process the input files using all specified models and processing types. It will generate LLM output, evaluate the performance, and save the evaluation results for each model and processing type.

## Processing Types and Models

The script supports three processing types:

1. **allContext**: Extracts entities from the full document that's been stuffed whole into the context window. 
2. **allPages**: Extracts entities by iterating over every page. 
3. **ner**: Preprocesses documents using Named Entity Recognition by finding the 20 pages with the most entities. Extracts entities from these 20 pages.

You can specify different models for each processing type by adjusting the model lists in the `main.py` file.

## Directory Structure

The script uses the following directory structure for outputs:

- LLM output: `model/{processing_type}/data/output/{model_name}/`
- Evaluation results: `evaluation/data/output/{processing_type}/{model_name}/`

Where:
- `{processing_type}` is one of: `allContext`, `allPages`, or `ner`
- `{model_name}` is the name of the model being used

## Results

The evaluation results will be saved in the following locations:

1. Individual model results: 
`evaluation/data/output/{processing_type}/{model_name}/results_{model_name}.csv`

2. Combined results for each processing type: 
`evaluation/data/output/{processing_type}/combined_results.csv`

These CSV files contain metrics such as Jaro-Winkler similarity and exact match ratios for each processed file.

## Extending the Script

To add new processing types or models:

1. Create a new `process_query` function in an appropriate module.
2. Import the new function in `main.py`.
3. Add a new model list for the processing type.
4. Add a call to `process_models()` with the new model list, processing type, and query function.

## Model Option