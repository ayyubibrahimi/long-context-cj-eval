import os
import logging
import pandas as pd
from model.allContext.src.src import process_query
from ocr.src.ocr import ocr_process
from evaluation.src.evaluation import (
    preprocess_data,
    compute_jaro_winkler_metrics,
    title_patterns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def ocr_files_exist(output_path):
    return os.path.exists(output_path) and len([f for f in os.listdir(output_path) if f.endswith('.json')]) > 0

def main(
    input_path_transcripts,
    input_path_reports,
    output_path_transcripts,
    output_path_reports,
    llm_output_directory,
    groundtruth_directory,
    model,
):
    # Check if OCR files already exist
    if ocr_files_exist(output_path_transcripts) and ocr_files_exist(output_path_reports):
        logging.info("OCR files already exist. Skipping OCR process.")
    else:
        # Process the PDF files using OCR
        logging.info("Starting OCR process...")
        ocr_process(input_path_transcripts, input_path_reports, output_path_transcripts, output_path_reports)

    # Process the OCR output files and generate LLM output
    logging.info("Starting LLM processing...")
    process_query(
        output_path_transcripts,
        output_path_reports,
        llm_output_directory,
        model
    )

    # Evaluate the LLM output
    logging.info("Starting evaluation...")
    results = []
    for file_name in os.listdir(llm_output_directory):
        if file_name.endswith(".csv"):
            llm_output_path = os.path.join(llm_output_directory, file_name)
            groundtruth_file_name = f"{os.path.splitext(file_name)[0]}-groundtruth.csv"
            groundtruth_path = os.path.join(
                groundtruth_directory,
                groundtruth_file_name
            )
            if os.path.exists(groundtruth_path):
                logging.info(f"Processing file: {file_name}")
                llm_output_df, groundtruth_df = preprocess_data(
                    llm_output_path,
                    groundtruth_path,
                    title_patterns
                )
                file_results = compute_jaro_winkler_metrics(
                    llm_output_df,
                    groundtruth_df
                )
                file_results["file_name"] = file_name
                results.append(file_results)
            else:
                logging.warning(f"Groundtruth file not found for {file_name}")

    logging.info("Concatenating results...")
    results_df = pd.concat(results, ignore_index=True)

    directory_suffix = os.path.basename(llm_output_directory).replace(" ", "_")
    results_directory = f"evaluation/data/output/{directory_suffix}"

    model_name = (
        results_df["model"].iloc[0]
        if not results_df["model"].empty
        else "unknown_model"
    )
    results_df_name = f"results_{directory_suffix}_{model_name}.csv"

    os.makedirs(results_directory, exist_ok=True)
    output_path = os.path.join(results_directory, results_df_name)
    logging.info(f"Saving results to: {output_path}")
    results_df.to_csv(output_path, index=False)
    print(results_df)


if __name__ == "__main__":
    input_path_transcripts = r"ocr/data/input/transcripts"
    input_path_reports = r"ocr/data/input/reports"
    output_path_transcripts = r"ocr/data/output/transcripts"
    output_path_reports = r"ocr/data/output/reports"
    
    # Ensure the LLM output directory is based on the model name and create it if it doesn't exist
    model = "claude-3-5-sonnet-20240620"  # adjustable parameter
    llm_output_directory = os.path.join(f"model/allContext/data/output/{model}")
    os.makedirs(llm_output_directory, exist_ok=True)

    groundtruth_directory = "evaluation/data/input"

    main(
        input_path_transcripts,
        input_path_reports,
        output_path_transcripts,
        output_path_reports,
        llm_output_directory,
        groundtruth_directory,
        model,
    )