import os
import logging
import pandas as pd
from model.ner.src import process_query  # adjustable parameter
# from model.allPages.src import process_query  # adjustable parameter
# from model.allContext.src import process_query
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

def main(
    input_path_transcripts,
    input_path_reports,
    output_path_transcripts,
    output_path_reports,
    llm_output_directory,
    groundtruth_directory,
    model,
):
    # Process the PDF files using OCR
    ocr_process(input_path_transcripts, input_path_reports, output_path_transcripts, output_path_reports)

    # Process the OCR output files and generate LLM output
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
    llm_output_directory = r"model/data/ner-20-pages"  # adjustable parameter
    groundtruth_directory = "evaluation/data/input"
    model = "claude-3-sonnet-20240229"  # adjustable parameter

    main(
        input_path_transcripts,
        input_path_reports,
        output_path_transcripts,
        output_path_reports,
        llm_output_directory,
        groundtruth_directory,
        model,
    )