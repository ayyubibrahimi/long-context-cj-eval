import os
import logging
from evaluation import preprocess_data, compute_jaro_winkler_metrics, title_patterns
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(llm_output_directory, groundtruth_directory):
    logging.info("Starting main function...")
    results = []
    for file_name in os.listdir(llm_output_directory):
        if file_name.endswith(".csv"):
            llm_output_path = os.path.join(llm_output_directory, file_name)
            groundtruth_file_name = f"{os.path.splitext(file_name)[0]}-groundtruth.csv"
            groundtruth_path = os.path.join(
                groundtruth_directory, groundtruth_file_name
            )

            if os.path.exists(groundtruth_path):
                logging.info(f"Processing file: {file_name}")
                llm_output_df, groundtruth_df = preprocess_data(
                    llm_output_path, groundtruth_path, title_patterns
                )
                file_results = compute_jaro_winkler_metrics(
                    llm_output_df, groundtruth_df
                )
                file_results["file_name"] = file_name
                results.append(file_results)
            else:
                logging.warning(f"Groundtruth file not found for {file_name}")

    logging.info("Concatenating results...")
    results_df = pd.concat(results, ignore_index=True)

    model_name = (
        results_df["model"].iloc[0]
        if not results_df["model"].empty
        else "unknown_model"
    )

    # Create the results filename with directory suffix and model name

    # Extract the suffix from llm_output_directory
    directory_suffix = os.path.basename(llm_output_directory).replace(" ", "_")
    results_directory = f"../data/output/{directory_suffix}"
    results_df_name = f"results_{directory_suffix}_{model_name}.csv"

    # Create the directory if it doesn't exist
    os.makedirs(results_directory, exist_ok=True)

    # Save the results_df in the newly created directory
    output_path = os.path.join(results_directory, results_df_name)
    logging.info(f"Saving results to: {output_path}")
    results_df.to_csv(output_path, index=False)
    print(results_df)


if __name__ == "__main__":
    llm_output_directory = "../../model/data/ner-20-pages"
    groundtruth_directory = "../data/input"
    main(llm_output_directory, groundtruth_directory)
