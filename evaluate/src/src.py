import pandas as pd
import os
import re
import numpy as np

title_patterns = [
    r"\bLt\.\b", r"\bLes\.\b", r"\bDet\.\b", r"\bSat\.\b", r"\bCet\.\b",
    r"\bDetective\b", r"\bDetectives\b", r"\bOfficer\b", r"\bOfficers\b",
    r"\bSgt\.\b", r"\bLieutenant\b", r"\bSergeant\b", r"\bCaptain\b",
    r"\bCorporal\b", r"\bDeputy\b", r"\bOfc\.\b", r"\b\(?Technician\)?\b",
    r"\b\(?Criminalist\)?\b", r"\b\(?Photographer\)?\b", r"\bPolice Officer\b",
    r"\bCrime Lab Technician\b", r"\bLt\.\b", r"\bLes\.\b", r"\bDet\.\b",
    r"\bSat\.\b", r"\bCet\.\b", r"\bDetective\b", r"\bDetectives\b",
    r"\bOfficer\b", r"\bSgt\.\b", r"\bLieutenant\b", r"\bSergeant\b",
    r"\bCaptain\b", r"\bCorporal\b", r"\bDeputy\b", r"\bOfc\.\b",
    r"\b\(?Technician\)?\b", r"\b\(?Criminalist\)?\b", r"\b\(?Photographer\)?\b",
    r"\bPolice Officer\b", r"\bUNIT \d+\b", r"\bMOUNTED OFFICERS\b",
    r"\bCrime Lab Technician\b", r"Officer Context:", r"P/O", r"Police", r"^De",
    r"^L", r"^SG"
]

def split_names(row):
    split_patterns = [r" and ", r" ?, ?", r" ?: ?", r" ?& ?"]
    for pattern in split_patterns:
        if re.search(pattern, row):
            return re.split(pattern, row)
    return [row]

def extract_last_name(name):
    words = name.split()
    return words[-1] if words else ""

def adjust_last_name(name):
    split_name = re.split(r"(?<=[a-z])(?=[A-Z])", name)
    return split_name[-1] if split_name else name

def refine_adjusted_last_name(name):
    name = adjust_last_name(name)
    split_name = re.split(r"(?<=[a-z])(?=[A-Z])", name)
    return split_name[-1] if split_name else name

def levenshtein_distance(s1, s2):
    dp = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[len(s1)][len(s2)]

def levenshtein_ratio(s1, s2):
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - distance / max_len

def find_best_match(name, potential_matches):
    sorted_matches = sorted(potential_matches, key=lambda x: -x[1])
    top_matches = sorted_matches[:3]

    for match in top_matches:
        if match[0] == name:
            return match[0]
    return sorted_matches[0][0]

def split_consolidated_names(name):
    if isinstance(name, str):
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    else:
        return ""

def clean_llm_output_officer_names(df, title_patterns):
    for pattern in title_patterns:
        df["Officer Name"] = (
            df["Officer Name"]
            .str.replace(pattern, "", regex=True)
            .str.strip()
            .fillna("")
            .apply(split_consolidated_names)
            .str.lower()
        )
    return df

def clean_groundtruth_officer_names(df, title_patterns):
    df["Officer Names to Match"] = df["Officer Names to Match"].astype(str)  
    for pattern in title_patterns:
        df["Officer Names to Match"] = (
            df["Officer Names to Match"]
            .str.replace(pattern, "", regex=True)
            .str.strip()
            .fillna("")
            .apply(split_consolidated_names)
            .str.lower()
        )
    return df

def preprocess_data(llm_output_path, groundtruth_path, title_patterns):
    llm_output_df = pd.read_csv(llm_output_path)
    groundtruth_df = pd.read_csv(groundtruth_path)
    
    llm_output_df = clean_llm_output_officer_names(llm_output_df, title_patterns)
    groundtruth_df = clean_groundtruth_officer_names(groundtruth_df, title_patterns)

    llm_output_df["Split Officer Name"] = llm_output_df["Officer Name"].apply(split_names).str[0]
    llm_output_df["Officer Last Name"] = llm_output_df["Split Officer Name"].apply(extract_last_name)
    llm_output_df["Officer Refined Adjusted Last Name"] = llm_output_df["Officer Last Name"].apply(refine_adjusted_last_name)

    groundtruth_df["Split Officer Names to Match"] = groundtruth_df["Officer Names to Match"].apply(split_names).str[0]
    groundtruth_df["Officer Match Last Name"] = groundtruth_df["Split Officer Names to Match"].apply(extract_last_name)
    groundtruth_df["Officer Match Refined Adjusted Last Name"] = groundtruth_df["Officer Match Last Name"].apply(refine_adjusted_last_name)

    return llm_output_df, groundtruth_df


def compute_levenshtein_metrics(llm_output_df, groundtruth_df, levenshtein_threshold=0.5):
    results = []
    matched_names = set()
    
    llm_output_last_names = llm_output_df["Officer Refined Adjusted Last Name"].unique()
    groundtruth_last_names = groundtruth_df["Officer Match Refined Adjusted Last Name"].unique()
    
    unique_entity_count = len(groundtruth_df) 
    
    for name in llm_output_last_names:
        similarities = [
            (gt_name, levenshtein_ratio(name, gt_name))
            for gt_name in groundtruth_last_names
        ]
        best_match = find_best_match(name, similarities)
        
        max_similarity = max([similarity[1] for similarity in similarities])
        if max_similarity > levenshtein_threshold:
            matched_names.add(best_match)
    
    true_positives = len(matched_names.intersection(set(groundtruth_last_names.tolist())))
    false_positives = len(matched_names - set(groundtruth_last_names.tolist()))
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )
    recall = (
        true_positives / len(groundtruth_last_names)
        if len(groundtruth_last_names) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0
    )
    
    beta = 2
    f_beta_score = (
        (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        if (precision + recall) > 0
        else 0
    )
    
    token_count = llm_output_df["token_count"].iloc[0] if not llm_output_df["token_count"].empty else None
    filename = llm_output_df["fn"].iloc[0] if not llm_output_df["fn"].empty else None
    filetype = llm_output_df["file_type"].iloc[0] if not llm_output_df["file_type"].empty else None
    
    results.append(
        {
            "matched_count": len(matched_names),
            "total_ground_truth": len(groundtruth_last_names),
            "percentage_matched": len(matched_names) / len(groundtruth_last_names) * 100,
            "matched_names": matched_names,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "f_beta_score": f_beta_score,
            "token_count": token_count,
            "filename": filename,
            "filetype": filetype,
            "unique_entity_count": unique_entity_count 
        }
    )
    
    results_df = pd.DataFrame(results)
    return results_df


def main(llm_output_directory, groundtruth_directory):
    results = []
    for file_name in os.listdir(llm_output_directory):
        if file_name.endswith(".csv"):
            llm_output_path = os.path.join(llm_output_directory, file_name)
            groundtruth_file_name = f"{os.path.splitext(file_name)[0]}-groundtruth.csv"
            groundtruth_path = os.path.join(groundtruth_directory, groundtruth_file_name)

            if os.path.exists(groundtruth_path):
                llm_output_df, groundtruth_df = preprocess_data(llm_output_path, groundtruth_path, title_patterns)
                file_results = compute_levenshtein_metrics(llm_output_df, groundtruth_df)
                file_results["file_name"] = file_name
                results.append(file_results)
            else:
                print(f"Groundtruth file not found for {file_name}")

    results_df = pd.concat(results, ignore_index=True)
    
    # Extract the suffix from llm_output_directory
    directory_suffix = os.path.basename(llm_output_directory).replace(" ", "_")
    results_df_name = f"results_{directory_suffix}.csv"

    # Save the results_df with the dynamic name
    results_df.to_csv(f"../data/output/{results_df_name}", index=False)
    print(results_df)

if __name__ == "__main__":
    llm_output_directory = "../../model/data/ner-20-pages"
    groundtruth_directory = "../data/input"
    main(llm_output_directory, groundtruth_directory)