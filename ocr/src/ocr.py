import os
import json
import pdf2image
import azure
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from io import BytesIO
import time
import logging


def getcreds():
    with open("ocr/creds/creds_cv.txt", "r") as c:
        creds = c.readlines()
    return creds[0].strip(), creds[1].strip()


def extract_content(result):
    contents = {}
    for read_result in result.analyze_result.read_results:
        lines = read_result.lines
        lines.sort(key=lambda line: line.bounding_box[1])

        page_content = []
        for line in lines:
            page_content.append(" ".join([word.text for word in line.words]))

        contents[f"page_{read_result.page}"] = "\n".join(page_content)

    return contents


def pdf2df(pdf_path, json_file, client):
    with open(pdf_path, "rb") as file:
        pdf_data = file.read()

        num_pages = pdf2image.pdfinfo_from_bytes(pdf_data)["Pages"]

        for i in range(num_pages):
            try:
                image = pdf2image.convert_from_bytes(
                    pdf_data, dpi=500, first_page=i + 1, last_page=i + 1
                )[0]

                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format="PNG")

                img_byte_arr.seek(0)
                ocr_result = client.read_in_stream(img_byte_arr, raw=True)
                operation_id = ocr_result.headers["Operation-Location"].split("/")[-1]

                while True:
                    result = client.get_read_result(operation_id)
                    if result.status.lower() not in ["notstarted", "running"]:
                        break
                    time.sleep(1)

                if result.status.lower() == "failed":
                    logging.error(f"OCR failed for page {i+1} of file {pdf_path}")
                    continue

                page_results = extract_content(result)

                with open(json_file, "a") as f:
                    json.dump(page_results, f)
                    f.write("\n")

            except azure.core.exceptions.HttpResponseError as e:
                logging.error(f"Error processing page {i+1} of file {pdf_path}: {e}")
                continue


def process(pdf_path, output_path):
    outname = os.path.basename(pdf_path).replace(".pdf", "")
    outstring = os.path.join(output_path, "{}.json".format(outname))
    outpath = os.path.abspath(outstring)

    if os.path.exists(outpath):
        logging.info(f"skipping {outpath}, file already exists")
        return outpath

    logging.info(f"sending document {outname}")

    with open(outpath, "w") as f:
        f.write('{ "messages": {\n')

    endpoint, key = getcreds()
    client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    pdf2df(pdf_path, outpath, client)

    with open(outpath, "a") as f:
        f.write("\n}}")

    logging.info(f"finished writing to {outpath}")
    return outpath


def update_page_keys_in_json(json_file):
    corrected_messages = {}

    with open(json_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines, start=0):
            if i == 0:
                continue
            try:
                content = json.loads(line.strip())
                corrected_key = f"page_{i}"
                corrected_messages[corrected_key] = content[f"page_1"]
            except json.JSONDecodeError:
                continue

    with open(json_file, "w") as f:
        json.dump({"messages": corrected_messages}, f, indent=4)


def reformat_json_structure(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    new_messages = []
    for key, content in data["messages"].items():
        page_num = int(key.split("_")[1])
        new_messages.append({"page_content": content, "page_number": page_num})

    new_data = {"messages": new_messages}

    with open(json_file, "w") as f:
        json.dump(new_data, f, indent=4)


def ocr_process(
    input_path_transcripts,
    input_path_reports,
    output_path_transcripts,
    output_path_reports,
):
    logger = logging.getLogger()
    azurelogger = logging.getLogger("azure")
    logger.setLevel(logging.INFO)
    azurelogger.setLevel(logging.ERROR)

    if not os.path.exists(output_path_transcripts):
        os.makedirs(output_path_transcripts)

    if not os.path.exists(output_path_reports):
        os.makedirs(output_path_reports)

    files_transcripts = [
        f
        for f in os.listdir(input_path_transcripts)
        if os.path.isfile(os.path.join(input_path_transcripts, f))
        and f.lower().endswith(".pdf")
    ]
    logging.info(f"starting to process {len(files_transcripts)} transcript files")
    for file in files_transcripts:
        json_file_path = process(
            os.path.join(input_path_transcripts, file), output_path_transcripts
        )
        update_page_keys_in_json(json_file_path)
        reformat_json_structure(json_file_path)

    files_reports = [
        f
        for f in os.listdir(input_path_reports)
        if os.path.isfile(os.path.join(input_path_reports, f))
        and f.lower().endswith(".pdf")
    ]
    logging.info(f"starting to process {len(files_reports)} report files")
    for file in files_reports:
        json_file_path = process(
            os.path.join(input_path_reports, file), output_path_reports
        )
        update_page_keys_in_json(json_file_path)
        reformat_json_structure(json_file_path)
