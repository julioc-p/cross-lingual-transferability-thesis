from helpers import (
    parse_xml_to_dataframe,
    parse_json_to_dataframe,
    eliminate_invalid_sparql_queries,
    load_to_hugging_face,
)
import requests
import pandas as pd
import json

SOURCE_FILE = "sources/qald_urls.json"
HUGGING_FACE_REPO = "julioc-p/Question-Sparql"
SOURCE_URLS_FILE = "qald_challenges.csv"

def fetch_and_parse_data(urls, parse_function):
    data = []
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            try:
                data.append(parse_function(response.text))
            except Exception as e:
                print(f"Failed to parse file with url {url}. Error: {e}")
        else:
            print(f"Failed to fetch file. Status code: {response.status_code}")
    return data

def main():
    with open(SOURCE_FILE, "r") as json_file:
        sources = json.load(json_file)
        xml_sources = sources["xml"]
        json_sources = sources["json"]

    all_data = fetch_and_parse_data(xml_sources, parse_xml_to_dataframe)
    all_data.extend(fetch_and_parse_data(json_sources, parse_json_to_dataframe))

    all_data_df = pd.concat(all_data)
    all_data_df.drop_duplicates(subset=["text_query", "language", "sparql_query"], inplace=True)
    all_data_df = all_data_df[all_data_df.sparql_query != "<SPARQL_QUERY_NOT_AVAILABLE>"]
    all_data_df.to_csv(SOURCE_URLS_FILE, index=False)
    load_to_hugging_face(SOURCE_URLS_FILE, HUGGING_FACE_REPO)

if __name__ == "__main__":
    main()
