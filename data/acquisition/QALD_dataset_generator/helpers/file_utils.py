import requests
import xml.etree.ElementTree as ET
import json
import pandas as pd


def parse_xml_to_dataframe(xml_content):
    root = ET.fromstring(xml_content)
    data = []
    for question in root.findall("question"):
        sparql_query = (
            question.find("query").text
            if question.find("query") is not None
            else "<SPARQL_QUERY_NOT_AVAILABLE>"
        )
        for string_element in question.findall("string"):
            language = string_element.attrib.get("lang", "en")
            if check_language(language):
                if string_element.text is None:
                    continue
                text_query = string_element.text.strip()
                data.append(
                    {
                        "text_query": text_query,
                        "language": language,
                        "sparql_query": sparql_query,
                    }
                )
    return pd.DataFrame(data)


def check_language(language):
    # return language in ["en", "de"]
    return True


def parse_json_to_dataframe(json_content):
    data = json.loads(json_content)
    rows = []

    for question in data.get("questions", []):
        sparql_query = question.get("query", "<SPARQL_QUERY_NOT_AVAILABLE>")
        for body in question.get("body", []):
            language = body.get("language", "en")
            if check_language(language):  # Filter for English and German entries
                text_query = body.get("string", "").strip()
                rows.append(
                    {
                        "text_query": text_query,
                        "language": language,
                        "sparql_query": sparql_query,
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/ag-sc/QALD/master/5/data/qald-5_train_raw.xml"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            df = parse_xml_to_dataframe(response.text)
            print(df.iloc[0]["sparql_query"])
        except Exception as e:
            print(f"Failed to parse XML file with url {url}. Error: {e}")
    else:
        print(f"Failed to fetch XML file. Status code: {response.status_code}")
