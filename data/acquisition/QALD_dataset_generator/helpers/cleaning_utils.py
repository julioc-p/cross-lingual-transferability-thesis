from .sparql_validation_utils import validate_sparql_query_result
from SPARQLWrapper import SPARQLWrapper, JSON
from ratelimit import limits, sleep_and_retry
import requests, time, pandas as pd
from tqdm import tqdm
import requests_cache


sparql = SPARQLWrapper("https://query.wikidata.org/")


def eliminate_invalid_sparql_queries(dataframe):
    valid_entries, invalid_entries = validate_queries(dataframe=dataframe)
    valid_entries.to_csv("valid_queries.csv", index=False)
    invalid_entries.to_csv("invalid_queries.csv", index=False)
    return valid_entries


def validate_queries(dataframe):
    valid_queries = []
    invalid_queries = []
    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
        query = row["sparql_query"]
        result = send_query_to_sparql_endpoint(query)
        if result and validate_sparql_query_result(result):
            valid_queries.append(row)
        else:
            invalid_queries.append(row)
    return pd.DataFrame(valid_queries), pd.DataFrame(invalid_queries)


@sleep_and_retry
@limits(calls=100, period=1)
def send_query_to_sparql_endpoint(query):
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        response = sparql.queryAndConvert()
    except Exception as e:
        print(f"Failed to send query to SPARQL endpoint. Error: {e}")
        return None
    return response


if __name__ == "__main__":

    print(send_query_to_sparql_endpoint("SELECT * WHERE { ?s ?p ?o } LIMIT 10"))
