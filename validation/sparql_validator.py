import json
import requests
import argparse
import concurrent.futures
import matplotlib.pyplot as plt
import logging
import time

# import requests_cache
from typing import Dict, Any, Tuple, Set, Optional

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "SPARQLValidatorBot/1.0 (mailto:your_email@example.com)"}
MAX_WORKERS = 5
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SPARQLEvaluator")

questions_to_ignore = [
    "How many international airports are located within the city of Hamburg ?",
    "How many paintings of Pablo Picasso were ever in a museum?",
    "What event killed the most people in the years 1910 to 1920?",
]


def retry_request(
    url: str, params: Dict[str, Any], headers: Dict[str, str], timeout: int
) -> Optional[requests.Response]:
    """Retry only on rate limit or server errors."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=timeout
            )
            if response.status_code == 200:
                return response
            elif response.status_code == 429 or 500 <= response.status_code < 600:
                logger.warning(
                    f"Retrying due to status {response.status_code} (Attempt {attempt + 1})"
                )
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None
        except requests.RequestException as e:
            logger.warning(f"Request exception: {e} (Attempt {attempt + 1})")
            time.sleep(RETRY_DELAY * (attempt + 1))
    logger.error("All retry attempts failed.")
    logger.error(f"SPARQL query: {params['query']}")
    return None


def validate_sparql(query: str) -> bool:
    """Send the query to check if it returns a valid response (syntax-level validation)."""
    response = retry_request(
        SPARQL_ENDPOINT, {"query": query, "format": "json"}, HEADERS, timeout=30
    )
    return response is not None


def safe_add_limit(query: str, limit: int = 100) -> str:
    query = query.rstrip(" ;") if query.endswith(";") else query
    upper_query = query.upper()

    if "ASK" in upper_query or "INSERT" in upper_query or "DELETE" in upper_query:
        return query

    if "LIMIT" not in upper_query:
        query = f"{query} LIMIT {limit}"
    return query + ";"


def execute_sparql(query: str) -> Optional[Dict[str, Any]]:
    """Executes a SPARQL query and returns the JSON result."""
    response = retry_request(
        SPARQL_ENDPOINT, {"query": query, "format": "json"}, HEADERS, timeout=30
    )
    if response:
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
    return None


def extract_results(json_response: Dict[str, Any]) -> Set:
    """Extracts comparable result set from SPARQL JSON response."""
    if not json_response:
        return set()
    if "boolean" in json_response:
        return {json_response["boolean"]}
    if "results" in json_response and "bindings" in json_response["results"]:
        return {
            tuple(binding[var]["value"] for var in binding)
            for binding in json_response["results"]["bindings"]
        }
    return set()


def process_entry(entry: Dict[str, Any]) -> Tuple[int, int, int]:
    """Evaluates a single pair of gold and generated queries."""
    generated_query = entry.get("generated_sparql", "")
    gold_query = entry.get("gold_sparql", "")

    if not generated_query or not validate_sparql(generated_query):
        return (0, 0, 0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_gold = executor.submit(execute_sparql, gold_query)
        future_gen = executor.submit(execute_sparql, safe_add_limit(generated_query))
        gold_result = extract_results(future_gen.result())
        gen_result = extract_results(future_gold.result())

    exact_match = gold_result == gen_result
    partial_match = not exact_match and bool(gold_result & gen_result)
    return (1, int(exact_match), int(partial_match))


def evaluate_queries(file_path: str) -> Tuple[int, int, int, int]:
    """Evaluates all queries in the JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = [entry for entry in data if entry["question"] not in questions_to_ignore]
    total = len(data)
    valid = correct = partial = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_entry, entry) for entry in data]
        for future in concurrent.futures.as_completed(futures):
            v, c, p = future.result()
            valid += v
            correct += c
            partial += p

    return total, valid, correct, partial


def plot_results(total: int, valid: int, correct: int, partial: int):
    """Plot evaluation result summary."""
    labels = ["Total Queries", "Valid Queries", "Correct Answers", "Partial Answers"]
    values = [total, valid, correct, partial]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=["gray", "blue", "green", "orange"])
    plt.xlabel("Categories")
    plt.ylabel("Number of Queries")
    plt.title("SPARQL Query Evaluation Results")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SPARQL query generation.")
    parser.add_argument("file_path", help="Path to the JSON file containing queries.")
    args = parser.parse_args()

    total, valid, correct, partial = evaluate_queries(args.file_path)
    print(f"Total Queries: {total}")
    print(f"Valid Queries: {valid}")
    print(f"Correct Answers: {correct}")
    print(f"Partial Answers: {partial}")

    plot_results(total, valid, correct, partial)
