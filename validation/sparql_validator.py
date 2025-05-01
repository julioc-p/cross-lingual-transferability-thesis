import json
import requests
import argparse
import concurrent.futures
import logging
import time
import requests_cache
from typing import Dict, Any, Tuple, Set, Optional, List
from tqdm import tqdm
import math


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "SPARQLValidatorBot/1.0 (mailto:your_real_email@example.com)"}
MAX_WORKERS = 5
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5
CACHE_NAME = "sparql_cache"
CACHE_BACKEND = "sqlite"
CACHE_EXPIRE_AFTER = 3600 * 24 * 7
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("SPARQLEvaluator")
requests_cache.install_cache(
    CACHE_NAME,
    backend=CACHE_BACKEND,
    expire_after=CACHE_EXPIRE_AFTER,
    allowable_methods=["GET", "POST"],
    allowable_codes=[200],
)
logger.info(f"Requests caching enabled. Cache file: {CACHE_NAME}.sqlite")

questions_to_ignore = [
    "How many international airports are located within the city of Hamburg ?",
    "How many paintings of Pablo Picasso were ever in a museum?",
    "What event killed the most people in the years 1910 to 1920?",
    "Wieviele internationale Flughäfen gibt es in der Stadt Hamburg?",
    "Wieviele Gemälde von Pablo Picasso waren jemals in einem Museum?",
    "Welches Ereignis zwischen 1910 und 1920 tötete die meisten Menschen?",
]


def retry_request(
    url: str, params: Dict[str, Any], headers: Dict[str, str], timeout: int
) -> Optional[requests.Response]:
    """
    Sends a request to the SPARQL endpoint with retry logic for specific errors.
    Leverages requests_cache if the request is identical to a previous one.
    """
    session = (
        requests_cache.CachedSession()
        if requests_cache.is_installed()
        else requests.Session()
    )
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            was_cached = getattr(response, "from_cache", False)
            if response.status_code == 200:
                return response
            elif response.status_code == 429 or 500 <= response.status_code < 600:
                wait_time = RETRY_DELAY * (2**attempt)
                logger.warning(
                    f"Status {response.status_code}. Retrying attempt {attempt + 1}/{RETRY_ATTEMPTS} "
                    f"after {wait_time:.2f} seconds. URL: {response.url}"
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Non-retriable status {response.status_code} received. URL: {response.url}"
                )
                logger.error(f"Query: {params.get('query', 'N/A')}")
                return None
        except requests.RequestException as e:
            wait_time = RETRY_DELAY * (2**attempt)
            logger.warning(
                f"Request exception: {e}. Retrying attempt {attempt + 1}/{RETRY_ATTEMPTS} after {wait_time:.2f} seconds."
            )
            time.sleep(wait_time)
    logger.error(
        f"All {RETRY_ATTEMPTS} retry attempts failed for query. SPARQL query: {params.get('query', 'N/A')}"
    )
    return None


def validate_sparql(query: str) -> bool:
    """
    Checks if a SPARQL query is syntactically valid by sending a request.
    Uses ASK query pattern for minimal response size (if possible) or applies a LIMIT 1.
    Returns True if valid syntax (receives 200 OK), False otherwise.
    """
    if not query or not query.strip():
        logger.warning("Attempted to validate an empty query.")
        return False

    validation_query = query
    upper_query = query.strip().upper()

    if upper_query.startswith("SELECT"):
        if "LIMIT" not in upper_query and "OFFSET" not in upper_query:
            validation_query = f"{query.rstrip(' ;')} LIMIT 1"
    elif upper_query.startswith("ASK"):
        pass

    response = retry_request(
        SPARQL_ENDPOINT,
        {"query": validation_query, "format": "json"},
        HEADERS,
        timeout=30,
    )

    if response is None:
        logger.warning(
            f"Validation failed for query (request/retry error). Original query snippet: {query[:100]}..."
        )
        return False
    elif response.status_code != 200:
        logger.warning(
            f"Validation failed for query with status code {response.status_code}. Query snippet: {query[:100]}..."
        )
        return False

    return True


def safe_add_limit(query: str, limit: int = 1000) -> str:
    """
    Adds a LIMIT clause to SELECT queries if one doesn't exist or is larger.
    Does not modify ASK, INSERT, DELETE queries.
    Tries to avoid adding LIMIT if OFFSET is present without a LIMIT.
    """
    query = query.strip()
    if not query:
        return ""
    upper_query = query.upper()

    if upper_query.startswith("SELECT"):
        if "LIMIT" not in upper_query:
            if "OFFSET" not in upper_query:
                if query.endswith(";"):
                    query = query[:-1].rstrip()
                return f"{query} LIMIT {limit}"
            else:
                logger.debug(
                    f"Skipping LIMIT addition due to existing OFFSET without LIMIT: {query[:100]}..."
                )
                return query
    return query


def execute_sparql(query: str) -> Optional[Dict[str, Any]]:
    """
    Executes a SPARQL query and returns the JSON result.
    Relies on retry_request for network handling and caching.
    """
    if not query:
        logger.warning("Attempted to execute an empty query.")
        return None

    response = retry_request(
        SPARQL_ENDPOINT,
        {"query": query, "format": "json"},
        HEADERS,
        timeout=60,
    )

    if response and response.status_code == 200:
        try:
            if not response.text:
                logger.warning(
                    f"Received empty response body for query: {query[:100]}..."
                )
                if query.strip().upper().startswith("ASK"):
                    return {"head": {}, "boolean": False}
                else:
                    return {"head": {"vars": []}, "results": {"bindings": []}}
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for query: {query[:100]}... Error: {e}")
            logger.error(f"Response text: {response.text[:500]}")
            return None
    else:
        logger.error(
            f"Failed to execute SPARQL query (post-retry or non-200). Query: {query[:100]}..."
        )
        return None


def extract_results(json_response: Optional[Dict[str, Any]]) -> Set:
    """
    Extracts a comparable set of results from SPARQL JSON response.
    Handles None input gracefully. Handles ASK results.
    """
    if json_response is None:
        return set()

    try:
        if "boolean" in json_response:
            return {str(json_response["boolean"])}

        if "results" in json_response and "bindings" in json_response["results"]:
            results = set()
            vars_list = json_response.get("head", {}).get("vars", [])
            if not vars_list:
                if not json_response["results"]["bindings"]:
                    return set()
                else:
                    logger.warning(
                        "Query has no variables in head, but bindings exist?"
                    )
                    return set()

            for binding in json_response["results"]["bindings"]:
                result_tuple = []
                for var in vars_list:
                    value_info = binding.get(var)
                    value = value_info.get("value") if value_info else None
                    result_tuple.append(value)
                results.add(tuple(result_tuple))
            return results
        else:
            logger.warning(
                f"Unexpected JSON structure for result extraction: {str(json_response)[:200]}"
            )
            return set()

    except Exception as e:
        logger.error(
            f"Error extracting results from JSON: {e}. Response snippet: {str(json_response)[:500]}"
        )
        return set()


def calculate_metrics_for_entry(
    entry: Dict[str, Any],
) -> Tuple[float, float, float, bool]:
    """
    Calculates Precision, Recall, and F1 score for a single entry based on GERBIL QA rules (QALD variant).

    Returns:
        Tuple[float, float, float, bool]: (precision, recall, f1, is_valid_syntax)
        Returns (0.0, 0.0, 0.0, False) if generated query is empty or syntax is invalid.
        Returns (0.0, 0.0, 0.0, True) if syntax is valid but query execution fails.
    """
    question = entry.get("question", "N/A")
    generated_query = entry.get("generated_sparql", "").strip()
    gold_query = entry.get("gold_sparql", "").strip()
    precision, recall, f1 = 0.0, 0.0, 0.0

    if not generated_query:
        logger.warning(
            f"Skipping entry due to empty generated query. Question: {question}"
        )
        return (0.0, 0.0, 0.0, False)

    is_valid_syntax = validate_sparql(generated_query)
    if not is_valid_syntax:
        logger.warning(
            f"Generated query failed syntax validation. Question: {question}"
        )
        return (0.0, 0.0, 0.0, False)

    limited_generated_query = safe_add_limit(generated_query)

    gold_json = execute_sparql(gold_query)
    if gold_json is None:
        logger.error(
            f"Failed to execute or parse GOLD query. Cannot compare. Assigning 0 scores. Question: {question}"
        )
        return (0.0, 0.0, 0.0, True)

    gen_json = execute_sparql(limited_generated_query)
    if gen_json is None:
        logger.error(
            f"Failed to execute or parse GENERATED query (post-validation). Assigning 0 scores. Question: {question}"
        )
        return (0.0, 0.0, 0.0, True)

    gold_results = extract_results(gold_json)
    gen_results = extract_results(gen_json)

    is_gold_empty = not gold_results
    is_gen_empty = not gen_results

    if is_gold_empty and is_gen_empty:
        precision, recall, f1 = 1.0, 1.0, 1.0
    elif is_gold_empty and not is_gen_empty:
        precision, recall, f1 = 0.0, 0.0, 0.0
    elif not is_gold_empty and is_gen_empty:
        precision, recall, f1 = 1.0, 0.0, 0.0
    else:
        tp = len(gold_results.intersection(gen_results))
        fp = len(gen_results.difference(gold_results))
        fn = len(gold_results.difference(gen_results))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    precision = 0.0 if math.isnan(precision) else precision
    recall = 0.0 if math.isnan(recall) else recall
    f1 = 0.0 if math.isnan(f1) else f1

    return (precision, recall, f1, True)


def evaluate_queries(file_path: str) -> Tuple[int, int, float, float, float]:
    """
    Loads data, filters ignored questions/empty queries, evaluates entries using a thread pool,
    and calculates Macro Precision, Recall, and F1 score over ALL attempted entries.

    Returns:
        Tuple: (total_evaluated_count, valid_syntax_count, macro_precision, macro_recall, macro_f1)
               valid_syntax_count is the number of queries passing the syntax check.
               Macro averages are calculated over total_evaluated_count.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        return (0, 0, 0.0, 0.0, 0.0)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file {file_path}: {e}")
        return (0, 0, 0.0, 0.0, 0.0)

    original_count = len(data)
    filtered_data = [
        entry for entry in data if entry.get("question") not in questions_to_ignore
    ]
    total_evaluated = len(filtered_data)
    logger.info(
        f"Loaded {original_count} entries, evaluating {total_evaluated} after filtering ignored questions and empty generated queries."
    )

    if not filtered_data:
        logger.warning("No entries left to evaluate after filtering.")
        return (0, 0, 0.0, 0.0, 0.0)

    valid_syntax_count = 0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(calculate_metrics_for_entry, entry): entry
            for entry in filtered_data
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=total_evaluated,
            desc="Evaluating Queries",
        ):
            entry = futures[future]
            question = entry.get("question", "N/A")
            try:
                p, r, f1, is_valid = future.result()

                if is_valid:
                    valid_syntax_count += 1

                total_precision += p
                total_recall += r
                total_f1 += f1

            except Exception as exc:
                logger.error(
                    f"Entry for question '{question}' generated an exception during future processing: {exc}",
                    exc_info=True,
                )

    if total_evaluated > 0:
        macro_precision = total_precision / total_evaluated
        macro_recall = total_recall / total_evaluated
        macro_f1 = total_f1 / total_evaluated
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    return total_evaluated, valid_syntax_count, macro_precision, macro_recall, macro_f1


def report_results(
    total: int, valid_syntax: int, macro_p: float, macro_r: float, macro_f1: float
):
    """Prints the evaluation results summary."""
    if total == 0:
        print("\nNo queries evaluated.")
        return

    print("\n--- Evaluation Summary ---")
    print(f"Total Entries Attempted (after filter): {total}")
    print(f"Entries with Valid SPARQL Syntax: {valid_syntax}")
    if total > 0:
        valid_perc = (valid_syntax / total) * 100
        print(f"  ({valid_perc:.2f}% of attempted passed syntax validation)")

    print("\n--- Macro Averaged Metrics (QALD Variant) ---")
    print(f"(Calculated over all {total} attempted entries)")
    print("(Entries failing syntax check or execution contribute 0.0 to scores)")
    print(f"Macro Precision: {macro_p:.8f}")
    print(f"Macro Recall:    {macro_r:.8f}")
    print(f"Macro F1-Score:  {macro_f1:.8f}")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate and evaluate generated SPARQL queries against gold standards using Wikidata, calculating Macro P/R/F1 over all entries."
    )
    parser.add_argument(
        "file_path", help="Path to the JSON file containing query pairs."
    )
    args = parser.parse_args()

    print(f"Starting evaluation for file: {args.file_path}")
    print(f"Using up to {MAX_WORKERS} concurrent workers.")
    print(f"Caching enabled: {requests_cache.is_installed()}")

    start_time = time.time()
    total, valid_syntax, macro_p, macro_r, macro_f1 = evaluate_queries(args.file_path)
    end_time = time.time()

    print(f"\nEvaluation finished in {end_time - start_time:.2f} seconds.")

    report_results(total, valid_syntax, macro_p, macro_r, macro_f1)
