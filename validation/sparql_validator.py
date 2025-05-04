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
from collections import Counter

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

CATEGORY_EMPTY_QUERY_GENERATED = "Failure: Empty Generated Query String"
CATEGORY_EXECUTION_FAILURE_GENERATED = "Failure: Generated Query Execution"
CATEGORY_EXECUTION_FAILURE_GOLD = "Failure: Gold Query Execution (Generated OK)"
CATEGORY_BOTH_EMPTY = "Correct (Both Empty)"
CATEGORY_GEN_EMPTY_GOLD_NOT = "Incorrect (Gen Empty, Gold Not)"
CATEGORY_GOLD_EMPTY_GEN_NOT = "Incorrect (Gold Empty, Gen Not)"
CATEGORY_EXACT_MATCH = "Correct (Exact Match)"
CATEGORY_PARTIAL_MATCH = "Correct (Partial Match)"
CATEGORY_NO_OVERLAP = "Incorrect (No Overlap)"
CATEGORY_PROCESSING_ERROR = "Failure: Internal Processing Error"




def retry_request(
    url: str, params: Dict[str, Any], headers: Dict[str, str], timeout: int
) -> Optional[requests.Response]:
    """Sends request with retry logic, uses cache."""
    session = (
        requests_cache.CachedSession()
        if requests_cache.is_installed()
        else requests.Session()
    )
    query_snippet = (
        str(params.get("query", "N/A"))[:150] + "..." if params.get("query") else "N/A"
    )
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response
            elif response.status_code == 429 or 500 <= response.status_code < 600:
                wait_time = RETRY_DELAY * (2**attempt)
                logger.warning(
                    f"Status {response.status_code}. Retrying attempt {attempt + 1}/{RETRY_ATTEMPTS} after {wait_time:.2f}s. URL: {response.url}"
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Non-retriable status {response.status_code}. URL: {response.url}"
                )
                return None
        except requests.exceptions.Timeout as e:
            wait_time = RETRY_DELAY * (2**attempt)
            logger.warning(
                f"Request timed out: {e}. Retrying attempt {attempt + 1}/{RETRY_ATTEMPTS} after {wait_time:.2f}s. Query: {query_snippet}"
            )
            time.sleep(wait_time)
        except requests.RequestException as e:
            wait_time = RETRY_DELAY * (2**attempt)
            logger.warning(
                f"Request exception: {e}. Retrying attempt {attempt + 1}/{RETRY_ATTEMPTS} after {wait_time:.2f}s. Query: {query_snippet}"
            )
            time.sleep(wait_time)
    logger.error(f"All {RETRY_ATTEMPTS} attempts failed for query: {query_snippet}")
    return None


def safe_add_limit(query: str, limit: int = 1000) -> str:
    """Adds LIMIT to SELECT queries safely."""
    query = query.strip()
    if not query:
        return ""
    upper_query = query.upper()
    if upper_query.startswith("SELECT") and not upper_query.endswith(f"LIMIT {limit}"):
        if (
            "LIMIT" not in upper_query.split("#")[0].split("\n")[-1]
            and "OFFSET" not in upper_query.split("#")[0].split("\n")[-1]
        ):
            if query.endswith(";"):
                query = query[:-1].rstrip()
            parts = query.split("#", 1)
            main_query = parts[0].rstrip()
            comment = f" #{parts[1]}" if len(parts) > 1 else ""
            return f"{main_query} LIMIT {limit}{comment}"
    return query


def execute_sparql(query: str) -> Optional[Dict[str, Any]]:
    """Executes SPARQL query, returns JSON or None on failure."""
    if not query or not query.strip():
        logger.warning("execute_sparql called with empty or whitespace query.")
        return None

    response = retry_request(
        SPARQL_ENDPOINT, {"query": query, "format": "json"}, HEADERS, timeout=60
    )
    if response is None:
        return None
    if response.status_code == 200:
        try:
            if not response.text:
                logger.warning(
                    f"Received empty response body (treated as no results). Query: {query[:100]}..."
                )
                if query.strip().upper().startswith("ASK"):
                    return {"head": {}, "boolean": False}
                else:
                    return {"head": {"vars": []}, "results": {"bindings": []}}
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(
                f"JSON parse error: {e}. Query: {query[:100]}... Response: {response.text[:500]}"
            )
            return None
    else:
        logger.error(
            f"Unexpected state: execute_sparql received non-None response with status {response.status_code}. Query: {query[:100]}..."
        )
        return None


def extract_results(json_response: Optional[Dict[str, Any]]) -> Set:
    """Extracts comparable set of results from SPARQL JSON."""
    if json_response is None:
        logger.error("INTERNAL: extract_results called with None json_response.")
        return set()
    try:
        if "boolean" in json_response:
            return {str(json_response["boolean"])}
        if "results" in json_response and "bindings" in json_response["results"]:
            results = set()
            vars_list = json_response.get("head", {}).get("vars", [])
            if not vars_list:
                if json_response["results"]["bindings"]:
                    logger.warning(
                        f"Query has no head vars but bindings exist - result extraction unclear. Bindings: {json_response['results']['bindings']}"
                    )
                    return set()
                else:
                    return set()
            for binding in json_response["results"]["bindings"]:
                result_tuple = tuple(
                    binding.get(var, {}).get("value") for var in vars_list
                )
                results.add(result_tuple)
            return results
        else:
            logger.warning(
                f"Unexpected JSON structure for result extraction: {str(json_response)[:200]}"
            )
            return set()
    except Exception as e:
        logger.error(
            f"Error extracting results: {e}. Response: {str(json_response)[:500]}"
        )
        return set()




def calculate_metrics_for_entry(
    entry: Dict[str, Any],
) -> Tuple[float, float, float, str]:
    """
    Calculates P, R, F1 and determines the result category for a single entry.
    Handles empty generated queries first.

    Returns:
        Tuple[float, float, float, str]: (precision, recall, f1, category_string)
    """
    question = entry.get("question", "N/A")
    generated_query = entry.get("generated_sparql", "").strip()
    gold_query = entry.get("gold_sparql", "").strip()
    precision, recall, f1 = 0.0, 0.0, 0.0

    if not generated_query:
        return (0.0, 0.0, 0.0, CATEGORY_EMPTY_QUERY_GENERATED)


    limited_generated_query = safe_add_limit(generated_query)
    gen_json = execute_sparql(limited_generated_query)

    if gen_json is None:
        return (0.0, 0.0, 0.0, CATEGORY_EXECUTION_FAILURE_GENERATED)


    gold_json = execute_sparql(gold_query)
    if gold_json is None:
        logger.error(
            f"GOLD query execution failed (Generated query was OK). Q: {question}"
        )
        return (0.0, 0.0, 0.0, CATEGORY_EXECUTION_FAILURE_GOLD)


    gold_results = extract_results(gold_json)
    gen_results = extract_results(gen_json)

    is_gold_empty = not gold_results
    is_gen_empty = not gen_results
    category = CATEGORY_PARTIAL_MATCH

    if is_gold_empty and is_gen_empty:
        precision, recall, f1 = 1.0, 1.0, 1.0
        category = CATEGORY_BOTH_EMPTY
    elif is_gold_empty and not is_gen_empty:
        precision, recall, f1 = 0.0, 0.0, 0.0
        category = CATEGORY_GOLD_EMPTY_GEN_NOT
    elif not is_gold_empty and is_gen_empty:
        precision, recall, f1 = 1.0, 0.0, 0.0
        category = CATEGORY_GEN_EMPTY_GOLD_NOT
    else:
        tp = len(gold_results.intersection(gen_results))
        fp = len(gen_results.difference(gold_results))
        fn = len(gold_results.difference(gen_results))
        if tp == 0:
            precision, recall, f1 = 0.0, 0.0, 0.0
            category = CATEGORY_NO_OVERLAP
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            if math.isclose(f1, 1.0):
                precision, recall, f1 = 1.0, 1.0, 1.0
                category = CATEGORY_EXACT_MATCH
            else:
                category = CATEGORY_PARTIAL_MATCH

    precision = 0.0 if math.isnan(precision) else precision
    recall = 0.0 if math.isnan(recall) else recall
    f1 = 0.0 if math.isnan(f1) else f1

    return (precision, recall, f1, category)


def evaluate_queries(file_path: str) -> Dict[str, Any]:
    """
    Loads data, filters ignored questions, evaluates ALL remaining entries using thread pool,
    collects detailed results and macro scores. Empty generated queries handled by worker.

    Returns:
        Dict[str, Any]: Dictionary containing counts, macro scores, etc.
    """
    results_summary = {
        "total_loaded": 0,
        "filtered_ignored_question": 0,
        "total_attempted": 0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "category_counts": Counter(),
        "processing_errors": 0,
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results_summary["total_loaded"] = len(data)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file {file_path}: {e}")
        return {}

    entries_to_process = []
    for entry in data:
        if entry.get("question") not in questions_to_ignore:
            entries_to_process.append(entry)
        else:
            results_summary["filtered_ignored_question"] += 1

    total_attempted = len(entries_to_process)
    results_summary["total_attempted"] = total_attempted

    logger.info(
        f"Loaded {results_summary['total_loaded']} entries. "
        f"Filtered {results_summary['filtered_ignored_question']} ignored questions. "
        f"Attempting evaluation for {total_attempted} entries (including those with potentially empty generated queries)."
    )

    if not entries_to_process:
        logger.warning("No entries left to evaluate after filtering ignored questions.")
        return results_summary

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(calculate_metrics_for_entry, entry): entry
            for entry in entries_to_process
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=total_attempted,
            desc="Evaluating Queries",
        ):
            entry = futures[future]
            question = entry.get("question", "N/A")
            try:
                p, r, f1, category = future.result()
                total_precision += p
                total_recall += r
                total_f1 += f1
                results_summary["category_counts"][category] += 1
            except Exception as exc:
                logger.error(
                    f"Entry for question '{question}' generated an exception during future processing: {exc}",
                    exc_info=True,
                )
                results_summary["processing_errors"] += 1
                results_summary["category_counts"][CATEGORY_PROCESSING_ERROR] += 1

    if total_attempted > 0:
        results_summary["macro_precision"] = total_precision / total_attempted
        results_summary["macro_recall"] = total_recall / total_attempted
        results_summary["macro_f1"] = total_f1 / total_attempted

    results_summary["category_counts"] = dict(results_summary["category_counts"])
    return results_summary


def report_results(summary: Dict[str, Any]):
    """Prints the detailed evaluation results summary."""

    total_loaded = summary.get("total_loaded", 0)
    total_attempted = summary.get("total_attempted", 0)
    counts = summary.get("category_counts", {})

    print("\n--- Evaluation Summary ---")
    print(f"Total Entries Loaded: {total_loaded}")
    print(f"Filtered (Ignored Question): {summary.get('filtered_ignored_question', 0)}")
    print(f"Total Entries Attempted Evaluation: {total_attempted}")
    if summary.get("processing_errors", 0) > 0:
        print(f"*** Internal Processing Errors: {summary['processing_errors']} ***")

    if total_attempted == 0 and total_loaded > 0:
        print("\nNo queries attempted evaluation (check ignored questions filter).")
        return
    elif total_attempted == 0:
        print("\nNo queries loaded or evaluated.")
        return

    print("\n--- Result Categories ---")
    categories_order = [
        CATEGORY_EXACT_MATCH,
        CATEGORY_BOTH_EMPTY,
        CATEGORY_PARTIAL_MATCH,
        CATEGORY_GEN_EMPTY_GOLD_NOT,
        CATEGORY_GOLD_EMPTY_GEN_NOT,
        CATEGORY_NO_OVERLAP,
        CATEGORY_EXECUTION_FAILURE_GENERATED,
        CATEGORY_EXECUTION_FAILURE_GOLD,
        CATEGORY_EMPTY_QUERY_GENERATED,
        CATEGORY_PROCESSING_ERROR,
    ]

    executable_gen_count = 0
    correct_count = 0
    partial_count = 0
    total_failures = 0

    print(
        f"(Percentages based on {total_attempted} entries attempted after ignoring questions)"
    )
    for category in categories_order:
        count = counts.get(category, 0)
        if count > 0:
            percentage = (count / total_attempted) * 100 if total_attempted > 0 else 0
            print(f"  {category:<40}: {count:>5} ({percentage:>6.2f}%)")

            if category not in [
                CATEGORY_EMPTY_QUERY_GENERATED,
                CATEGORY_EXECUTION_FAILURE_GENERATED,
                CATEGORY_PROCESSING_ERROR,
            ]:
                executable_gen_count += count
            if category in [CATEGORY_EXACT_MATCH, CATEGORY_BOTH_EMPTY]:
                correct_count += count
            if category == CATEGORY_PARTIAL_MATCH:
                partial_count += count
            if category.startswith("Failure:") or category.startswith("Incorrect"):
                total_failures += count

    print("\n--- Derived Counts ---")
    empty_gen_count = counts.get(CATEGORY_EMPTY_QUERY_GENERATED, 0)
    gen_exec_fail_count = counts.get(CATEGORY_EXECUTION_FAILURE_GENERATED, 0)
    gold_exec_fail_count = counts.get(CATEGORY_EXECUTION_FAILURE_GOLD, 0)

    exec_gen_perc = (
        (executable_gen_count / total_attempted) * 100 if total_attempted > 0 else 0
    )
    correct_perc = (correct_count / total_attempted) * 100 if total_attempted > 0 else 0
    partial_perc = (partial_count / total_attempted) * 100 if total_attempted > 0 else 0
    total_fail_perc = (
        (total_failures / total_attempted) * 100 if total_attempted > 0 else 0
    )

    print(
        f"  Generated Query Non-Empty & Executable: {executable_gen_count:>5} ({exec_gen_perc:>6.2f}%)"
    )
    print(
        f"  Failure - Empty Generated Query       : {empty_gen_count:>5} ({(empty_gen_count/total_attempted*100):>6.2f}%)"
    )
    print(
        f"  Failure - Generated Query Execution   : {gen_exec_fail_count:>5} ({(gen_exec_fail_count/total_attempted*100):>6.2f}%)"
    )
    print(
        f"  Failure - Gold Query Execution        : {gold_exec_fail_count:>5} ({(gold_exec_fail_count/total_attempted*100):>6.2f}%)"
    )
    print(
        f"  Correct (Exact + Both Empty)      : {correct_count:>5} ({correct_perc:>6.2f}%)"
    )
    print(
        f"  Correct (Partial Match)           : {partial_count:>5} ({partial_perc:>6.2f}%)"
    )

    print("\n--- Macro Averaged Metrics (QALD Variant) ---")
    print(f"(Calculated over all {total_attempted} attempted entries)")
    print("(Empty queries & execution failures contribute 0.0 to scores)")
    macro_p = summary.get("macro_precision", 0.0)
    macro_r = summary.get("macro_recall", 0.0)
    macro_f1 = summary.get("macro_f1", 0.0)
    print(f"Macro Precision: {macro_p:.8f}")
    print(f"Macro Recall:    {macro_r:.8f}")
    print(f"Macro F1-Score:  {macro_f1:.8f}")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate/Evaluate generated SPARQL queries against gold standards using Wikidata. Handles empty generated queries."
    )
    parser.add_argument(
        "file_path", help="Path to the JSON file containing query pairs."
    )
    args = parser.parse_args()

    print(f"Starting evaluation for file: {args.file_path}")
    print(f"Using up to {MAX_WORKERS} concurrent workers.")
    print(f"Caching enabled: {requests_cache.is_installed()}")

    start_time = time.time()
    evaluation_summary = evaluate_queries(args.file_path)
    end_time = time.time()

    print(f"\nEvaluation finished in {end_time - start_time:.2f} seconds.")
    report_results(evaluation_summary)
