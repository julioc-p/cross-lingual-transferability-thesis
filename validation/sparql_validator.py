import json
import requests
import argparse
import concurrent.futures
import matplotlib.pyplot as plt
import logging
import time
import requests_cache
from typing import Dict, Any, Tuple, Set, Optional
from tqdm import tqdm

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
            if was_cached:
                pass
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
    """
    if not query or not query.strip():
        logger.warning("Attempted to validate an empty query.")
        return False
    validation_query = query
    upper_query = query.strip().upper()
    if upper_query.startswith("SELECT") and "LIMIT" not in upper_query:
        validation_query = f"{query.rstrip(' ;')} LIMIT 1"
    response = retry_request(
        SPARQL_ENDPOINT,
        {"query": validation_query, "format": "json"},
        HEADERS,
        timeout=30,
    )
    if response is None:
        logger.warning(
            f"Validation failed for query (after potential retries). Original query snippet: {query[:100]}..."
        )
        return False
    elif response.status_code != 200:
        logger.warning(
            f"Validation failed for query with status code {response.status_code}. Query snippet: {query[:100]}..."
        )
        return False
    return True


def safe_add_limit(query: str, limit: int = 100) -> str:
    """
    Adds a LIMIT clause to SELECT queries if one doesn't exist.
    Does not modify ASK, INSERT, DELETE queries.
    """
    query = query.strip()
    if not query:
        return ""
    upper_query = query.upper()
    if upper_query.startswith("SELECT") and "LIMIT" not in upper_query:
        if query.endswith(";"):
            query = query[:-1].rstrip()
        return f"{query} LIMIT {limit}"
    else:
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
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for query: {query[:100]}... Error: {e}")
            logger.error(f"Response text: {response.text[:500]}")
            return None
    else:
        logger.error(
            f"Failed to execute SPARQL query after retries or due to non-200 status. Query: {query[:100]}..."
        )
        return None


def extract_results(json_response: Optional[Dict[str, Any]]) -> Set:
    """
    Extracts a comparable set of results from SPARQL JSON response.
    Handles None input gracefully.
    """
    if json_response is None:
        return set()
    try:
        if "boolean" in json_response:
            return {str(json_response["boolean"])}
        if "results" in json_response and "bindings" in json_response["results"]:
            results = set()
            vars = json_response.get("head", {}).get("vars", [])
            if not vars:
                return set()
            for binding in json_response["results"]["bindings"]:
                result_tuple = tuple(binding.get(var, {}).get("value") for var in vars)
                results.add(result_tuple)
            return results
    except Exception as e:
        logger.error(
            f"Error extracting results from JSON: {e}. Response snippet: {str(json_response)[:500]}"
        )
    return set()


def process_entry(entry: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Evaluates a single pair of gold and generated queries for one entry.
    Returns (is_valid, is_exact_match, is_partial_match).
    """
    question = entry.get("question", "N/A")
    generated_query = entry.get("generated_sparql", "").strip()
    gold_query = entry.get("gold_sparql", "").strip()
    if not generated_query:
        logger.warning(
            f"Skipping entry due to empty generated query. Question: {question}"
        )
        return (0, 0, 0)
    if not validate_sparql(generated_query):
        logger.warning(
            f"Generated query failed syntax validation. Question: {question}"
        )
        return (0, 0, 0)
    is_valid = 1
    gold_json = execute_sparql(gold_query)
    gen_json = execute_sparql(safe_add_limit(generated_query))
    if gold_json is None:
        logger.error(
            f"Failed to execute or parse GOLD query. Cannot compare. Question: {question}"
        )
        return (is_valid, 0, 0)
    if gen_json is None:
        logger.error(
            f"Failed to execute or parse GENERATED query (post-validation). Cannot compare. Question: {question}"
        )
        return (is_valid, 0, 0)
    gold_results = extract_results(gold_json)
    gen_results = extract_results(gen_json)
    is_exact_match = 0
    is_partial_match = 0
    if not gold_results and not gen_results:
        is_exact_match = 1
    elif not gold_results and gen_results:
        is_exact_match = 0
        is_partial_match = 0
    elif gold_results and not gen_results:
        is_exact_match = 0
        is_partial_match = 0
    elif gold_results == gen_results:
        is_exact_match = 1
    elif gold_results & gen_results:
        is_partial_match = 1
    return (is_valid, is_exact_match, is_partial_match)


def evaluate_queries(file_path: str) -> Tuple[int, int, int, int]:
    """
    Loads data, filters ignored questions, and evaluates all query pairs using a thread pool.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        return (0, 0, 0, 0)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file {file_path}: {e}")
        return (0, 0, 0, 0)
    original_count = len(data)
    data = [entry for entry in data if entry.get("question") not in questions_to_ignore]
    filtered_count = len(data)
    logger.info(
        f"Loaded {original_count} entries, evaluating {filtered_count} after filtering."
    )
    if not data:
        logger.warning("No entries left to evaluate after filtering.")
        return (0, 0, 0, 0)
    total = filtered_count
    valid_count = 0
    correct_count = 0
    partial_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_entry, entry) for entry in data]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=total,
            desc="Evaluating Queries",
        ):
            try:
                v, c, p = future.result()
                valid_count += v
                correct_count += c
                partial_count += p
            except Exception as exc:
                logger.error(f"An entry generated an exception: {exc}", exc_info=True)
    return total, valid_count, correct_count, partial_count


def plot_results(total: int, valid: int, correct: int, partial: int):
    """Generates and displays a bar chart of the evaluation results."""
    if total == 0:
        print("\nNo queries evaluated, cannot plot results.")
        return
    labels = ["Total Evaluated", "Syntactically Valid", "Exact Match", "Partial Match"]
    values = [total, valid, correct, partial]
    valid_perc = (valid / total * 100) if total else 0
    correct_perc = (correct / total * 100) if total else 0
    partial_perc = (partial / total * 100) if total else 0
    correct_perc_of_valid = (correct / valid * 100) if valid else 0
    print("\n--- Evaluation Summary ---")
    print(f"Total Queries Evaluated: {total}")
    print(f"Executable Queries: {valid} ({valid_perc:.2f}%)")
    print(
        f"Exact Match Answers: {correct} ({correct_perc:.2f}% of total, {correct_perc_of_valid:.2f}% of executable)"
    )
    print(f"Partial Match Answers: {partial} ({partial_perc:.2f}% of total)")
    print("--------------------------")
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=["gray", "blue", "green", "orange"])
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                yval,
                int(yval),
                va="bottom",
                ha="center",
            )
        plt.xlabel("Evaluation Categories")
        plt.ylabel("Number of Queries")
        plt.title("SPARQL Query Evaluation Results Summary")
        plt.ylim(0, total * 1.1)
        plt.tight_layout()
        plt.show()
    except ImportError:
        logger.warning(
            "Matplotlib not found. Skipping plot generation. Install with: pip install matplotlib"
        )
    except Exception as e:
        logger.error(f"An error occurred during plotting: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate and evaluate generated SPARQL queries against gold standards using Wikidata."
    )
    parser.add_argument(
        "file_path", help="Path to the JSON file containing query pairs."
    )
    args = parser.parse_args()
    print(f"Starting evaluation for file: {args.file_path}")
    print(f"Using {MAX_WORKERS} concurrent workers.")
    print(f"Caching enabled: {requests_cache.is_installed()}")
    start_time = time.time()
    total, valid, correct, partial = evaluate_queries(args.file_path)
    end_time = time.time()
    print(f"\nEvaluation finished in {end_time - start_time:.2f} seconds.")
    plot_results(total, valid, correct, partial)
