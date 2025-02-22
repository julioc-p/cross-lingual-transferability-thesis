from collections import defaultdict
import requests
import os
from tqdm import tqdm
import requests_cache
import logging
import time
import collections
import json

HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
BASE_API_URL = "https://api.github.com"
BASE_REPOS = [
    "https://github.com/ag-sc/QALD",
    "https://github.com/KGQA/QALD-10",
    "https://github.com/KGQA/QALD_9_plus",
]
EXTENSIONS = ["xml", "json"]
OUTPUT_PATH = "sources/qald_urls.json"

# Cache requests for 7 days
requests_cache.install_cache("github_cache", expire_after=604800)


def get_all_data_files(repo_url):
    json_urls = []
    xml_urls = []

    urls = get_urls(repo_url)
    for ext in urls:
        for url in urls[ext]:
            # if "data" is not in path ignore
            if "data" not in url:
                continue
            if ext[-4:] == "json":
                json_urls.extend(urls[ext])
            elif ext[-3:] == "xml":
                xml_urls.extend(urls[ext])
    return json_urls, xml_urls


def make_request(url):
    """
    Makes an HTTP GET request to the provided URL with error handling and rate limit handling.

    Args:
        url (str): The URL to make the request to.

    Returns:
        requests.Response or None: The response object if the request was successful, None otherwise.
    """
    print("making request to url: ", url)
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making request to {url}: {e}")
        return None

    if "retry_after" in response.headers:
        logging.warning(
            f'Rate limit exceeded. Retrying after {response.headers["retry-after"]} seconds'
        )
        time.sleep(int(response.headers["retry-after"]))
    elif (
        "x-ratelimit-remaining" in response.headers
        and int(response.headers["x-ratelimit-remaining"]) == 0
    ):
        logging.warning(
            f'Rate limit exceeded. Retrying after {response.headers["x-ratelimit-remaining"]} seconds'
        )
        time.sleep(int(response.headers["x-ratelimit-reset"]))
    return response


def get_urls(
    repo_url: str,
    default_branch: str = "",
    tree_sha: str = "",
    file_path: str = "",
    res: defaultdict = None,
) -> defaultdict:
    """
    Retrieves the URLs of files with specific extensions from a GitHub repository.

    Args:
        repo_url (str): The URL of the GitHub repository.
        default_branch (str, optional): The default branch of the repository. Defaults to "".
        tree_sha (str, optional): The SHA of the tree object. Defaults to "".
        file_path (str, optional): The path to a specific file or directory within the repository. Defaults to "".
        res (defaultdict, optional): A defaultdict to store the URLs of files with specific extensions. Defaults to None.

    Returns:
        defaultdict: A defaultdict containing the URLs of files with specific extensions.

    """
    if res is None:
        res = defaultdict(list)

    if not default_branch:
        default_branch = get_default_branch(repo_url)

    if not tree_sha:
        tree_sha = default_branch

    url = f'{BASE_API_URL}/repos/{repo_url.split("https://github.com/")[1]}/git/trees/{tree_sha}?recursive=1'
    response = make_request(url).json()
    truncated = False
    if response.get("truncated"):
        logging_path = file_path if file_path else "root"
        logging.info(
            f"request for files in path {logging_path} in repository: {repo_url} got truncated because of file number limit"
        )
        truncated = True
        url = f'{BASE_API_URL}/repos/{repo_url.split("https://github.com/")[1]}/git/trees/{tree_sha}'
        response = make_request(url).json()

    tree = response.get("tree")

    if not tree:
        return res

    base_download_url = f'https://raw.githubusercontent.com/{repo_url.split("https://github.com/")[1]}/{default_branch}/'

    for file in tree:
        ext = file.get("path").split(".")[-1]
        new_file_path = f"{file_path}/{file['path']}" if file_path else file["path"]
        if file.get("type") == "blob" and ext in EXTENSIONS:
            res[ext].append(base_download_url + new_file_path)

        if truncated and file.get("type") == "tree":
            logging.debug(f"Recursively fetching URLs for path {new_file_path}")
            get_urls(repo_url, default_branch, file.get("sha"), new_file_path, res)

    return res


def get_default_branch(repo_url: str) -> str:
    """
    Retrieves the default branch of a GitHub repository.

    Args:
        repo_url (str): The URL of the GitHub repository.

    Returns:
        str: The name of the default branch.

    Raises:
        requests.exceptions.RequestException: If there is an error making the HTTP request.

    """
    url = f'{BASE_API_URL}/repos/{repo_url.split("https://github.com/")[1]}'

    response = make_request(url)

    return response.json().get("default_branch")


if __name__ == "__main__":
    json_urls, xml_urls = [], []
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    for repo in BASE_REPOS:
        json_urls_repo, xml_urls_repo = get_all_data_files(repo)
        json_urls_repo = list(set(json_urls_repo))
        xml_urls_repo = list(set(xml_urls_repo))
        json_urls.extend(json_urls_repo)
        xml_urls.extend(xml_urls_repo)
    # print(json_urls, xml_urls)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"json": json_urls, "xml": xml_urls}, f)
