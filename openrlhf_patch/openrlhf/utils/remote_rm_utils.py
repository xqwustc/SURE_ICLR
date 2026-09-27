import os
import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

_sticky_urls = {}


def _ordered_urls(url_spec):
    urls = [url.strip() for url in url_spec.split(",") if url.strip()]
    if not urls:
        raise ValueError("At least one remote RM URL is required")

    sticky_url = _sticky_urls.get(url_spec)
    if sticky_url in urls:
        return [sticky_url] + [url for url in urls if url != sticky_url]
    return urls


def request_api_wrapper(url, data, score_key="rewards"):
    """Call one of the candidate RM endpoints and keep using the last healthy one."""
    headers = {
        "Content-Type": "application/json",
    }
    connect_timeout = float(os.environ.get("REMOTE_RM_CONNECT_TIMEOUT", "2"))
    read_timeout = float(os.environ.get("REMOTE_RM_READ_TIMEOUT", "1800"))
    retry_seconds = float(os.environ.get("REMOTE_RM_RETRY_SECONDS", "1200"))
    poll_interval = float(os.environ.get("REMOTE_RM_POLL_INTERVAL", "5"))
    deadline = time.monotonic() + retry_seconds
    last_report = 0.0
    last_errors = {}

    while True:
        for candidate_url in _ordered_urls(url):
            try:
                response = requests.post(
                    url=candidate_url,
                    json=data,
                    headers=headers,
                    timeout=(connect_timeout, read_timeout),
                )
                response.raise_for_status()  # Raise an HTTPError for bad responses
                response = response.json()
                assert score_key in response, f"{score_key} not in {response}"
                if _sticky_urls.get(url) != candidate_url:
                    logger.info(f"Using remote RM endpoint: {candidate_url}")
                _sticky_urls[url] = candidate_url
                if "uncertainties" in response:
                    return response.get(score_key), response.get("uncertainties")
                return response.get(score_key)
            except requests.HTTPError as e:
                detail = e.response.text[:1000] if e.response is not None else ""
                first_query = data.get("query", [None])[0] if data.get("query") else None
                query_tail = repr(first_query)[-300:]
                last_errors[candidate_url] = (
                    f"{e}; server={detail}; first_query_tail={query_tail!r}"
                )
            except requests.RequestException as e:
                last_errors[candidate_url] = str(e)
            except Exception as e:
                last_errors[candidate_url] = str(e)

            if _sticky_urls.get(url) == candidate_url:
                _sticky_urls.pop(url, None)

        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            break
        if now - last_report >= 60:
            error_summary = "; ".join(
                f"{candidate}: {error[:240]}" for candidate, error in last_errors.items()
            )
            logger.info(
                "All remote RM endpoints are unavailable; polling for another %.0f seconds. Last errors: %s",
                remaining,
                error_summary,
            )
            last_report = now
        time.sleep(min(poll_interval, remaining))

    raise Exception(
        f"All remote RM endpoints remained unavailable for {retry_seconds:.0f} seconds: {_ordered_urls(url)}"
    )


def remote_rm_fn(api_url, queries, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    scores = request_api_wrapper(api_url, {"query": queries}, score_key)
    if isinstance(scores, tuple):
        scores, uncertainties = scores
        uncertainties = torch.tensor(uncertainties)
        scores = torch.tensor(scores)
        return scores, uncertainties
    return torch.tensor(scores)


@ray.remote
def remote_rm_fn_ray(api_url, queries, score_key="rewards"):
    return remote_rm_fn(api_url, queries, score_key)


if __name__ == "__main__":
    # test utils
    url = "http:xxx/get_rm_score"
    score = remote_rm_fn(url, ["example query"], ["example response"])
    print(score)
