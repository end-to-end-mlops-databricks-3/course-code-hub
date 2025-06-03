from marvelous.github_app import acquire_access_token
from marvelous.common import get_dbr_host
import argparse
import os
import requests
from loguru import logger

parser = argparse.ArgumentParser()

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--repo",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
host = get_dbr_host()

org = args.org
repo = args.repo
git_sha = args.git_sha
job_id = args.job_id
run_id = args.job_run_id


token = acquire_access_token(private_key=os.environ["PRIVATE_KEY"])
url = f"https://api.github.com/repos/{org}/{repo}/statuses/{git_sha}"
link_to_databricks_run = f"{host}/jobs/{job_id}/runs/{run_id}"

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {token}",
    "X-GitHub-Api-Version": "2022-11-28"
}

payload = {
    "state": "success",
    "target_url": f"{link_to_databricks_run}",
    "description": "Integration test is succesful!",
    "context": "integration-testing/databricks"
}

response = requests.post(url, headers=headers, json=payload)
logger.info("Status:", response.status_code)
