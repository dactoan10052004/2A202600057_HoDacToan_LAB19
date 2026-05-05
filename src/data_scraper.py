from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from src.corpus_loader import save_jsonl

LOGGER = logging.getLogger(__name__)

DEFAULT_COMPANY_PAGES = [
    "OpenAI",
    "Google",
    "Microsoft",
    "Apple Inc.",
    "Amazon (company)",
    "Meta Platforms",
    "Nvidia",
    "Tesla, Inc.",
    "IBM",
    "Oracle Corporation",
    "Intel",
    "Advanced Micro Devices",
    "Samsung Electronics",
    "Sony",
    "Salesforce",
    "Adobe Inc.",
    "Netflix",
    "Uber",
    "Airbnb",
    "Spotify",
    "ByteDance",
    "Tencent",
    "Alibaba Group",
    "DeepMind",
    "Anthropic",
    "Hugging Face",
    "Databricks",
    "Snowflake Inc.",
    "Palantir Technologies",
    "SpaceX",
    "YouTube",
    "Facebook",
    "PayPal",
    "LinkedIn",
    "GitHub",
    "GitLab",
    "Zoom Communications",
    "Slack Technologies",
    "Shopify",
    "Twitter",
    "X Corp.",
    "Cisco",
    "Qualcomm",
    "Broadcom Inc.",
    "Taiwan Semiconductor Manufacturing Company",
    "ASML Holding",
    "SAP",
    "ServiceNow",
    "Atlassian",
    "Dropbox",
    "Square (financial services)",
    "Stripe, Inc.",
    "Reddit",
    "Discord",
    "Pinterest",
    "Snap Inc.",
    "Baidu",
    "Xiaomi",
    "Huawei",
    "Grab (company)",
    "Didi Global",
]

RELATED_ENTITY_PAGES = [
    "YouTube",
    "Facebook",
]

INFOBOX_FIELDS = {
    "founder": "Founders",
    "founders": "Founders",
    "founded": "Founded",
    "foundation": "Founded",
    "hq_location": "Headquarters",
    "hq_location_city": "Headquarters",
    "headquarters": "Headquarters",
    "location_city": "Headquarters",
    "parent": "Parent company",
    "owner": "Owner",
    "key_people": "Key people",
    "industry": "Industry",
    "products": "Products",
    "services": "Services",
    "subsidiaries": "Subsidiaries",
    "acquired_by": "Acquired by",
}


def fetch_wikipedia_summary(title: str, user_agent: str) -> dict[str, Any]:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}"
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    extract = str(payload.get("extract", "")).strip()
    if not extract:
        raise ValueError(f"No extract found for {title}")
    row = {
        "title": payload.get("title") or title,
        "url": payload.get("content_urls", {}).get("desktop", {}).get("page", url),
        "extract": extract,
        "source": "wikipedia",
    }
    try:
        row["full_extract"] = fetch_wikipedia_extract(title, user_agent)
    except Exception as exc:
        LOGGER.warning("Could not fetch full extract for %s: %s", title, exc)
        row["full_extract"] = extract
    try:
        row["infobox"] = fetch_wikipedia_infobox(title, user_agent)
    except Exception as exc:
        LOGGER.warning("Could not fetch infobox for %s: %s", title, exc)
        row["infobox"] = {}
    return row


def _wiki_api_get(title: str, user_agent: str, extra_params: dict[str, Any]) -> dict[str, Any]:
    params = {
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
        "titles": title,
        **extra_params,
    }
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params=params,
        headers={"User-Agent": user_agent},
        timeout=25,
    )
    response.raise_for_status()
    return response.json()


def fetch_wikipedia_extract(title: str, user_agent: str, max_chars: int = 3500) -> str:
    payload = _wiki_api_get(
        title,
        user_agent,
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "exchars": str(max_chars),
        },
    )
    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        return ""
    return str(pages[0].get("extract", "")).strip()


def fetch_wikipedia_wikitext(title: str, user_agent: str) -> str:
    payload = _wiki_api_get(
        title,
        user_agent,
        {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
        },
    )
    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        return ""
    revisions = pages[0].get("revisions", [])
    if not revisions:
        return ""
    return str(revisions[0].get("slots", {}).get("main", {}).get("content", ""))


def _extract_template(wikitext: str, template_name: str) -> str:
    start_match = re.search(r"\{\{\s*" + re.escape(template_name), wikitext, flags=re.IGNORECASE)
    if not start_match:
        return ""
    start = start_match.start()
    depth = 0
    index = start
    while index < len(wikitext) - 1:
        pair = wikitext[index : index + 2]
        if pair == "{{":
            depth += 1
            index += 2
            continue
        if pair == "}}":
            depth -= 1
            index += 2
            if depth == 0:
                return wikitext[start:index]
            continue
        index += 1
    return ""


def _split_template_params(template: str) -> dict[str, str]:
    params: dict[str, str] = {}
    body = template.strip()
    if body.startswith("{{"):
        body = body[2:]
    if body.endswith("}}"):
        body = body[:-2]
    current: list[str] = []
    depth = 0
    parts: list[str] = []
    index = 0
    while index < len(body):
        pair = body[index : index + 2]
        if pair in ("{{", "[["):
            depth += 1
            current.append(pair)
            index += 2
            continue
        if pair in ("}}", "]]"):
            depth = max(0, depth - 1)
            current.append(pair)
            index += 2
            continue
        char = body[index]
        if char == "|" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    parts.append("".join(current))
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower().replace(" ", "_")
        value = clean_wikitext(value)
        if key and value:
            params[key] = value
    return params


def clean_wikitext(value: str) -> str:
    value = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL)
    value = re.sub(r"<br\s*/?>", "; ", value, flags=re.IGNORECASE)
    value = re.sub(r"<ref[^>/]*/>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"<ref[^>]*>.*?</ref>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"\{\{ubl\|", "{{plainlist|", value, flags=re.IGNORECASE)
    value = re.sub(r"\{\{(?:plainlist|flatlist|hlist|unbulleted list)\|", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\{\{start date(?: and age)?\|(\d{4})(?:\|[^}]*)?\}\}", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"\{\{nowrap\|([^{}]+)\}\}", r"\1", value, flags=re.IGNORECASE)
    value = re.sub(r"\{\{[^{}|]+\|([^{}]+)\}\}", r"\1", value)
    value = re.sub(r"\{\{[^{}]+\}\}", "", value)
    value = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", value)
    value = re.sub(r"\[\[([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"'''?", "", value)
    value = value.replace("}}", "")
    value = re.sub(r"&nbsp;", " ", value)
    value = re.sub(r"\s*\|\s*", " | ", value)
    value = re.sub(r"\s*;\s*", "; ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,;.\n\t")


def fetch_wikipedia_infobox(title: str, user_agent: str) -> dict[str, str]:
    wikitext = fetch_wikipedia_wikitext(title, user_agent)
    template = _extract_template(wikitext, "Infobox")
    if not template:
        template = _extract_template(wikitext, "Infobox company")
    if not template:
        return {}
    raw_fields = _split_template_params(template)
    return {
        label: raw_fields[key]
        for key, label in INFOBOX_FIELDS.items()
        if key in raw_fields and raw_fields[key]
    }


def scrape_companies(
    titles: list[str] | None,
    user_agent: str,
    sleep_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for title in titles or DEFAULT_COMPANY_PAGES:
        try:
            rows.append(fetch_wikipedia_summary(title, user_agent))
            LOGGER.info("Fetched Wikipedia summary for %s", title)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", title, exc)
        time.sleep(sleep_seconds)
    return rows


def write_text_corpus(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for row in rows:
        block_lines = [f"# {row['title']}", str(row.get("extract", ""))]
        infobox = row.get("infobox") or {}
        if infobox:
            block_lines.append("\nStructured facts from Wikipedia infobox:")
            for label, value in infobox.items():
                block_lines.append(f"- {label}: {value}")
        full_extract = str(row.get("full_extract") or "").strip()
        if full_extract and full_extract != row.get("extract"):
            block_lines.append("\nAdditional article extract:")
            block_lines.append(full_extract)
        blocks.append("\n".join(block_lines))
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def save_corpus(rows: list[dict[str, Any]], jsonl_path: str | Path, txt_path: str | Path) -> None:
    save_jsonl(rows, jsonl_path)
    write_text_corpus(rows, txt_path)
