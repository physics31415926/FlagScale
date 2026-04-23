"""Web fetch tool — retrieve and extract content from URLs."""

import re

import requests
from bs4 import BeautifulSoup

from flagscale.agent.react.tools.base import Tool


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a URL and extract its main text content. Useful for reading documentation, GitHub pages, error references, etc."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
        },
        "required": ["url"],
    }
    max_result_size = 80000

    def __init__(self, timeout: int = 30, proxies: dict = None):
        self._timeout = timeout
        self._proxies = proxies

    def execute(self, **kwargs) -> str:
        url = kwargs["url"]
        try:
            resp = requests.get(
                url,
                timeout=self._timeout,
                headers={"User-Agent": "FlagScale-Agent/1.0"},
                allow_redirects=True,
                proxies=self._proxies,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            msg = f"ERROR: Failed to fetch {url}: {e}"
            if not self._proxies and _is_network_error(str(e)):
                msg += _PROXY_HINT
            return msg

        content_type = resp.headers.get("Content-Type", "")

        if "text/plain" in content_type or url.endswith((".txt", ".log", ".yaml", ".yml", ".json", ".md", ".rst", ".cfg", ".ini", ".toml")):
            return resp.text

        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return f"ERROR: Unsupported content type: {content_type}"

        return _extract_text(resp.text, url)


def _extract_text(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"role": "main"})
        or soup.find("div", class_=re.compile(r"(content|article|post|entry|readme)", re.I))
    )

    target = main or soup.body or soup
    text = target.get_text(separator="\n", strip=True)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    result = "\n".join(lines)

    if len(result) < 50:
        return f"[Fetched {url} but extracted very little content ({len(result)} chars)]\n{result}"

    return result


_PROXY_HINT = (
    "\n\n💡 Network error detected and no proxy configured. "
    "Set proxy in ~/.flagscale/agent.yaml:\n"
    "  shell_env:\n"
    '    HTTP_PROXY: "http://host:port"\n'
    '    HTTPS_PROXY: "http://host:port"\n'
    "Then use /reload to apply."
)


def _is_network_error(msg: str) -> bool:
    patterns = (
        "ConnectionError", "ConnectTimeout", "ProxyError",
        "SSLError", "NewConnectionError", "MaxRetryError",
        "Connection refused", "Name or service not known",
        "Temporary failure in name resolution",
        "Network is unreachable", "No route to host",
    )
    return any(p.lower() in msg.lower() for p in patterns)
