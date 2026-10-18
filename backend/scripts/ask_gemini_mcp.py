"""Ask Gemini a household member's question, answered from LastScene's read-only MCP server.

    python scripts/ask_gemini_mcp.py "When did I last have my wallet?"
    python scripts/ask_gemini_mcp.py --list-tools        # check the token against /mcp, no Gemini call

Gemini (Interactions API) calls https://www.lastscene.photo/mcp itself, with the token
below as its bearer header, then answers from what the tools return. See docs/MCP.md.

Reads from the env file (default backend/.env; real environment variables win):
    GEMINI_API_KEY        https://aistudio.google.com/apikey
    LASTSCENE_MCP_TOKEN   a household MCP key (lsk_..., Household page > MCP keys) or a
                          household member's Auth0 access token. Prefer a key: it only reads one
                          household and can be revoked.
    LASTSCENE_MCP_URL     optional, default https://www.lastscene.photo/mcp

The token is never printed. It is sent to Google for this one call (Gemini needs it to
reach /mcp), and the interaction is not stored there (store=False).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

DEFAULT_ENV = Path(__file__).resolve().parents[1] / ".env"
DEFAULT_URL = "https://www.lastscene.photo/mcp"  # www: the apex redirects and drops the header
MODEL = "gemini-3.8-flash"
TOOLS = ["list_households", "search_memories", "item_last_seen", "timeline", "items", "missing_times", "places"]
SYSTEM = (
    "You help someone find their things. Answer from the LastScene tools only: "
    "call list_households first, then the tools that fit. Say when something was seen and where, in "
    "plain words and local-sounding times, and say so if the data doesn't show it. Be brief and kind."
)


def read_env(path: Path) -> dict[str, str]:
    """KEY=value lines; real environment variables win."""
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip("'\"")
    return {**values, **{k: v for k, v in os.environ.items() if k in values or k.startswith(("LASTSCENE_", "GEMINI_"))}}


def need(env: dict[str, str], name: str) -> str:
    if not env.get(name):
        sys.exit(f"{name} is not set (in the env file or the environment)")
    return env[name]


def list_tools(url: str, token: str) -> None:
    """tools/list straight from /mcp: proves the URL and token work, costs no Gemini call."""
    r = httpx.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream",
                 "MCP-Protocol-Version": "2025-06-18"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        timeout=30,
    )
    if r.status_code != 200:
        sys.exit(f"/mcp answered {r.status_code}: {r.text[:200]}")  # error bodies never echo the token
    for tool in r.json()["result"]["tools"]:
        print(f"{tool['name']}: {tool['description'].splitlines()[0]}")


def ask(question: str, url: str, token: str, api_key: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    interaction = client.interactions.create(
        model=MODEL,
        system_instruction=SYSTEM,
        input=question,
        tools=[{
            "type": "mcp_server",
            "name": "lastscene",  # snake_case, no "-"
            "url": url,
            "headers": {"Authorization": f"Bearer {token}"},
            "allowed_tools": [{"mode": "auto", "tools": TOOLS}],
        }],
        store=False,  # personal data: don't keep the conversation on Google's side
    )
    return interaction.output_text or "(no answer)"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("question", nargs="?", default="When did I last have my wallet?")
    p.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    p.add_argument("--list-tools", action="store_true", help="only list the server's tools")
    p.add_argument("--json", action="store_true", help="print the answer as JSON")
    args = p.parse_args()

    env = read_env(args.env_file)
    url = env.get("LASTSCENE_MCP_URL") or DEFAULT_URL
    token = need(env, "LASTSCENE_MCP_TOKEN")
    if args.list_tools:
        list_tools(url, token)
        return
    answer = ask(args.question, url, token, need(env, "GEMINI_API_KEY"))
    print(json.dumps({"question": args.question, "answer": answer}) if args.json else answer)


if __name__ == "__main__":
    main()
