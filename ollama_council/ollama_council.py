#!/usr/bin/env python3
"""
ollama_council.py

Multi-agent "council" wrapper for Ollama with filebase-aware RAG.

Architecture:
  - OllamaClient: thin HTTP client for Ollama /api/chat
  - AgentResult: dataclass for each agent's output
  - CouncilCoordinator: high-level orchestrator that:
      * fetches RAG context
      * runs agents in parallel
      * collects votes
      * resolves ties with a judge model
  - main(): thin CLI wrapper around CouncilCoordinator

Usage:
    python ollama_council.py "Your question about the project"
"""

from __future__ import annotations

import sys
import os
import json
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from agents import AGENTS, AGENT_MODEL, JUDGE_MODEL
from rag_helper import get_context


# -----------------------------
# CONFIGURATION
# -----------------------------

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAX_WORKERS = 5  # how many agents to run in parallel
REQUEST_TIMEOUT = 600


# -----------------------------
# AGENT RULES
# -----------------------------

GLOBAL_AGENT_RULES = """
You are one of several agents in a council. You must follow these universal rules:

1. You MUST answer the user's prompt directly, from your own perspective.
2. You MUST stay fully in your own persona and style.
3. You MUST NOT talk about other agents, critique them, or reference their existence.
4. You MUST output ONLY a valid JSON object with exactly these keys:
   - "response": your answer to the user, as a string.
   - "vote": the NAME (string) of ONE other agent that you expect will provide
             the most correct or helpful answer, based on your perspective.

5. The ONLY thing in your final message must be the JSON object.
   - No commentary before or after.
   - No markdown.
   - No extra fields.
   - No trailing commas.
   - No explanations outside the JSON.

Valid example output:
{
  "response": "Your full answer goes here.",
  "vote": "Engineer"
}
"""


# -----------------------------
# OLLAMA CLIENT
# -----------------------------

class OllamaClient:
    """
    Thin wrapper around Ollama's /api/chat endpoint.
    """

    def __init__(self, base_url: str = OLLAMA_URL, timeout: int = REQUEST_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        stream: bool = False,
        json_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Call /api/chat on the Ollama server.

        If json_mode=True, we request structured JSON output via "format": "json".
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if json_mode:
            payload["format"] = "json"

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def get_content(response: Dict[str, Any]) -> str:
        """
        Extract assistant content from the /api/chat response.
        """
        try:
            return response["message"]["content"]
        except KeyError:
            raise ValueError(f"Unexpected response structure: {response!r}")


# -----------------------------
# DATA MODEL
# -----------------------------

@dataclass
class AgentResult:
    """
    Holds a single agent's result.

    - name: agent name (e.g., "Detective")
    - description: short description from agents.py
    - response: natural language answer from that agent
    - vote: name of the agent this one voted for (or None if invalid)
    """
    name: str
    description: str
    response: str
    vote: Optional[str]


# -----------------------------
# COUNCIL COORDINATOR
# -----------------------------

class CouncilCoordinator:
    """
    High-level orchestrator for the multi-agent council.

    Responsibilities:
      - Retrieve RAG context
      - Build prompts per agent
      - Run agents in parallel
      - Tally votes and resolve ties (if any)
      - Return the final combined answer as a string
    """

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        agents: Optional[List[Dict[str, Any]]] = None,
        max_workers: int = MAX_WORKERS,
    ) -> None:
        self.client = client or OllamaClient()
        self.agents = agents or AGENTS
        self.max_workers = min(max_workers, len(self.agents)) if self.agents else 1

    # ---------- Public API ----------

    def ask(self, user_prompt: str) -> str:
        """
        Main pipeline:
          1. Get RAG context
          2. Run agents in parallel
          3. Tally votes
          4. Return winner's response (or all responses if no valid votes)
        """
        # 1) Retrieve filebase context
        context = self._get_context_with_progress(user_prompt)

        # 2) Run all agents in parallel
        results = self._run_agents_with_progress(user_prompt, context)

        if not results:
            raise RuntimeError("No agent outputs produced. Check council logs.")

        # 3) Tally votes
        print("[*] Tallying votes...", file=sys.stderr)
        tally = self._tally_votes(results)

        # If no valid votes, fall back to printing all responses
        if not tally:
            print("[!] No valid votes were cast. Returning all agent responses.\n", file=sys.stderr)
            return self._format_all_responses(results)

        winners, max_votes = self._choose_winner_from_tally(tally)
        result_by_name = {r.name: r for r in results}

        # 4) Determine winner or run tie-breaker
        if len(winners) == 1:
            winner_name = winners[0]
            winner_result = result_by_name[winner_name]
            print(f"[+] Winner by majority vote: {winner_name} ({max_votes} votes)", file=sys.stderr)
            return self._format_majority_winner(winner_name, winner_result)
        else:
            print(f"[*] Tie detected ({max_votes} votes each) for: {', '.join(winners)}", file=sys.stderr)
            print("[*] Running impartial tie-breaker...", file=sys.stderr)

            tied_results = [result_by_name[name] for name in winners]
            winner_name, reasoning = self._run_tie_breaker_with_progress(
                user_prompt, context, tied_results
            )
            print(f"[+] Tie resolved. Winner: {winner_name}", file=sys.stderr)
            print(f"[+] Tie-breaker reasoning: {reasoning}", file=sys.stderr)

            winning_result = result_by_name.get(winner_name, tied_results[0])
            return self._format_tie_winner(winner_name, winning_result, reasoning)

    def run_cli(self, argv: Optional[List[str]] = None) -> None:
        """
        Thin CLI wrapper: parse args, run ask(), print final answer to stdout.
        """
        if argv is None:
            argv = sys.argv[1:]

        if not argv:
            print("Usage: python ollama_council.py \"Your question about the project\"", file=sys.stderr)
            sys.exit(1)

        user_prompt = " ".join(argv).strip()
        answer = self.ask(user_prompt)
        print(answer)

    # ---------- RAG / Context ----------

    def _get_context_with_progress(self, user_prompt: str) -> str:
        """
        Fetch RAG context with a small progress bar.
        """
        print("[*] Retrieving filebase context via RAG...", file=sys.stderr)
        with tqdm(
            total=1,
            desc="RAG Context",
            unit="step",
            file=sys.stderr,
            leave=False,
        ) as pbar:
            context = get_context(user_prompt)
            pbar.update(1)
        return context

    # ---------- Agent Prompt Building ----------

    def _build_agent_user_content(
        self,
        agent: Dict[str, Any],
        user_prompt: str,
        context: str,
    ) -> str:
        """
        Combine user question + RAG context into a user message tailored to an agent.
        """
        return textwrap.dedent(f"""
            You are the {agent['name']} agent.

            Below is relevant context from the project's filebase. Use it as your primary source
            when analyzing or answering. If something is not covered by the context,
            say that clearly instead of hallucinating.

            === FILEBASE CONTEXT START ===
            {context}
            === FILEBASE CONTEXT END ===

            User request:
            {user_prompt}

            As the {agent['name']} agent, focus specifically on your specialization/persona:
            {agent['description']}

            Provide a concise but detailed answer.
        """).strip()

    def _build_agent_messages(
        self,
        agent: Dict[str, Any],
        user_prompt: str,
        context: str,
        other_agent_names: List[str],
    ) -> List[Dict[str, str]]:
        """
        Build /api/chat messages for a single agent, using a global ruleset
        plus the agent's own persona prompt.
        """
        agent_user_content = self._build_agent_user_content(agent, user_prompt, context)
        valid_votes_list = ", ".join(sorted(n for n in other_agent_names if n != agent["name"]))

        json_instruction = textwrap.dedent(f"""
            Remember:
            - In the "vote" field of your JSON, you MUST choose exactly ONE of these agents:
              [{valid_votes_list}]
            - You CANNOT vote for yourself.
        """).strip()

        return [
            {
                "role": "system",
                "content": GLOBAL_AGENT_RULES,
            },
            {
                "role": "system",
                "content": agent["system_prompt"],
            },
            {
                "role": "user",
                "content": agent_user_content + "\n\n" + json_instruction,
            },
        ]

    # ---------- Agent Execution ----------

    def _run_single_agent(
        self,
        agent: Dict[str, Any],
        user_prompt: str,
        context: str,
        all_agent_names: List[str],
    ) -> AgentResult:
        """
        Run a single agent and parse its JSON {response, vote}.
        """
        other_agent_names = [n for n in all_agent_names if n != agent["name"]]
        messages = self._build_agent_messages(agent, user_prompt, context, other_agent_names)

        try:
            resp = self.client.chat(AGENT_MODEL, messages, json_mode=True)
            content = self.client.get_content(resp)

            # In json_mode, the assistant content should be JSON text.
            parsed = json.loads(content)

            response_text = str(parsed.get("response", "")).strip()
            vote = parsed.get("vote")

            if vote not in other_agent_names:
                # Invalid vote target; ignore it.
                print(f"[!] {agent['name']} produced invalid vote '{vote}'. Ignoring.", file=sys.stderr)
                vote = None

            if not response_text:
                response_text = (
                    f"Error: agent {agent['name']} produced empty 'response'. "
                    f"Raw JSON: {parsed!r}"
                )

            return AgentResult(
                name=agent["name"],
                description=agent["description"],
                response=response_text,
                vote=vote,
            )

        except Exception as e:
            # Preserve failure, but still return an AgentResult so the council can proceed.
            print(f"[!] {agent['name']} execution failed: {e}", file=sys.stderr)
            return AgentResult(
                name=agent["name"],
                description=agent["description"],
                response=f"Error during agent execution: {e}",
                vote=None,
            )

    def _run_agents_with_progress(
        self,
        user_prompt: str,
        context: str,
    ) -> List[AgentResult]:
        """
        Run all agents in parallel with a progress bar.
        """
        print("[*] Running agents...", file=sys.stderr)
        if not self.agents:
            print("[!] No agents configured.", file=sys.stderr)
            return []

        agent_names = [a["name"] for a in self.agents]
        results: List[AgentResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_agent = {
                executor.submit(
                    self._run_single_agent,
                    agent,
                    user_prompt,
                    context,
                    agent_names,
                ): agent
                for agent in self.agents
            }

            with tqdm(
                total=len(future_to_agent),
                desc="Agents",
                unit="agent",
                file=sys.stderr,
            ) as pbar:
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"[!] {agent['name']} agent crashed: {e}", file=sys.stderr)
                    finally:
                        pbar.update(1)

        return results

    # ---------- Voting & Tie-breaker ----------

    def _tally_votes(self, results: List[AgentResult]) -> Dict[str, int]:
        """
        Count how many votes each agent received.
        """
        tally: Dict[str, int] = {}
        for r in results:
            if r.vote is not None:
                tally[r.vote] = tally.get(r.vote, 0) + 1
                print(f"    - {r.name} voted for {r.vote}", file=sys.stderr)
            else:
                print(f"    - {r.name} cast no valid vote.", file=sys.stderr)
        return tally

    @staticmethod
    def _choose_winner_from_tally(tally: Dict[str, int]) -> Tuple[List[str], int]:
        """
        Return (winners, max_votes) based on majority vote.
        """
        if not tally:
            return [], 0
        max_votes = max(tally.values())
        winners = [name for name, count in tally.items() if count == max_votes]
        return winners, max_votes

    def _build_tie_breaker_messages(
        self,
        user_prompt: str,
        context: str,
        tied_results: List[AgentResult],
    ) -> List[Dict[str, str]]:
        """
        Build messages for the tie-breaker judge.
        """
        candidates_block = []
        for r in tied_results:
            candidates_block.append(textwrap.dedent(f"""
                ### Candidate: {r.name}
                Role: {r.description}

                Response:
                {r.response}
            """).strip())

        candidates_joined = "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(candidates_block)

        user_content = textwrap.dedent(f"""
            The user asked the following question:
            >>> {user_prompt}

            Context from the filebase was used:
            === FILEBASE CONTEXT START ===
            {context}
            === FILEBASE CONTEXT END ===

            The council of agents has produced multiple candidate answers that are tied by vote.
            Your job is to select the SINGLE best answer among them.

            --- CANDIDATE RESPONSES (Tied) ---
            {candidates_joined}

            You MUST respond with a single JSON object with exactly these keys:
            - "winner": the NAME of the winning agent (string, one of {[r.name for r in tied_results]})
            - "reasoning": a concise explanation (string) for why you chose this winner.

            Example:
            {{
              "winner": "Engineer",
              "reasoning": "Engineer's answer is more technically correct and comprehensive."
            }}
        """).strip()

        return [
            {
                "role": "system",
                "content": "You are an expert, impartial software engineer and tie-breaker.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def _run_tie_breaker_with_progress(
        self,
        user_prompt: str,
        context: str,
        tied_results: List[AgentResult],
    ) -> Tuple[str, str]:
        """
        Run the tie-breaker model with a small progress bar.

        Returns:
          (winner_name, reasoning_text)
        """
        messages = self._build_tie_breaker_messages(user_prompt, context, tied_results)

        with tqdm(
            total=1,
            desc="Tie-breaker",
            unit="step",
            file=sys.stderr,
            leave=False,
        ) as pbar:
            resp = self.client.chat(JUDGE_MODEL, messages, json_mode=True)
            content = self.client.get_content(resp)
            pbar.update(1)

        try:
            parsed = json.loads(content)
            winner = str(parsed.get("winner", "")).strip()
            reasoning = str(parsed.get("reasoning", "")).strip()

            if not winner:
                raise ValueError(f"Tie-breaker did not provide 'winner'. Raw JSON: {parsed!r}")

            valid_names = {r.name for r in tied_results}
            if winner not in valid_names:
                raise ValueError(f"Tie-breaker chose invalid winner '{winner}'. Valid: {valid_names}")

            return winner, reasoning or "(no reasoning provided)"

        except Exception as e:
            # If judge JSON fails, fall back to "first tied"
            print(f"[!] Tie-breaker JSON failed: {e}", file=sys.stderr)
            fallback = tied_results[0].name
            return fallback, f"Fallback to first tied agent due to tie-breaker error: {e}"

    # ---------- Output Formatting ----------

    @staticmethod
    def _format_all_responses(results: List[AgentResult]) -> str:
        """
        Return a combined string of all agent responses.
        """
        parts = []
        for r in results:
            parts.append(f"--- {r.name} ---\n{r.response}\n")
        return "\n".join(parts).rstrip() + "\n"

    @staticmethod
    def _format_majority_winner(name: str, result: AgentResult) -> str:
        """
        Format output when a single winner is chosen by majority vote.
        """
        header = f"--- WINNER: {name} (Response by Majority Vote) ---\n"
        return header + "\n" + result.response + "\n"

    @staticmethod
    def _format_tie_winner(name: str, result: AgentResult, reasoning: str) -> str:
        """
        Format output when a tie-breaker was used.
        """
        header = f"--- TIE RESOLVED: {name} (Selected by Impartial Tie-Breaker) ---\n"
        body = result.response
        reasoning_block = "\n--- Tie-breaker reasoning ---\n" + reasoning
        return header + "\n" + body + reasoning_block + "\n"


# -----------------------------
# MAIN ENTRYPOINT (Thin CLI)
# -----------------------------

def main(argv: Optional[List[str]] = None) -> None:
    coordinator = CouncilCoordinator()
    coordinator.run_cli(argv)


if __name__ == "__main__":
    main()
