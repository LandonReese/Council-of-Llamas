#!/usr/bin/env python3
"""
ollama_council.py

Simplified Multi-agent "council" wrapper for Ollama with optional filebase-aware RAG.
Refactored for two-stage deliberation (Respond -> Review -> Vote).

Default behavior:
  - NO RAG (no local filebase reads) unless explicitly enabled.

Usage examples:

  # Models only (no local filebase context)
  council "Explain nuclear fusion"

  # Use local filebase RAG context (SQLite + embeddings)
  council --context "Explain the architecture of this project"
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

from agents import AGENTS, JUDGE_MODEL
from rag_helper import get_context


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAX_WORKERS = 5
REQUEST_TIMEOUT = 600


def get_agent_response_prompt(agent_name: str, agent_sys_prompt: str) -> str:
    """
    Stage 1 System Prompt: Builds the system prompt for the initial, non-voting 
    response generation.
    """
    return textwrap.dedent(f"""
        You are the {agent_name} agent. Your core instructions are: "{agent_sys_prompt}"

        --- MANDATORY RESPONSE FORMATTING ---

        1. You MUST output ONLY a single, valid JSON object.
        2. DO NOT include ANY commentary, NO markdown fences (```json), and NO text before or after the JSON.
        3. The JSON MUST have exactly one key: "response", containing your detailed answer to the user's request.

        Example: {{"response": "My answer here."}}
    """).strip()


def get_agent_deliberation_prompt(agent_name: str, agent_sys_prompt: str, valid_votes: List[str]) -> str:
    """
    Stage 2 System Prompt: Builds the system prompt for the deliberation and voting phase.
    """
    valid_votes_list = ", ".join(sorted(valid_votes))

    return textwrap.dedent(f"""
        You are the {agent_name} agent. Your core instructions are: "{agent_sys_prompt}"

        Your task is to review the 'CANDIDATE RESPONSES' provided below, considering the 'USER REQUEST' and 'CONTEXT'.

        --- MANDATORY RESPONSE FORMATTING ---

        1. You MUST output ONLY a single, valid JSON object.
        2. DO NOT include ANY commentary, NO markdown fences (```json), and NO text before or after the JSON.
        3. The JSON MUST have exactly these two keys:
           - "reasoning": A brief explanation of why you chose your vote.
           - "vote": The NAME of the single agent you predict has the best answer.

        4. Your vote MUST be the **EXACT** name (case-sensitive) of ONE agent from this list:
           [{valid_votes_list}]
        5. You CANNOT vote for yourself.
        
        Example: {{"reasoning": "The explanation here.", "vote": "{valid_votes[0] if valid_votes else 'N/A'}"}}
    """).strip()


@dataclass
class AgentResult:
    """Holds a single agent's result."""
    name: str
    response: str
    vote: Optional[str]


class OllamaClient:
    """Thin wrapper around Ollama's /api/chat endpoint."""
    def __init__(self, base_url: str = OLLAMA_URL, timeout: int = REQUEST_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        json_mode: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
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
        try:
            return response["message"]["content"]
        except KeyError:
            raise ValueError(f"Unexpected response structure: {response!r}")


class CouncilCoordinator:
    """High-level orchestrator for the multi-agent council."""
    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        agents: Optional[List[Dict[str, Any]]] = None,
        max_workers: int = MAX_WORKERS,
    ) -> None:
        self.client = client or OllamaClient()
        self.agents = agents or AGENTS
        self.max_workers = min(max_workers, len(self.agents)) if self.agents else 1
        self.all_agent_names = [a["name"] for a in self.agents]
        self.agent_defs_by_name = {a['name']: a for a in self.agents}

    def _run_single_response(
        self,
        agent: Dict[str, Any],
        user_prompt: str,
        context: str,
    ) -> AgentResult:
        """
        Stage 1 Execution: Runs an agent to generate its response (no voting).
        """
        system_content = get_agent_response_prompt(
            agent["name"],
            agent["system_prompt"],
        )

        user_content = textwrap.dedent(f"""
            === FILEBASE CONTEXT START (Top 3 Chunks) ===
            {context}
            === FILEBASE CONTEXT END ===

            Your task is to answer the user's request using the context if it is helpful.
            If the context is irrelevant or explicitly states that RAG is disabled,
            rely on your own knowledge and persona instead.

            User Request: {user_prompt}
        """).strip()

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        try:
            resp = self.client.chat(agent["model"], messages)
            content = self.client.get_content(resp)
            parsed = json.loads(content)

            response_text = str(parsed.get("response", "")).strip()

            if not response_text:
                response_text = f"Error: {agent['name']} failed to generate a response."

            return AgentResult(
                name=agent["name"],
                response=response_text,
                vote=None,
            )

        except Exception as e:
            print(f"[!] {agent['name']} execution failed (JSON/Ollama error): {e}", file=sys.stderr)
            return AgentResult(
                name=agent["name"],
                response=f"Error during execution: {e}",
                vote=None,
            )

    def _run_response_generation(self, user_prompt: str, context: str) -> List[AgentResult]:
        """Orchestrates parallel execution of Stage 1."""
        print("[*] Stage 1: Running agents to generate responses...", file=sys.stderr)
        results: List[AgentResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_agent = {
                executor.submit(self._run_single_response, agent, user_prompt, context): agent
                for agent in self.agents
            }

            with tqdm(
                total=len(future_to_agent),
                desc="Responses",
                unit="agent",
                file=sys.stderr,
            ) as pbar:
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        print(f"[!] {agent['name']} agent crashed: {e}", file=sys.stderr)
                    finally:
                        pbar.update(1)
        return results

    def _run_single_deliberator(
        self,
        agent: Dict[str, Any],
        user_prompt: str,
        context: str,
        candidate_responses: str,
    ) -> Tuple[str, Optional[str]]:
        """
        Stage 2 Execution: Runs an agent to review candidate responses and cast a vote.
        Returns (agent_name, vote_target).
        """
        other_agent_names = [n for n in self.all_agent_names if n != agent["name"]]

        system_content = get_agent_deliberation_prompt(
            agent["name"],
            agent["system_prompt"],
            other_agent_names,
        )

        user_content = textwrap.dedent(f"""
            The original user request: "{user_prompt}"

            === FILEBASE CONTEXT START (Top 3 Chunks) ===
            {context}
            === FILEBASE CONTEXT END ===

            Review the candidate responses below and select the single best answer
            based on correctness, clarity, and completeness. If the context indicates
            RAG is disabled, rely on your own knowledge and the responses themselves.

            === CANDIDATE RESPONSES ===
            {candidate_responses}
        """).strip()

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        try:
            resp = self.client.chat(agent["model"], messages)
            content = self.client.get_content(resp)
            parsed = json.loads(content)

            vote = parsed.get("vote")
            reasoning = parsed.get("reasoning", "")

            if vote is None or vote not in other_agent_names:
                print(
                    f"[!] {agent['name']} produced an invalid vote: None / missing.\n"
                    f"    Reasoning: {reasoning or '(no response provided)'}\n"
                    f"    Parsed JSON: {parsed!r}",
                    file=sys.stderr,
                )
                return agent["name"], None

            return agent["name"], vote

        except Exception as e:
            print(f"[!] {agent['name']} deliberation failed (JSON/Ollama error): {e}", file=sys.stderr)
            return agent["name"], None

    def _run_deliberation_and_voting(
        self,
        user_prompt: str,
        context: str,
        initial_results: List[AgentResult]
    ) -> List[AgentResult]:
        """Orchestrates parallel execution of Stage 2 (Voting)."""
        print("[*] Stage 2: Agents deliberating and voting...", file=sys.stderr)

        candidate_responses_block = "\n\n" + ("\n\n" + "-" * 60 + "\n\n").join([
            f"--- Agent: {r.name} ---\n{r.response}" for r in initial_results
        ])

        deliberation_results = initial_results[:]
        result_map = {r.name: r for r in deliberation_results}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_agent = {
                executor.submit(
                    self._run_single_deliberator,
                    self.agent_defs_by_name[r.name],
                    user_prompt,
                    context,
                    candidate_responses_block
                ): r.name for r in initial_results
            }

            with tqdm(
                total=len(future_to_agent),
                desc="Deliberation",
                unit="agent",
                file=sys.stderr,
                leave=False,
            ) as pbar:
                for future in as_completed(future_to_agent):
                    agent_name = future_to_agent[future]
                    try:
                        _, vote = future.result()
                        result_map[agent_name].vote = vote
                    except Exception as e:
                        print(f"[!] {agent_name} deliberation crashed: {e}", file=sys.stderr)
                    finally:
                        pbar.update(1)

        return deliberation_results

    def _tally_votes(self, results: List[AgentResult]) -> Dict[str, int]:
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
        if not tally:
            return [], 0
        max_votes = max(tally.values())
        winners = [name for name, count in tally.items() if count == max_votes]
        return winners, max_votes

    def _run_tie_breaker(
        self,
        user_prompt: str,
        context: str,
        tied_results: List[AgentResult]
    ) -> Tuple[str, str]:
        candidates_block = []
        for r in tied_results:
            agent_def = self.agent_defs_by_name.get(r.name)
            role_context = agent_def['system_prompt'] if agent_def else "Role context unavailable."

            candidates_block.append(
                f"### Candidate: {r.name}\n"
                f"Role Context: {role_context}\n\n"
                f"Response:\n{r.response}"
            )

        candidates_joined = "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(candidates_block)

        user_content = textwrap.dedent(f"""
            The user asked: "{user_prompt}"

            Use the context and candidate responses below to select the SINGLE best answer.
            === CONTEXT ===
            {context}
            === CANDIDATES ===
            {candidates_joined}

            You MUST respond with a single JSON object: {{"winner": "NAME", "reasoning": "EXPLANATION"}}.
        """).strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert, impartial judge for technical answers. "
                    "You choose the best answer based on correctness, clarity, and completeness."
                )
            },
            {"role": "user", "content": user_content},
        ]

        try:
            resp = self.client.chat(JUDGE_MODEL, messages)
            content = self.client.get_content(resp)
            parsed = json.loads(content)
            winner = str(parsed.get("winner", "")).strip()
            reasoning = str(parsed.get("reasoning", "")).strip()

            valid_names = {r.name for r in tied_results}
            if winner not in valid_names:
                raise ValueError(f"Judge chose invalid winner '{winner}'.")

            return winner, reasoning or "(no reasoning provided)"

        except Exception as e:
            print(f"[!] Tie-breaker failed: {e}", file=sys.stderr)
            fallback = tied_results[0].name
            return fallback, f"Fallback to first tied agent due to error: {e}"

    def ask(self, user_prompt: str, use_rag: bool = False) -> str:
        """
        Orchestrates the two-stage process: Response Generation (Stage 1) and 
        Deliberation & Voting (Stage 2).

        If use_rag is True, we pull context from the local filebase via RAG.
        If use_rag is False, we do NOT touch the filebase and the models rely
        only on their own knowledge and personas.
        """
        if use_rag:
            context = self._get_context_with_progress(user_prompt)
        else:
            context = "No external filebase context. RAG is disabled for this question."

        initial_results = self._run_response_generation(user_prompt, context)

        if not initial_results:
            return "Error: No agent outputs produced in Stage 1."

        final_results = self._run_deliberation_and_voting(user_prompt, context, initial_results)

        print("[*] Tallying votes...", file=sys.stderr)
        tally = self._tally_votes(final_results)

        if not tally:
            print("[!] No valid votes were cast. Returning all responses.\n", file=sys.stderr)
            return self._format_all_responses(final_results)

        winners, max_votes = self._choose_winner_from_tally(tally)
        result_by_name = {r.name: r for r in final_results}

        if len(winners) == 1:
            winner_name = winners[0]
            winner_result = result_by_name[winner_name]
            print(f"[+] Winner by majority vote: {winner_name} ({max_votes} votes)", file=sys.stderr)
            return self._format_majority_winner(winner_name, winner_result)
        else:
            print(f"[*] Tie detected ({max_votes} votes each) for: {', '.join(winners)}", file=sys.stderr)
            print("[*] Running impartial tie-breaker...", file=sys.stderr)

            tied_results = [result_by_name[name] for name in winners]
            winner_name, reasoning = self._run_tie_breaker(user_prompt, context, tied_results)

            print(f"[+] Tie resolved. Winner: {winner_name}", file=sys.stderr)
            winning_result = result_by_name.get(winner_name, tied_results[0])
            return self._format_tie_winner(winner_name, winning_result, reasoning)

    def run_cli(self, argv: Optional[List[str]] = None) -> None:
        import argparse

        if argv is None:
            argv = sys.argv[1:]

        parser = argparse.ArgumentParser(
            prog="council",
            description="Multi-Agent Council (optionally using local filebase RAG)."
        )

        parser.add_argument(
            "-c", "--context",
            dest="use_rag",
            action="store_true",
            help=(
                "Use local filebase RAG context (SQLite + embeddings). "
                "If omitted, no local files will be read."
            ),
        )

        parser.add_argument(
            "prompt",
            nargs="+",
            help="Your main question for the council."
        )

        args = parser.parse_args(argv)
        user_prompt = " ".join(args.prompt).strip()

        answer = self.ask(user_prompt=user_prompt, use_rag=args.use_rag)
        print(answer)

    def _get_context_with_progress(self, user_prompt: str) -> str:
        print("[*] Retrieving filebase context via RAG...", file=sys.stderr)
        with tqdm(total=1, desc="RAG Context", unit="step", file=sys.stderr, leave=False) as pbar:
            context = get_context(user_prompt)
            pbar.update(1)
        return context

    @staticmethod
    def _format_all_responses(results: List[AgentResult]) -> str:
        parts = [f"--- {r.name} ---\n{r.response}\n" for r in results]
        return "\n".join(parts).rstrip() + "\n"

    @staticmethod
    def _format_majority_winner(name: str, result: AgentResult) -> str:
        header = f"--- WINNER: {name} (Majority Vote) ---\n"
        return header + "\n" + result.response + "\n"

    @staticmethod
    def _format_tie_winner(name: str, result: AgentResult, reasoning: str) -> str:
        header = f"--- TIE RESOLVED: {name} (Judge Selection) ---\n"
        body = result.response
        reasoning_block = "\n--- Judge Reasoning ---\n" + reasoning
        return header + "\n" + body + reasoning_block + "\n"


def main(argv: Optional[List[str]] = None) -> None:
    coordinator = CouncilCoordinator()
    coordinator.run_cli(argv)


if __name__ == "__main__":
    main()
