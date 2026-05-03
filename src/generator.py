"""
generator.py — Rationale Generation Module

Generates human-readable explanations for retrieved BIS standards.
Uses a single batched Groq API call (not per-standard) to stay within rate limits.
Falls back to chunk-derived text if API is unavailable — never shows generic messages.
"""
import re
import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from groq import Groq

logger = logging.getLogger("BIS-RAG")


class RationaleGenerator:
    def __init__(self, api_key=None):
        self.llm_client = self._init_groq(api_key) if api_key else None
        if not self.llm_client:
            logger.warning("GROQ_API_KEY not set or invalid — using chunk-text rationale fallback.")

    def _init_groq(self, api_key):
        """
        Initialize and health-check the Groq client.
        Returns None if the key is invalid or network is unavailable,
        so all downstream steps use the chunk-text fallback automatically.
        """
        try:
            client = Groq(api_key=api_key)
            client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            logger.info("Groq API key verified — LLM rationale enabled.")
            return client
        except Exception as e:
            logger.warning(f"Groq API check failed ({e}) — switching to chunk-text fallback.")
            return None

    def expand_query(self, query_text):
        """
        Rewrite user query into BIS technical terminology using LLM.
        Hard 3-second timeout — never blocks the pipeline.
        Returns original query if expansion fails or times out.
        """
        if self.llm_client is None:
            return query_text

        prompt = (
            "Rewrite the following user query into a short technical search query "
            "(max 15 words). Use BIS/IS standard terminology only. "
            "Output ONLY the rewritten query, nothing else.\n\n"
            f"Query: {query_text}\nRewritten:"
        )

        def _call():
            return self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=40,
                temperature=0.0,
            )

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                response = future.result(timeout=3.0)
                expanded = response.choices[0].message.content.strip().split('\n')[0]
                if 5 < len(expanded) < 200:
                    logger.info(f"Query expanded: '{expanded}'")
                    return expanded
        except FuturesTimeoutError:
            logger.warning("Query expansion timed out (>3s) — using original query.")
        except Exception as e:
            logger.warning(f"Query expansion skipped: {e}")

        return query_text

    def generate(self, query_text, standards, top_texts):
        """
        Generate rationale for each retrieved standard.

        Single batched API call for all standards — avoids rate limits.
        Falls back to first sentence of BIS chunk text when API is unavailable.

        Args:
            query_text: user's original query
            standards:  list of IS standard labels (e.g. ["IS 269 : 1989", ...])
            top_texts:  corresponding chunk texts from the dataset

        Returns:
            dict mapping standard label → rationale string
        """
        # Chunk-text fallback — always meaningful, never generic
        fallback = {}
        for std, txt in zip(standards, top_texts):
            first_sentence = txt.split('.')[0].strip()[:120] if txt else ""
            fallback[std] = first_sentence or "Relevant BIS standard for the specified product."

        if self.llm_client is None or not standards:
            return fallback

        context_blocks = "\n\n".join(
            f"[{std}]: {txt[:300]}" for std, txt in zip(standards, top_texts)
        )
        standards_list = "\n".join(f"- {s}" for s in standards)

        prompt = (
            "You are a BIS compliance expert helping Indian MSEs.\n\n"
            f"Product/Query: {query_text}\n\n"
            f"Standard summaries from BIS SP 21:\n{context_blocks}\n\n"
            f"For each of these standards:\n{standards_list}\n\n"
            "Write ONE sentence (max 20 words) explaining WHY it is relevant to the product above. "
            "Ground your answer strictly in the summary text provided. "
            "Return ONLY valid JSON. No markdown. No extra text.\n"
            'Format: {"IS XXX : YYYY": "reason", ...}'
        )

        try:
            response = self.llm_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            result = json.loads(raw)
            # Fill any missing keys with fallback
            for std in standards:
                if std not in result:
                    result[std] = fallback[std]
            return result
        except Exception as e:
            logger.warning(f"Rationale generation failed ({e}) — using chunk-text fallback.")
            return fallback
