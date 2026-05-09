"""API-based inference utilities for Claude, GPT, Gemini, and DeepSeek models."""

import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


def run_anthropic_inference(
    data: list[dict],
    system_prompt: str,
    output_file: str,
    pred_key: str = "CLAUDEOPUS4.6_PRED",
    model: str = "claude-opus-4-6",
    api_key: Optional[str] = None,
    save_interval: int = 10,
    retry_wait: int = 300,
) -> None:
    """Run inference using the Anthropic API.

    Args:
        data: List of data entries to process. Modified in-place with predictions.
        system_prompt: System prompt for the model.
        output_file: Path to save results.
        pred_key: Key to store predictions under in each entry.
        model: Anthropic model identifier.
        api_key: API key. If None, uses ANTHROPIC_API_KEY env var.
        save_interval: Save results every N entries.
        retry_wait: Seconds to wait before retrying after an API error.
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    for i, entry in enumerate(data):
        if entry.get(pred_key) is not None:
            continue

        predicted_formula = ""
        for attempt in range(2):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=system_prompt,
                    output_config={"effort": "low"},
                    messages=[{"role": "user", "content": entry["NL"]}],
                    temperature=0.0,
                )
                predicted_formula = response.content[0].text.split("\ud835\udf19=")[-1]
                break
            except Exception as e:
                logger.warning("API error (attempt %d): %s", attempt + 1, e)
                if attempt == 0:
                    time.sleep(retry_wait)

        entry[pred_key] = predicted_formula.strip()

        if i % save_interval == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    logger.info("Inference complete. Results saved to %s", output_file)


def run_openai_inference(
    data: list[dict],
    system_prompt: str,
    output_file: str,
    pred_key: str = "GPT4o_PRED",
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    request_delay: float = 0.1,
) -> None:
    """Run inference using the OpenAI API.

    Args:
        data: List of data entries to process. Modified in-place with predictions.
        system_prompt: System prompt for the model.
        output_file: Path to save results.
        pred_key: Key to store predictions under in each entry.
        model: OpenAI model identifier.
        api_key: API key. If None, uses OPENAI_API_KEY env var.
        request_delay: Seconds to wait between requests.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    for i, entry in enumerate(data):
        try:
            response = client.responses.create(
                model=model,
                temperature=0.0,
                max_output_tokens=250,
                input=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": entry["NL"]},
                ],
            )
            predicted_formula = response.output_text.split("\ud835\udf19=")[-1]
            entry[pred_key] = predicted_formula.strip()
        except Exception as e:
            logger.warning("API error at entry %d: %s", i, e)
            time.sleep(10)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        time.sleep(request_delay)

    logger.info("OpenAI inference complete. Results saved to %s", output_file)


def run_deepseek_inference(
    data: list[dict],
    system_prompt: str,
    output_file: str,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    max_workers: int = 10,
    temperature: float = 1.3,
    max_retries: int = 3,
    entry_processor: Optional[callable] = None,
) -> list[dict]:
    """Run inference using the DeepSeek API with concurrent requests.

    Args:
        data: List of input data entries.
        system_prompt: System prompt for the model.
        output_file: Path to save results.
        api_key: DeepSeek API key.
        base_url: API base URL.
        model: Model identifier.
        max_workers: Number of parallel requests.
        temperature: Sampling temperature.
        max_retries: Number of retry attempts per request.
        entry_processor: Optional function(entry, data) -> dict for custom processing.

    Returns:
        List of result dictionaries.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def default_processor(entry):
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": entry["NL"]},
        ]
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                    temperature=temperature,
                )
                return {
                    "NL": entry["NL"],
                    "FOL_PRED": response.choices[0].message.content,
                    "FOL_GROUND-TRUTH": entry["FOL"],
                }
            except Exception as e:
                logger.warning("Request error (attempt %d/%d): %s", attempt + 1, max_retries, e)
                time.sleep(5**attempt)

        return {"NL": entry["NL"], "FOL_PRED": "ERROR", "FOL_GROUND-TRUTH": entry["FOL"]}

    results = []
    processor = entry_processor or default_processor

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if entry_processor:
            futures = [executor.submit(processor, entry, data) for entry in data]
        else:
            futures = [executor.submit(processor, entry) for entry in data]

        for future in tqdm(futures, total=len(data)):
            result = future.result()
            results.append(result)
            with open(output_file, "w", encoding="utf-8") as out_file:
                json.dump(results, out_file, ensure_ascii=False, indent=4)

    logger.info("DeepSeek inference complete. Results saved to %s", output_file)
    return results


def run_deepseek_reasoner_inference(
    data: list[dict],
    system_prompt: str,
    output_file: str,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    max_workers: int = 20,
    max_retries: int = 5,
    base_backoff: int = 4,
    fallback_system_prompt: Optional[str] = None,
) -> list[dict]:
    """Run inference using DeepSeek Reasoner with fallback to chat model.

    Args:
        data: List of input data entries.
        system_prompt: System prompt for the reasoner model.
        output_file: Path to save results.
        api_key: DeepSeek API key.
        base_url: API base URL.
        max_workers: Number of parallel requests.
        max_retries: Number of retry attempts.
        base_backoff: Base backoff time in seconds.
        fallback_system_prompt: System prompt for fallback. If None, uses system_prompt.

    Returns:
        List of result dictionaries.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

    def call_with_retries(client, model_name, messages, max_tokens=None, temperature=None):
        for attempt in range(max_retries):
            try:
                return client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except (APIConnectionError, RateLimitError, APIStatusError, TimeoutError) as e:
                wait_time = base_backoff * (2**attempt)
                logger.warning("Retry %d/%d - %s. Waiting %ds", attempt + 1, max_retries, e, wait_time)
                time.sleep(wait_time)
            except Exception as e:
                wait_time = base_backoff * (2**attempt)
                logger.warning("Retry %d/%d - unexpected: %s. Waiting %ds", attempt + 1, max_retries, e, wait_time)
                time.sleep(wait_time)
        raise RuntimeError(f"Failed after {max_retries} retries")

    def process_entry(entry):
        client = OpenAI(api_key=api_key, base_url=base_url)
        reasoning = ""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": entry["NL"]},
            ]
            response = call_with_retries(client, "deepseek-reasoner", messages, max_tokens=2500)
            message = response.choices[0].message
            if message.content == "":
                reasoning = message.reasoning_content
                raise RuntimeError("Empty response from reasoner")
            return {
                "NL": entry["NL"],
                "FOL_PRED": message.content,
                "FOL_PRED_REASONING": message.reasoning_content,
                "FOL_GROUND-TRUTH": entry["FOL"],
            }
        except Exception as e:
            logger.warning("Reasoner failed for: %s - %s", entry["NL"][:50], e)
            fb_prompt = fallback_system_prompt or system_prompt
            if reasoning:
                fb_prompt += f" Use your earlier thoughts on this: {reasoning}"
            messages = [
                {"role": "system", "content": fb_prompt},
                {"role": "user", "content": entry["NL"]},
            ]
            fallback_response = call_with_retries(client, "deepseek-chat", messages, temperature=1.3)
            return {
                "NL": entry["NL"],
                "FOL_PRED": fallback_response.choices[0].message.content,
                "FOL_PRED_REASONING": reasoning,
                "FOL_GROUND-TRUTH": entry["FOL"],
                "FALLBACK_USED": True,
            }

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_entry, entry): entry for entry in data}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
                with open(output_file, "w", encoding="utf-8") as out_file:
                    json.dump(results, out_file, ensure_ascii=False, indent=4)
            except Exception as e:
                logger.error("Processing failed: %s", e)

    logger.info("DeepSeek reasoner inference complete. Results saved to %s", output_file)
    return results
