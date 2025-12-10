import logging

import runpod

from engine import OpenAIvLLMEngine, vLLMEngine
from utils import JobInput

vllm_engine = None
openai_engine = None


async def handler(job):
    logging.info(f"Handler received job: {job}")
    job_input = JobInput(job["input"])
    engine = openai_engine if job_input.openai_route else vllm_engine
    if engine is None:
        logging.error("Engine is None!")
        return {"error": "Engine not initialized"}

    results_generator = engine.generate(job_input)

    if job_input.stream:
        return results_generator
    else:
        results = []
        async for batch in results_generator:
            results.append(batch)
        return results


if __name__ == "__main__":
    vllm_engine = vLLMEngine()
    openai_engine = OpenAIvLLMEngine(vllm_engine)

    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        }
    )
