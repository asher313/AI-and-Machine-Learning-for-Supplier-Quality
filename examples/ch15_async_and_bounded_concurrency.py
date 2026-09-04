# Chapter 15 — 15.5 Async and Bounded Concurrency
import asyncio

from sqm_ai.llm import MODELS, aclient

# SYSTEM and NCRClassification: imports as in §15.3.


async def classify_one(description: str) -> NCRClassification:
    response = await aclient.messages.parse(
        model=MODELS["standard"],
        max_tokens=512,
        system=SYSTEM,
        messages=[{"role": "user", "content": description}],
        output_format=NCRClassification,
        output_config={"effort": "low"},
    )
    return response.parsed_output


async def classify_many(
    descriptions: list[str], limit: int = 8
) -> list[NCRClassification | BaseException]:
    """Run all calls; at most `limit` in flight at once."""
    gate = asyncio.Semaphore(limit)

    async def bounded(desc: str):
        async with gate:
            return await classify_one(desc)

    return await asyncio.gather(
        *(bounded(d) for d in descriptions),
        return_exceptions=True,
    )


results = asyncio.run(classify_many(batch))
failed = [r for r in results if isinstance(r, BaseException)]
