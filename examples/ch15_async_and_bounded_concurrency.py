# Chapter 15 teaching listing. Supply the inputs described in the text.
import asyncio
from sqm_ai.llm import MODELS, aclient, awith_retry, parsed_response, request_options
from sqm_ai.structured import SYSTEM, NCRClassification

async def classify_one(description):
    response = await awith_retry(
        aclient.messages.parse,
        model=MODELS["standard"],
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": description}],
        output_format=NCRClassification,
        **request_options(MODELS["standard"]),
    )
    return parsed_response(response)


async def classify_many(descriptions, limit=8):
    """Bound in-flight calls and scheduled task count; this is not a TPM limiter."""
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("positive concurrency limit required")
    results = []
    for start in range(0, len(descriptions), limit):
        results.extend(
            await asyncio.gather(
                *(
                    classify_one(d)
                    for d in descriptions[start : start + limit]
                ),
                return_exceptions=True,
            )
        )
    return results


results = asyncio.run(classify_many(batch))
failed = [r for r in results if isinstance(r, BaseException)]
