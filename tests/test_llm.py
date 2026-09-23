"""OpenRouter client against a mocked HTTP transport (no keys, no network)."""

import asyncio
import json
import unittest

try:
    import httpx

    from metabot.budget import Budget, BudgetExceeded
    from metabot.config import GEMINI_FLASH
    from metabot.llm import LlmError, OpenRouterClient
except ImportError:  # httpx not installed
    httpx = None


def ok_body(text="Probability: 30%", cost=0.0123):
    return {
        "model": "google/gemini-3.8-flash",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": cost},
    }


@unittest.skipIf(httpx is None, "httpx not installed")
class ClientTests(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def make(self, handler, **kw):
        return OpenRouterClient("test-key", transport=httpx.MockTransport(handler), **kw)

    def test_success_records_real_cost(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=ok_body())

        costs = []

        async def go():
            client = self.make(handler, on_cost=costs.append)
            budget = Budget(1.0)
            res = await client.complete(GEMINI_FLASH, "hi", budget=budget)
            await client.aclose()
            return res, budget

        res, budget = self.run_async(go())
        self.assertEqual(res.text, "Probability: 30%")
        self.assertAlmostEqual(budget.spent, 0.0123)
        self.assertAlmostEqual(budget.reserved, 0.0)
        self.assertEqual(costs, [0.0123])
        self.assertEqual(seen["body"]["reasoning"], {"effort": "high", "exclude": True})

    def test_retry_on_429(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, json={"error": {"code": 429, "message": "slow down"}})
            return httpx.Response(200, json=ok_body())

        async def go():
            client = self.make(handler)
            import metabot.llm as llm

            original = llm.random.uniform
            llm.random.uniform = lambda a, b: 0.0
            try:
                sleep = asyncio.sleep

                async def fast_sleep(_):
                    await sleep(0)

                llm.asyncio.sleep = fast_sleep
                return await client.complete(GEMINI_FLASH, "hi")
            finally:
                llm.random.uniform = original
                llm.asyncio.sleep = sleep
                await client.aclose()

        res = self.run_async(go())
        self.assertEqual(calls["n"], 2)
        self.assertTrue(res.text)

    def test_402_is_not_retried(self):
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(402, json={"error": {"code": 402, "message": "Insufficient credits"}})

        async def go():
            client = self.make(handler)
            try:
                await client.complete(GEMINI_FLASH, "hi")
            finally:
                await client.aclose()

        with self.assertRaises(LlmError) as ctx:
            self.run_async(go())
        self.assertEqual(ctx.exception.status, 402)
        self.assertEqual(calls["n"], 1)

    def test_budget_blocks_expensive_call(self):
        async def go():
            client = self.make(lambda r: httpx.Response(200, json=ok_body()))
            try:
                await client.complete(GEMINI_FLASH, "hi", budget=Budget(0.001))
            finally:
                await client.aclose()

        with self.assertRaises(BudgetExceeded):
            self.run_async(go())

    def test_empty_answer_with_length_is_an_error(self):
        body = ok_body(text="")
        body["choices"][0]["finish_reason"] = "length"

        async def go():
            client = self.make(lambda r: httpx.Response(200, json=body))
            try:
                await client.complete(GEMINI_FLASH, "hi")
            finally:
                await client.aclose()

        with self.assertRaises(LlmError):
            self.run_async(go())


if __name__ == "__main__":
    unittest.main()
