"""FutureEval ensemble forecasting bot.

Pure modules (config, budget, qctx, cdf, aggregation, parsing, prompts, state)
only need numpy and the standard library. llm/research need httpx and the
asknews SDK; bot/runner need forecasting-tools. Nothing heavy is imported here.
"""

__version__ = "1.0.0"
