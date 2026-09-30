"""CPU-only checks for the concurrent semantic replay client."""

from __future__ import annotations

import asyncio
import json
import unittest

import httpx

from replay_workflows import parse_metrics, send_request
from profile_spec_decode import TraceRequest


class ReplayClientTests(unittest.TestCase):
    def test_metrics_parser(self) -> None:
        body = (
            'vllm:num_requests_running{model_name="m"} 2\n'
            'vllm:coreflow_scheduled_requests_per_step_count{model_name="m"} 12\n'
            'vllm:coreflow_scheduled_requests_per_step_sum{model_name="m"} 30\n'
            'vllm:other_metric 99\n'
        )
        parsed = parse_metrics(body)
        self.assertEqual(parsed["vllm:num_requests_running"], 2)
        self.assertEqual(parsed["vllm:coreflow_scheduled_requests_per_step_sum"], 30)
        self.assertNotIn("vllm:other_metric", parsed)

    def test_request_uses_saved_token_ids_and_stream_usage(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            events = (
                'data: {"choices":[{"text":"ok","finish_reason":null}]}\n\n'
                'data: {"choices":[{"text":"","finish_reason":"stop"}]}\n\n'
                'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
                'data: [DONE]\n\n'
            )
            return httpx.Response(200, text=events)

        request = TraceRequest(
            query_id="q", source_file="test_q.log", logged_task=0,
            agent_name="user_agent", repeat=0, messages=[],
            original_prompt_tokens=None, original_completion_tokens=None,
            prompt_token_ids=[11, 12, 13],
        )

        async def check() -> dict:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                return await send_request(
                    client, request, "http://local", "target", 32, 2026
                )

        row = asyncio.run(check())
        self.assertEqual(captured["prompt"], [11, 12, 13])
        self.assertFalse(captured["add_special_tokens"])
        self.assertEqual(row["agent"], "Plan")
        self.assertEqual(row["request_id"], "q:user_agent:0:0")
        self.assertEqual(row["output_tokens"], 1)
        self.assertEqual(row["finish_reason"], "stop")
        self.assertTrue(row["success"])


if __name__ == "__main__":
    unittest.main()
