import os
import asyncio
from python.helpers import memory, zhipu_search, duckduckgo_search
from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle
from python.helpers.errors import handle_error

class Knowledge(Tool):
    async def execute(self, question="", **kwargs):
        # Create tasks for all three search methods
        tasks = [
            self.zhipu_search(question),
            self.duckduckgo_search(question),
            self.mem_search(question)
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        zhipu_result, duckduckgo_result, memory_result = results

        # Handle exceptions and format results
        zhipu_result = self.format_result(zhipu_result, "Zhipu Search")
        duckduckgo_result = self.format_result(duckduckgo_result, "DuckDuckGo")
        memory_result = self.format_result(memory_result, "Memory")

        msg = self.agent.read_prompt("tool.knowledge.response.md", 
                              online_sources = ((zhipu_result + "\n\n") if zhipu_result else "") + str(duckduckgo_result),
                              memory = memory_result)

        await self.agent.handle_intervention(msg)  # wait for intervention and handle it, if paused

        return Response(message=msg, break_loop=False)

    async def zhipu_search(self, question):
        if os.getenv("ZHIPU"):
            return await asyncio.to_thread(zhipu_search.zhipu_search, question)
        else:
            PrintStyle.hint("No API key provided for Zhipu. Skipping Zhipu search.")
            self.agent.context.log.log(type="hint", content="No API key provided for Zhipu. Skipping Zhipu search.")
            return None

    async def duckduckgo_search(self, question):
        return await asyncio.to_thread(duckduckgo_search.search, question)

    async def mem_search(self, question: str):
        db = await memory.Memory.get(self.agent)
        docs = await db.search_similarity_threshold(query=question, limit=5, threshold=0.5)
        text = memory.Memory.format_docs_plain(docs)
        return "\n\n".join(text)

    def format_result(self, result, source):
        if isinstance(result, Exception):
            handle_error(result)
            return f"{source} search failed: {str(result)}"
        return result if result else ""