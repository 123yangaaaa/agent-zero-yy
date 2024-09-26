from agent import Agent
from python.helpers.extension import Extension
from python.helpers.files import read_file
from python.helpers.vector_db import Area
import json
from python.helpers import errors, files
from python.tools.memory_tool import get_db


class RecallMemories(Extension):

    INTERVAL = 3
    HISTORY = 5
    RESULTS = 3
    THRESHOLD = 0.1

    async def execute(self, loop_data={}, **kwargs):

        iter = loop_data.get("iteration", 0)
        if (
            iter % RecallMemories.INTERVAL == 0
        ):  # every 3 iterations (or the first one) recall memories
            await self.search_memories(loop_data=loop_data, **kwargs)

    async def search_memories(self, loop_data={}, **kwargs):
        # try:
            # show temp info message
            self.agent.context.log.log(
                type="info", content="Searching memories...", temp=True
            )

            # show full util message, this will hide temp message immediately if turned on
            log_item = self.agent.context.log.log(
                type="util",
                heading="Searching memories...",
            )

            # get system message and chat history for util llm
            msgs_text = self.agent.concat_messages(
                self.agent.history[-RecallMemories.HISTORY :]
            )  # only last X messages
            system = self.agent.read_prompt(
                "memory.memories_query.sys.md", history=msgs_text
            )

            # log query streamed by LLM
            def log_callback(content):
                log_item.stream(query=content)

            # call util llm to summarize conversation
            query = await self.agent.call_utility_llm(
                system=system, msg=loop_data["message"], callback=log_callback
            )

            # get solutions database
            vdb = get_db(self.agent)

            memories = vdb.search_similarity_threshold(
                query=query,
                results=RecallMemories.RESULTS,
                threshold=RecallMemories.THRESHOLD,
                filter=f"area != '{Area.SOLUTIONS.value}'" # exclude solutions
            )

            # log the short result
            if not isinstance(memories, list) or len(memories) == 0:
                log_item.update(
                    heading="No useful memories found.",
                )
                return
            else:
                log_item.update(
                    heading=f"\n\n{len(memories)} memories found.",
                )

            # concatenate memory.page_content in memories:
            memories_text = ""
            for memory in memories:
                memories_text += memory.page_content + "\n\n"
            memories_text = memories_text.strip()

            # log the full results
            log_item.update(memories=memories_text)

            # place to prompt
            memories_prompt = self.agent.read_prompt(
                "agent.system.memories.md", memories=memories_text
            )

            # append to system message
            loop_data["system"] += memories_prompt

        # except Exception as e:
        #     err = errors.format_error(e)
        #     self.agent.context.log.log(
        #         type="error", heading="Recall memories extension error:", content=err
        #     )