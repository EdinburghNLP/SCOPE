# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import importlib
import logging
import os
import sys
from enum import Enum

from omegaconf import OmegaConf

from verl.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__file__)


class SchemaOnlyTool:
    """A tool that only provides a schema — no executable backend.

    Used for terminal actions (e.g. submit_task, submit_answer) that
    appear in the model's tool list but signal conversation completion
    rather than requiring execution.
    """

    def __init__(self, tool_schema: OpenAIFunctionToolSchema):
        self.tool_schema = tool_schema
        self.name = tool_schema.function.name

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the tool schema in OpenAI format.

        Returns:
            OpenAIFunctionToolSchema for this tool.
        """
        return self.tool_schema
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ToolType(Enum):
    NATIVE = "native"
    MCP = "mcp"
    SCHEMA_ONLY = "schema_only"


async def initialize_mcp_tool(tool_cls, tool_config) -> list:
    from verl.tools.utils.mcp_clients.McpClientManager import ClientManager

    tool_list = []
    mcp_servers_config_path = tool_config.mcp.mcp_servers_config_path
    tool_selected_list = tool_config.mcp.tool_selected_list if "tool_selected_list" in tool_config.mcp else None
    await ClientManager.initialize(mcp_servers_config_path, tool_config.config.rate_limit)
    # Wait for MCP client to be ready
    max_retries = 10
    retry_interval = 2  # seconds
    for i in range(max_retries):
        tool_schemas = await ClientManager.fetch_tool_schemas(tool_selected_list)
        if tool_schemas:
            break
        if i < max_retries - 1:
            logger.debug(f"Waiting for MCP client to be ready, attempt {i + 1}/{max_retries}")
            await asyncio.sleep(retry_interval)
    else:
        raise RuntimeError("Failed to initialize MCP tools after maximum retries")
    # mcp registry
    assert len(tool_schemas), "mcp tool is empty"
    for tool_schema_dict in tool_schemas:
        logger.debug(f"tool_schema_dict: {tool_schema_dict}")
        tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
        tool = tool_cls(
            config=OmegaConf.to_container(tool_config.config, resolve=True),
            tool_schema=tool_schema,
        )
        tool_list.append(tool)
    return tool_list


def get_tool_class(cls_name):
    module_name, class_name = cls_name.rsplit(".", 1)
    if module_name not in sys.modules:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    else:
        module = sys.modules[module_name]

    tool_cls = getattr(module, class_name)
    return tool_cls


def initialize_tools_from_config(tools_config_file):
    """Initialize tools from a YAML configuration file.

    Tools with ``type: schema_only`` are returned as schema-only entries:
    their JSON schema is parsed but no backend class is instantiated.
    This allows tools like ``submit_task`` / ``submit_answer`` to appear
    in the model's tool list without requiring an executable backend.

    Args:
        tools_config_file: Path to the YAML tool configuration file.

    Returns:
        list[BaseTool | SchemaOnlyTool]: Initialized tool instances.
    """
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []
    for tool_config in tools_config.tools:
        tool_type = ToolType(tool_config.config.type)

        match tool_type:
            case ToolType.SCHEMA_ONLY:
                # Schema-only tools: parse the schema but skip class
                # instantiation.  These are terminal actions (e.g.
                # submit_task, submit_answer) that signal conversation
                # completion without needing execution.
                if tool_config.get("tool_schema", None) is None:
                    raise ValueError(
                        "schema_only tools must define a tool_schema"
                    )
                tool_schema_dict = OmegaConf.to_container(
                    tool_config.tool_schema, resolve=True
                )
                tool_schema = OpenAIFunctionToolSchema.model_validate(
                    tool_schema_dict
                )
                tool_list.append(SchemaOnlyTool(tool_schema=tool_schema))
            case ToolType.NATIVE:
                cls_name = tool_config.class_name
                tool_cls = get_tool_class(cls_name)
                if tool_config.get("tool_schema", None) is None:
                    tool_schema = None
                else:
                    tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                    tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
                tool = tool_cls(
                    config=OmegaConf.to_container(tool_config.config, resolve=True),
                    tool_schema=tool_schema,
                )
                tool_list.append(tool)
            case ToolType.MCP:
                cls_name = tool_config.class_name
                tool_cls = get_tool_class(cls_name)
                loop = asyncio.get_event_loop()
                mcp_tools = loop.run_until_complete(initialize_mcp_tool(tool_cls, tool_config))
                tool_list.extend(mcp_tools)
            case _:
                raise NotImplementedError
    return tool_list
