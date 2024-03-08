import json
import logging
import os
import re
import sys

import folder_paths

from pprint import pprint

"""
The MIT License

Copyright (c) Jerry Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import openai

from llama_index.llms.openai import OpenAI
from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.readers.web import RssReader
from llama_index.readers.notion import NotionPageReader

from llama_index.core.indices.service_context import ServiceContext
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter, NodeParser
#from llama_index.core.base.embeddings.base import BaseEmbedding
#from llama_index.core.schema import Document, BaseNode, IndexGraph, LLM, BasePromptTemplate
from llama_index.core.indices.tree import TreeIndex

from llama_index.core.indices.struct_store import JSONQueryEngine as BaseJSONQueryEngine


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def valid_url(url):
        regex = re.compile(
            r'^(?:http|ftp)s?://'
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
            r'(?::\d+)?'
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return re.match(regex, url) is not None

# OpenAI

class OpenAIModel:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-1106",
                    "gpt-3.5-turbo-instruct",
                    "gpt-3.5-turbo-16k",
                    "gpt-3.5-turbo-0613",
                    "gpt-3.5-turbo-16k-0613",
                    "gpt-4-0125-preview",
                    "gpt-4-turbo-preview",
                    "gpt-4-1106-preview",
                    "gpt-4-vision-preview",
                    "gpt-4-1106-vision-preview",
                    "gpt-4",
                    "gpt-4-0613",
                    "gpt-4-32k",
                    "gpt-4-32k-0613"
                ],),
                "api_key": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("LLM_MODEL", )
    RETURN_NAMES = ("llm_model", )

    FUNCTION = "load_model"
    CATEGORY = "SALT/Llama-Index/Loaders"

    def load_model(self, model, api_key):
        openai.api_key = api_key
        llm_model = OpenAI(model=model, api_key=api_key)
        return (llm_model, )
    

class LLMComplete:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "The circumference of the Earth is"}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("completion", )

    FUNCTION = "complete"
    CATEGORY = "SALT/Llama-Index/Querying"

    def complete(self, llm_model, prompt):
        response = llm_model.complete(prompt)
        pprint(response, indent=4)
        return (response.text, )
    
class LLMChatMessages:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "prompt"}),
                "role": (["SYSTEM", "USER"],),
            },
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("llm_message", )

    FUNCTION = "prepare_messages"
    CATEGORY = "SALT/Llama-Index/Messages"

    def prepare_messages(self, prompt, role):
        messages = [
            ChatMessage(role=MessageRole.SYSTEM if role == "SYSTEM" else MessageRole.USER, content=prompt ),
        ]
        return (messages,)

class LLMChatMessagesAdv:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "You are a dog, you cannot speak, only woof, and react as a dog would."}),
                "user_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "plaeholder": "What is your name?"}),
            },
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("llm_message", )

    FUNCTION = "prepare_messages"
    CATEGORY = "SALT/Llama-Index/Messages"

    def prepare_messages(self, system_prompt, user_prompt):
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt ),
            ChatMessage(role=MessageRole.USER, content=user_prompt ),
        ]
        return (messages,)
    
class LLMChatMessageConcat:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "message_a": ("LIST", ),
                "message_b": ("LIST", ),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("llm_message", )

    FUNCTION = "concat_messages"
    CATEGORY = "SALT/Llama-Index/Messages"

    def concat_messages(self, message_a, message_b):
        return (message_a + message_b, )
    
class LLMChat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL", ),
                "message": ("LIST", ),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("response", )

    FUNCTION = "chat"
    CATEGORY = "SALT/Llama-Index/Querying"

    def chat(self, llm_model, message):
        response = llm_model.chat(message)
        pprint(response, indent=4)
        return (response.message.content, )

# Service Context    

class LLMServiceContextDefault:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
            },
        }

    RETURN_TYPES = ("LLM_CONTEXT",)
    RETURN_NAMES = ("llm_context",)

    FUNCTION = "context"
    CATEGORY = "SALT/Llama-Index/Context"

    def context(self, llm_model):
        service_context = ServiceContext.from_defaults(llm=llm_model)
        return (service_context,)

class LLMServiceContextAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
            },
            "optional": {
                "llm_embed_model": ("LLM_EMBED",),
                "llm_node_parser": ("LLM_NODE_PARSER",),
                "enable_chunk_overlap": ("BOOLEAN", {"default": True}),
                "chunk_overlap": ("INT", {"default": 50, "min": 0, "max": 100}),
                "enable_context_window": ("BOOLEAN", {"default": True}),
                "context_window": ("INT", {"default": 4096, "min": 2048, "max": 8192}),
                "enable_num_output": ("BOOLEAN", {"default": True}),
                "num_output": ("INT", {"default": 256, "min": 64, "max": 1024}),
                "enable_chunk_size_limit": ("BOOLEAN", {"default": True}),
                "chunk_size_limit": ("INT", {"default": 1024, "min": 512, "max": 2048}),
            },
        }

    RETURN_TYPES = ("LLM_CONTEXT",)
    RETURN_NAMES = ("llm_context",)

    FUNCTION = "context"
    CATEGORY = "SALT/Llama-Index/Context"

    def context(self, llm_model, llm_embed_model="default", llm_node_parser=None, enable_chunk_size=True, chunk_size=1024, 
                               enable_chunk_overlap=True, chunk_overlap=50, enable_context_window=True, context_window=4096, 
                               enable_num_output=True, num_output=256, enable_chunk_size_limit=True, chunk_size_limit=1024):
        prompt_helper = None
        if enable_context_window and enable_num_output:
            prompt_helper = PromptHelper(
                context_window=context_window if enable_context_window else None,
                num_output=num_output if enable_num_output else None,
                chunk_overlap_ratio=(chunk_overlap / 100.0) if enable_chunk_overlap else None,
                chunk_size_limit=chunk_size_limit if enable_chunk_size_limit else None,
            )

        service_context = ServiceContext.from_defaults(
            llm=llm_model,
            prompt_helper=prompt_helper,
            embed_model=llm_embed_model if llm_embed_model != "default" else None,
            node_parser=llm_node_parser,
            chunk_size=chunk_size if enable_chunk_size else None,
            chunk_overlap=chunk_overlap if enable_chunk_overlap else None,
        )
        return (service_context,)

# Index Store
    
class LLMVectorStoreIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_documents": ("LLM_DOCUMENTS",),
                "llm_context": ("LLM_CONTEXT",),
            },
        }

    RETURN_TYPES = ("LLM_INDEX",)
    RETURN_NAMES = ("llm_index",)

    FUNCTION = "index"
    CATEGORY = "SALT/Llama-Index/Indexing"

    def index(self, llm_documents, llm_context):
        index = VectorStoreIndex.from_documents(llm_documents, service_context=llm_context)
        return (index,)

class LLMSummaryIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_documents": ("LLM_DOCUMENTS",),
                "llm_context": ("LLM_CONTEXT",),
            },
        }

    RETURN_TYPES = ("LLM_INDEX",)
    RETURN_NAMES = ("llm_index",)

    FUNCTION = "index"
    CATEGORY = "SALT/Llama-Index/Indexing"

    def index(self, llm_documents, llm_context):
        index = SummaryIndex.from_documents(llm_documents, service_context=llm_context)
        return (index,)
    
class LLMTreeIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_nodes": ("LLM_NODES",),
                "service_context": ("LLM_CONTEXT",),
            },
            "optional": {
                "num_children": ("INT", {"default": 10}),
                "build_tree": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLM_INDEX",)
    RETURN_NAMES = ("llm_index",)

    FUNCTION = "index"
    CATEGORY = "SALT/Llama-Index/Indexing"

    def index(self, llm_nodes, service_context, num_children=10, build_tree=True):
        index = TreeIndex(
            nodes=llm_nodes,
            num_children=num_children,
            build_tree=build_tree,
            use_async=False,
            show_progress=True,
            service_context=service_context,
        )
        return (index,)
    
# Node Parser

class LLMSentenceSplitterNodeCreator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_documents": ("LLM_DOCUMENTS",),
            },
            "optional": {
                "chunk_size": ("INT", {"default": 1024, "min": 1}),
                "chunk_overlap": ("INT", {"default": 20, "min": 0}),
            },
        }

    RETURN_TYPES = ("LLM_NODES",)
    RETURN_NAMES = ("llm_nodes",)

    FUNCTION = "create_nodes"
    CATEGORY = "SALT/Llama-Index/Nodes"

    def create_nodes(self, llm_documents, chunk_size=1024, chunk_overlap=20):
        node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(llm_documents, show_progress=False)        
        return (nodes,)


# TODO
class LLMSemanticSplitterNodeParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_documents": ("LLM_DOCUMENTS",),
                "llm_embed_model": ("LLM_EMBED_MODEL",),
            },
            "optional": {
                "buffer_size": ("INT", {"default": 1, "min": 1}),
                "sentence_splitter": ("LLM_SENTENCE_SPLITTER",),
                "include_metadata": ("BOOLEAN", {"default": True}),
                "include_prev_next_rel": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLM_NODE_PARSER",)
    RETURN_NAMES = ("llm_node_parser",)

    FUNCTION = "semantic_nodes"
    CATEGORY = "SALT/Llama-Index/NodeParsing"

    def semantic_nodes(self, llm_documents, llm_embed_model, buffer_size=1, sentence_splitter=None, include_metadata=True, include_prev_next_rel=True):
        parser = SemanticSplitterNodeParser(
            embed_model=llm_embed_model,
            buffer_size=buffer_size,
            sentence_splitter=sentence_splitter,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
        )
        return (parser.build_semantic_nodes_from_documents(llm_documents, show_progress=True), )


# Query Engine
    
class LLMQueryEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_index": ("LLM_INDEX",),
            },
            "optional": {
                "query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Type your query here"}),
                "llm_message": ("LIST", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results",)

    FUNCTION = "query_engine"
    CATEGORY = "SALT/Llama-Index/Querying"

    def query_engine(self, llm_index, query=None, llm_message=None):
        query_components = []
        
        if llm_message and isinstance(llm_message, list):
            for msg in llm_message:
                if str(msg).strip():
                    query_components.append(str(msg))

        if query:
            if query.strip():
                query_components.append("user: " + query)

        pprint(query_components, indent=4)

        query_join = "\n".join(query_components)

        query_engine = llm_index.as_query_engine()
        response = query_engine.query(query_join)
        pprint(response, indent=4)
        return (response.response,)
    

class LLMQueryEngineAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_index": ("LLM_INDEX",),
            },
            "optional": {
                "query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Type your query here"}),
                "llm_message": ("LIST", {}),
                "top_k": ("INT", {"default": 10}),
                "similarity_cutoff": ("FLOAT", {"default": 0.7}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results",)

    FUNCTION = "query_engine"
    CATEGORY = "SALT/Llama-Index/Querying"

    def query_engine(self, llm_index, query=None, llm_message=None, top_k=10, similarity_cutoff=0.7):
        query_components = []
        
        if llm_message and isinstance(llm_message, list):
            for msg in llm_message:
                if str(msg).strip():
                    query_components.append(str(msg))

        if query and query.strip():
            query_components.append("user: " + query)

        pprint(query_components, indent=4)
        query_join = "\n".join(query_components)

        retriever = VectorIndexRetriever(index=llm_index, similarity_top_k=top_k)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
        )

        response = query_engine.query(query_join)
        pprint(response, indent=4)
        return (response.response,)

    
class LLMChatEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_index": ("LLM_INDEX",),
                "query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Ask a question"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "chat"
    CATEGORY = "Chat Engine"

    def chat(self, llm_index, query):
        chat_engine = llm_index.as_chat_engine()
        response = chat_engine.chat(query)
        pprint(response, indent=4)
        return (response.response,)
    

class LLMJSONQueryEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "json_data": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Enter JSON data here..."}),
                "json_schema": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Enter JSON schema here..."}),
                "json_query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Ener your JSON query / question here..."}),
                "output_mode": (["RAW", "Human Readable"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "json_path")

    FUNCTION = "query_engine"
    CATEGORY = "SALT/Llama-Index/Querying"

    def query_engine(self, llm_model, json_schema, json_data, json_query, output_mode):

        try:
            schema = json.loads(json_schema)
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            error_message = f"JSON parsing error: {str(e)}. Please ensure your JSON schema and data are correctly formatted."
            print(error_message)
            return (error_message, "")

        query_engine = BaseJSONQueryEngine(
            json_value = data,
            json_schema = schema,
            llm = llm_model,
            synthesize_response = True if output_mode == "Human Readable" else False,
        )

        response = query_engine.query(json_query)

        pprint(response, indent=4)

        return (response, response.metadata["json_path_response_str"])

# Readers
    
class LLMDirectoryReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_directory": ("STRING", {}),
            },
            "optional": {
                "recursive": ("BOOLEAN", {"default": False}),
                "required_ext_list": ("STRING", {"default": ".json, .txt, .html"}),
                "exclude_glob_list": ("STRING", {"default": ".sqlite, .zip"}),
            },
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "read_directory"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_directory(self, input_directory, recursive=False, required_ext_list=None, exclude_glob_list=None):
        full_path = os.path.join(folder_paths.get_input_directory(), input_directory.strip())

        input_dir = full_path if os.path.isdir(full_path) else None
        if not input_dir:
            raise ValueError("The provided subdirectory does not exist.")
        
        if required_ext_list.strip():
            required_exts = [ext.strip() for ext in required_ext_list.split(",") if ext.strip()]
        else:
            required_exts = None

        if exclude_glob_list.strip():
            exclude = [pattern.strip() for pattern in exclude_glob_list.split(",") if pattern.strip()]
        else:
            exclude = None

        print("Excluding: ", exclude)
        print("Required Extensions: ", required_exts)
        
        reader = SimpleDirectoryReader(
            input_dir=input_dir,
            exclude_hidden=True,
            recursive=recursive,
            required_exts=required_exts,
            exclude=exclude
        )

        documents = reader.load_data()
        if not documents:
            raise ValueError("No documents found in the specified directory.")

        return (documents,)
    
class LLMSimpleWebPageReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_1": ("STRING", {}),
            },
            "optional": {
                "url_2": ("STRING", {}),
                "url_3": ("STRING", {}),
                "url_4": ("STRING", {}),
                "html_to_text": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "read_web"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_web(self, url_1, url_2=None, url_3=None, url_4=None, html_to_text=True):
        if not url_1.strip():
            raise ValueError("At least one URL must be provided to LLMSimpleWebPageReader")
        
        urls = [url_1.strip()]
        if url_2.strip():
            urls.append(url_2.strip())
        if url_3.strip():
            urls.append(url_3.strip())
        if url_4.strip():
            urls.append(url_4.strip())
        
        urls = [url for url in urls if valid_url(url)]

        print("Valided URLs:", urls)

        documents = SimpleWebPageReader(html_to_text=html_to_text).load_data(urls)
        return (documents,)
    
class LLMTrafilaturaWebReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_1": ("STRING", {}),
            },
            "optional": {
                "url_2": ("STRING", {}),
                "url_3": ("STRING", {}),
                "url_4": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "read_web_trafilatura"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_web_trafilatura(self, url_1, url_2=None, url_3=None, url_4=None):
        if not url_1.strip():
            raise ValueError("At least one URL must be provided to LLMSimpleWebPageReader")
        
        urls = [url_1.strip()]
        if url_2.strip():
            urls.append(url_2.strip())
        if url_3.strip():
            urls.append(url_3.strip())
        if url_4.strip():
            urls.append(url_4.strip())
        
        urls = [url for url in urls if valid_url(url)]

        print("Valided URLs:", urls)

        documents = TrafilaturaWebReader().load_data(urls)
        return (documents,)
    
class LLMRssReaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_1": ("STRING", {}),
            },
            "optional": {
                "url_2": ("STRING", {}),
                "url_3": ("STRING", {}),
                "url_4": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "read_rss"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_rss(self, url_1, url_2=None, url_3=None, url_4=None):
        if not url_1.strip():
            raise ValueError("At least one URL must be provided to LLMSimpleWebPageReader")
        
        urls = [url_1.strip()]
        if url_2.strip():
            urls.append(url_2.strip())
        if url_3.strip():
            urls.append(url_3.strip())
        if url_4.strip():
            urls.append(url_4.strip())
        
        urls = [url for url in urls if valid_url(url)]

        print("Valided URLs:", urls)

        documents = RssReader().load_data(urls)
        return (documents,)
    
class LLMNotionReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "notion_integration_token": ("STRING", {}),
                "page_ids": ("STRING", {"multiline": False, "dynamicPrompts": False, "placeholder": "Page ID 1, Page ID 2"}),
                "database_id": ("STRING", {"multiline": False, "dynamicPrompts": False, "placeholder": "Database ID", "optional": True}),
            },
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "read_notion"
    CATEGORY = "SALT/Llama-Index/Readers"

    def read_notion(self, notion_integration_token, page_ids, database_id=None):

        page_id_list = None
        if page_ids:
            page_id_list = [page_id.strip() for page_id in page_ids.split(",") if page_id.strip()] if page_ids.strip() else None

        db_id = None
        if database_id:
            db_id = database_id.strip() if database_id.strip() else None
        
        if db_id:
            documents = NotionPageReader(integration_token=notion_integration_token).load_data(database_id=db_id)
        else:
            documents = NotionPageReader(integration_token=notion_integration_token).load_data(page_ids=page_id_list)
        
        return (documents,)
    
# Processing

class LLMPostProcessDocuments:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_documents": ("LLM_DOCUMENTS",),
            },
            "optional": {
                "required_keywords": ("STRING", {}),
                "exclude_keywords": ("STRING", {}),
            },
        }

    RETURN_TYPES = ("LLM_DOCUMENTS",)
    RETURN_NAMES = ("llm_documents",)

    FUNCTION = "process_documents"
    CATEGORY = "SALT/Llama-Index/Processing"

    def process_documents(self, llm_documents, required_keywords=[], exclude_keywords=[]):

        if required_keywords.strip():
            required = [ext.strip() for ext in required_keywords.split(",") if ext.strip()]
        else:
            required = None

        if exclude_keywords.strip():
            excluded = [pattern.strip() for pattern in exclude_keywords.split(",") if pattern.strip()]
        else:
            excluded = None

        if required or excluded:
            llm_documents = [doc for doc in llm_documents if not set(required).isdisjoint(doc.keywords) and set(excluded).isdisjoint(doc.keywords)]

        return (llm_documents,)

    
NODE_CLASS_MAPPINGS = {
    # Models
    "OpenAIModel": OpenAIModel,

    # Messages
    "LLMChatMessages": LLMChatMessages,
    "LLMChatMessagesAdv": LLMChatMessagesAdv,
    "LLMChatMessageConcat": LLMChatMessageConcat,

    # Service Context
    "LLMServiceContextDefault": LLMServiceContextDefault,
    "LLMServiceContextAdv": LLMServiceContextAdv,

    # Indexing
    "LLMVectorStoreIndex": LLMVectorStoreIndex,
    "LLMSummaryIndex": LLMSummaryIndex,
    "LLMTreeIndex": LLMTreeIndex,

    # Nodes
    "LLMSentenceSplitterNodeCreator": LLMSentenceSplitterNodeCreator,

    # Parser
    "LLMSemanticSplitterNodeParser": LLMSemanticSplitterNodeParser,

    # Querying / Chattings
    "LLMQueryEngine": LLMQueryEngine,
    "LLMQueryEngineAdv": LLMQueryEngineAdv,
    "LLMChatEngine": LLMChatEngine,
    "LLMChat": LLMChat,
    "LLMComplete": LLMComplete,
    # "LLMJSONQueryEngine": LLMJSONQueryEngine,

    # Readers 
    "LLMDirectoryReader": LLMDirectoryReader,
    "LLMSimpleWebPageReader": LLMSimpleWebPageReader,
    "LLMTrafilaturaWebReader": LLMTrafilaturaWebReader,
    "LLMRssReaderNode": LLMRssReaderNode,
    "LLMNotionReader": LLMNotionReader,

    # Post processing
    "LLMPostProcessDocuments": LLMPostProcessDocuments,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Models
    "OpenAIModel": "Load OpenAI LLM Model",

    # Messages
    "LLMChatMessages": "LLM Message",
    "LLMChatMessagesAdv": "LLM Message (Advanced)",
    "LLMChatMessageConcat": "LLM Messages Concat",

    # Service Context
    "LLMServiceContextDefault": "LLM Service Context",
    "LLMServiceContextAdv": "LLM Service Context (Advanced)",

    # Indexing
    "LLMVectorStoreIndex": "LLM Vector Store Index",
    "LLMSummaryIndex": "LLM Summary Index",
    "LLMTreeIndex": "LLM Tree Index",

    # Nodes
    "LLMSentenceSplitterNodeCreator": "LLM Setence Splitter Node Creator",

    # Parsers
    "LLMSemanticSplitterNodeParser": "LLM Semantics Splitter Node Parser",

    # Querying / Chattings
    "LLMQueryEngine": "LLM Query Engine",
    "LLMQueryEngineAdv": "LLM Query Engine (Advanced)",
    "LLMChatEngine": "LLM Chat Engine",
    "LLMChat": "LLM Chat",
    "LLMComplete": "LLM Complete",
    # "LLMJSONQueryEngine": "LLM JSON Query Engine",

    # Readers 
    "LLMDirectoryReader": "LLM Simple Directory Reader",
    "LLMSimpleWebPageReader": "LLM Simple Web Page Reader",
    "LLMTrafilaturaWebReader": "LLM Trafilatura Web Reader",
    "LLMRssReaderNode": "LLM RSS Reader",
    "LLMNotionReader": "LLM Notion Reader",

    # Post processing
    "LLMPostProcessDocuments": "LLM Post Process Documents",
}
