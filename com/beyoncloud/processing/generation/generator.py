import logging
import time
import com.beyoncloud.config.settings.env_config as config
from typing import List, Dict, Any
from com.beyoncloud.schemas.rag_reqres_data_model import (
    ChatHistory, RagReqDataModel, 
    StructureInputData, UserInput,
    RagLogQryModel, RagLogQryModelBuilder
)
from com.beyoncloud.processing.prompt.prompt_template import (
    get_prompt_template_old,get_prompt_param, 
    get_prompt_input_variables,
    SafeDict, get_temp_prompt_template, get_prompt_template
)
from com.beyoncloud.models.model_service import ModelServiceLoader
from com.beyoncloud.models import model_singleton
from com.beyoncloud.processing.prompt.prompt_processing import PromptGenerator
from com.beyoncloud.common.constants import CommonPatterns
from com.beyoncloud.processing.prompt.prompt_generation.huggingface_llama_connect import HuggingFaceLlama3Client
from com.beyoncloud.db.postgresql_impl import PostgresSqlImpl
from com.beyoncloud.db.postgresql_connectivity import PostgreSqlConnectivity
from com.beyoncloud.utils.date_utils import get_current_timestamp_string
from com.beyoncloud.config.settings.table_mapper_config import TableSettings

logger = logging.getLogger(__name__)

class RagGeneratorProcess:
    """
    A class responsible for generating LLM responses using LangChain's Conversational Retrieval Chain.

    This class builds contextual answers by combining retrieved documents, user queries, and chat history.
    It supports dynamic prompt templating based on domain and response type configurations.

    Author: Jenson
    Date: 16-June-2025
    """

    def __init__(self):
        """
        Initializes an instance of the RagGeneratorProcess.

        The class assumes that the language model (LLM) is preloaded into the `ModelRegistry.hf_llama_model_pipeline`.
        """
        self.prompt_generator = PromptGenerator()
        self.table_settings = TableSettings()

    async def generate_answer(self, rag_req_data_model: RagReqDataModel, search_results: List[Dict[str, Any]]):
        """
        Generates a response using LangChain's prompt templating and a loaded LLM model.

        This method combines the user query, top retrieved document chunks (`searchResults`), 
        and optional chat history to create a context-aware input for the model. 
        The prompt template and required input variables are fetched dynamically based on the 
        request's domain and response type.

        Args:
            ragReqDataModel (RagReqDataModel): Object containing metadata such as domain ID and response type.
            query (str): The current user question or query.
            searchResults (List[Document]): List of LangChain `Document` objects retrieved from the vector store.
            chat_history (List[ChatHistory], optional): List of past query-response pairs for conversational context.

        Returns:
            str: The generated answer from the LLM based on the provided inputs.
        
        Raises:
            ValueError: If the LLM model is not initialized in the `ModelRegistry`.
        """
        

        # Ensure LLM model is properly initialized
        all_model_objects = model_singleton.modelServiceLoader or ModelServiceLoader()
        llm = all_model_objects.get_hf_llama_model_pipeline()

        if llm is None:
            raise ValueError("generate_answer - LLM model not loaded. Please check ModelRegistry.hf_llama_model_pipeline")

        query = self.get_query(rag_req_data_model)
        logger.info(f"Inpur query --> {query}")
        
        # Build chat history string (if available)
        history_prompt = []
        chat_history: List[ChatHistory] = self.get_chat_history(rag_req_data_model)
        if chat_history:
            for chat in chat_history[-5:]:
                history_prompt.append((chat.query, chat.response))


        # Prompt Generation
        try:
            final_prompt = self.prompt_generator.prompt_generator(query,rag_req_data_model.domain_id, search_results, chat_history)
        
        except Exception as e:
            logger.error(f"❌ Error in demonstration: {e}")
            import traceback
            traceback.print_exc()

        response = await llm.ainvoke(final_prompt)

        print("\n Chat response:\n", response)
        logger.info(f"Chat response: --> {response}")

        return response

    async def hf_response(self, structure_input_data: StructureInputData):
        """
        Structured Response Generation
        """
        
        # Combine context from search results
        full_context = structure_input_data.context_data
        print(f"fullContext -----------> {full_context}")

        # Prompt Generation
        prompt_output = await get_prompt_template(
            structure_input_data.domain_id, structure_input_data.document_type, 
            structure_input_data.organization_id, structure_input_data.prompt_type,
            structure_input_data.output_format
        )
        param_result = prompt_output["prompt_param"]
        
        print(f"prompt_temp Output 123 -->{prompt_output["prompt_id"]} -  {prompt_output["system_prompt_template"]} - {prompt_output["prompt_param"]}")

        llama3_client = HuggingFaceLlama3Client(
            api_key=config.HF_KEY,
            model_name=config.HF_MODEL_NAME
        )

        # Test connection
        if not llama3_client.test_connection():
            raise ConnectionError("Failed to connect to Hugging Face Llama3 API")

        variable_map = {
            "context": full_context,
            "output_type": structure_input_data.output_format
        }
        if param_result:
            variable_map.update(param_result)

        system_prompt = prompt_output["system_prompt_template"].format_map(SafeDict(variable_map))
        user_prompt = prompt_output["user_prompt_template"].format_map(SafeDict(variable_map))

        response = llama3_client.generate_sync(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout=60,
                    temperature=0.1,
                    max_tokens=2048
                )

     
        print("\n Hugging face Response:\n", response)
        logger.info(f"Hugging face Response: --> {response}")

        return response

    async def generate_structured_response(self, structure_input_data: StructureInputData):
        """
        Structured Response Generation
        """
        
        # Ensure LLM model is properly initialized
        all_model_objects = model_singleton.modelServiceLoader or ModelServiceLoader()
        llm = all_model_objects.get_hf_llama_model_pipeline()

        if llm is None:
            raise ValueError("generate_structured_response - LLM model not loaded. Please check ModelRegistry.hf_llama_model_pipeline")


        # Combine context from search results
        full_context = structure_input_data.context_data

        # Prompt Generation
        prompt_output = await get_prompt_template(
            structure_input_data.domain_id, structure_input_data.document_type, 
            structure_input_data.organization_id, structure_input_data.prompt_type,
            structure_input_data.output_format
        )
        param_result = prompt_output["prompt_param"]

        final_prompt = prompt_output["prompt"]
        prompt_id = prompt_output["prompt_id"]
        input_variables = prompt_output["input_variables"]
        print(f"input_variables1 -->{prompt_id} -  {input_variables}")
        print(f"param_result --> {param_result}")

        variable_map = {
            "context": full_context,
            "output_type": structure_input_data.output_format
        }
        if param_result:
            variable_map.update(param_result)

        # Dynamically build the inputs dictionary
        inputs = {var: variable_map[var] for var in input_variables}
        print(f"Inputs : --> {inputs}")
        print(f"final_prompt --> {final_prompt}")

        # Build runnable sequence (prompt | llm)
        chain = final_prompt | llm
        # Invoke asynchronously
        response = await chain.ainvoke(
            inputs
        )

        print("\n Structured response:\n", response)
        logger.info(f"Structured response --> {response}")

        return response

    async def temp_generate_structured_response(self, structure_input_data: StructureInputData,
        system_prompt: str,
        user_prompt: str,
        str_input_variables: str
    ):
        """
        Structured Response Generation
        """
        
        # Ensure LLM model is properly initialized
        all_model_objects = model_singleton.modelServiceLoader or ModelServiceLoader()
        llm = all_model_objects.get_hf_llama_model_pipeline()

        if llm is None:
            raise ValueError("temp_generate_structured_response - LLM model not loaded. Please check ModelRegistry.hf_llama_model_pipeline")


        # Combine context from search results
        full_context = structure_input_data.context_data

        # Prompt Generation
        prompt_output = get_temp_prompt_template(system_prompt, user_prompt, str_input_variables)
        final_prompt = prompt_output["prompt"]
        input_variables = prompt_output["input_variables"]
        print(f"input_variables1 -->  {input_variables}")

        variable_map = {
            "context": full_context,
            "output_type": structure_input_data.output_format
        }

        # Dynamically build the inputs dictionary
        inputs = {var: variable_map[var] for var in input_variables}
        print(f"Inputs : --> {inputs}")
        print(f"final_prompt --> {final_prompt}")

        # Build runnable sequence (prompt | llm)
        chain = final_prompt | llm
        # Invoke asynchronously
        response = await chain.ainvoke(
            inputs
        )

        print("\n Response:\n", response)
        logger.info(f"Response --> {response}")

        return response

    async def temp_hf_response(self, structure_input_data: StructureInputData,
        system_prompt: str,
        user_prompt: str,
        str_input_variables: str
    ):
        """
        Structured Huggingface Response Generation
        """
        
        # Get context data
        full_context = structure_input_data.context_data
        print(f"temp_hf_response - fullContext -----------> {full_context}")

        # Prompt Generation
        prompt_output = get_temp_prompt_template(
            system_prompt, 
            user_prompt,
            str_input_variables
        )

        llama3_client = HuggingFaceLlama3Client(
            api_key=config.HF_KEY,
            model_name=config.HF_MODEL_NAME
        )

        # Test connection
        if not llama3_client.test_connection():
            raise ConnectionError("Failed to connect to Hugging Face Llama3 API")

        variable_map = {
            "context": full_context,
            "output_type": structure_input_data.output_format
        }

        final_system_prompt = prompt_output["system_prompt_template"].format_map(SafeDict(variable_map))
        final_user_prompt = prompt_output["user_prompt_template"].format_map(SafeDict(variable_map))

        starttime = get_current_timestamp_string()
        start_time = time.time()
        response = llama3_client.generate_sync(
                    system_prompt=final_system_prompt,
                    user_prompt=final_user_prompt,
                    timeout=60,
                    temperature=0.1,
                    max_tokens=2048
                )
        endtime = get_current_timestamp_string()
        end_time = time.time()
        elapsed = end_time - start_time  # in seconds (float)
        
        print("\n Temp hugging face response:\n", response)
        logger.info(f"Temp hugging face response: --> {response}")

        # Save response data into query log table
        try:
            metadata = {
                    "starttime": starttime,
                    "start_time": start_time,
                    "endtime": endtime,
                    "end_time": end_time,
                    "elapsed": elapsed,
                    "file_path": structure_input_data.source_path
            }

            rag_log_qry_model = (RagLogQryModelBuilder()
                .with_org_id(structure_input_data.organization_id)
                .with_query(CommonPatterns.EMPTY_SPACE)
                .with_response(response)
                .with_search_result_json([full_context])
                .with_time_elapsed(elapsed)
                .with_metadata(metadata)
                .build()
            )

            await self.save_model_response(rag_log_qry_model)
        except Exception as e:
            logger.error(f"Error processing RAG query log save : {e}")

        return response

    async def save_model_response(
        self, 
        rag_log_qry_model = RagLogQryModel
    ):
        """
        Saves RAG responses as log entry in the database.

        Args:
            ragReqDataModel (RagReqDataModel): The RAG request data.
            response : The RAG response string data.

        """
        
        pg_conn = PostgreSqlConnectivity()
        column_data = {
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "org_id"): rag_log_qry_model.orgId,
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "qry_txt"): rag_log_qry_model.query,
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "resp_txt"): rag_log_qry_model.response,
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "top_results"): rag_log_qry_model.search_result_json,
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "resp_time"): rag_log_qry_model.time_elapsed,
            self.table_settings.get_db_column_name("schema1", "RagQryLogs", "metadata"): rag_log_qry_model.metadata,

        }
        rag_qry_logs = pg_conn.base.classes[self.table_settings.get_db_table_name("schema1", "RagQryLogs")]
        qry_log = rag_qry_logs(**column_data)
        pg = PostgresSqlImpl()
        await pg.sqlalchemy_insert_one(qry_log, return_field = None)

        logger.info("RAG Query log data saved successfully.")

    def get_query(self, rag_req_data_model: RagReqDataModel):
        """
        Extracts the latest user query from the RAG request model.

        Args:
            ragReqDataModel (RagReqDataModel): The RAG input object containing user inputs.

        Returns:
            str: The first available textual query from the input.
        """

        query = ""
        user_inputs = rag_req_data_model.user_input
        print(user_inputs)
        if user_inputs:
            query_lst = [ui.content for ui in user_inputs if (ui.type == "text" or ui.type == "audio") and ui.content]
            query = query_lst[0]

        return query

    def get_chat_history(self, rag_req_data_model: RagReqDataModel) -> List[Dict[str, str]]:
        """
        Constructs the chat history as a list of query-response pairs for use in the prompt.

        Args:
            rag_req_data_model (RagReqDataModel): The RAG input containing past dialog history.

        Returns:
            List[ChatHistory]: List of ChatHistory objects representing prior exchanges.
        """

        dialog_details = rag_req_data_model.context.dialogDetails
        chat_history: List[ChatHistory] = []
        for dialog in dialog_details:
            user_inputs: List[UserInput] = dialog.user_input
            if user_inputs:
                query_lst = [ui.content for ui in user_inputs if (ui.type == "text" or ui.type == "audio") and ui.content]
                query_str = query_lst[0]

            chat_history.append(
                ChatHistory(query= query_str, response= dialog.response.text)
            )
        print(chat_history)
        return chat_history

