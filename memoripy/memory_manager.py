import numpy as np
import json
import time
import uuid
import ollama
from groq import Groq
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PIL import Image
import base64
import faster_whisper

from whisper.whispermodel import WhisperTranscriber

from .memory_store import MemoryStore

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional

class ConceptExtractionResponse(BaseModel):
    concepts: list[str] = Field(description="List of key concepts extracted from the text.")

class VisionCheckResponse(BaseModel):
    is_visual: bool = Field(description="Whether the request requires visual processing")
    reasoning: str = Field(description="Step by step reasoning for the decision")

class MemoryManager:
    """
    Manages the memory store, including loading and saving history,
    adding interactions, retrieving relevant interactions, and generating responses.
    """

    def __init__(self, api_key, chat_model="ollama",
                                chat_model_name="llama3.1:8b",
                                vision_model_name="llama-3.2-11b-vision-preview",
                                speech_model_name="whisper-large-v3",
                                embedding_model="ollama",
                                embedding_model_name="mxbai-embed-large",
                                storage=None):
        self.api_key = api_key
        self.chat_model = chat_model
        self.chat_model_name = chat_model_name
        self.vision_model_name = vision_model_name
        self.speech_model_name = speech_model_name
        self.embedding_model = embedding_model 
        self.embedding_model_name = embedding_model_name
        self.storage = storage

        self.in_memory = storage is None

        # Set chat model
        if chat_model.lower() == "openai":
            self.llm = ChatOpenAI(model=chat_model_name, api_key=self.api_key)
        elif chat_model.lower() == "ollama":
            self.llm = ChatOllama(model=chat_model_name, temperature=0)
        elif chat_model.lower() == "groq":
            self.llm = ChatGroq(model=chat_model_name, api_key=self.api_key)
        else:
            raise ValueError("Unsupported chat model. Choose either 'openai', 'ollama', or 'groq'.")

        # Set embedding model and dimension
        if embedding_model.lower() == "groq":
            if embedding_model_name == "mxbai-embed-large":
                self.dimension = 1024
            else:
                self.dimension = self.initialize_embedding_dimension()
            self.embeddings_model = lambda text: ollama.embeddings(model=self.embedding_model_name, prompt=text)["embedding"]
        
        elif embedding_model.lower() == "openai":
            if embedding_model_name == "text-embedding-3-small":
                self.dimension = 1536
            else:
                raise ValueError("Unsupported OpenAI embedding model name for specified dimension.")
            self.embeddings_model = OpenAIEmbeddings(model=embedding_model_name, api_key=self.api_key)
        elif embedding_model.lower() == "ollama":
            if embedding_model_name == "mxbai-embed-large":
                self.dimension = 1024
            else:
                self.dimension = self.initialize_embedding_dimension()
            self.embeddings_model = lambda text: ollama.embeddings(model=self.embedding_model_name, prompt=text)["embedding"]
        else:
            raise ValueError("Unsupported embedding model. Choose either 'openai', 'ollama', or 'groq'.")

        # Initialize memory store with the correct dimension
        self.memory_store = MemoryStore(dimension=self.dimension)

        if chat_model.lower() == "groq":
            self.vision_llm = ChatGroq(model=vision_model_name, api_key=self.api_key)
        else:
            self.vision_llm = None

        self.groq_client = Groq(api_key=api_key)

        self.initialize_memory()

    def initialize_embedding_dimension(self):
        """
        Retrieve embedding dimension from Ollama by generating a test embedding.
        """
        print("Determining embedding dimension for Ollama model...")
        test_text = "Test to determine embedding dimension"
        response = ollama.embeddings(
            model=self.embedding_model_name,
            prompt=test_text
        )
        embedding = response.get("embedding")
        if embedding is None:
            raise ValueError("Failed to retrieve embedding for dimension initialization.")
        return len(embedding)

    def initialize_groq_embedding_dimension(self):
        """
        Retrieve embedding dimension from Groq by generating a test embedding.
        """
        print("Determining embedding dimension for Groq model...")
        test_text = "Test to determine embedding dimension"
        embedding = self.embeddings_model.embed_text(test_text)
        if embedding is None:
            raise ValueError("Failed to retrieve embedding for dimension initialization.")
        return len(embedding)

    def standardize_embedding(self, embedding):
        """
        Standardize embedding to the target dimension by padding with zeros or truncating.
        """
        current_dim = len(embedding)
        if current_dim == self.dimension:
            return embedding
        elif current_dim < self.dimension:
            # Pad with zeros
            return np.pad(embedding, (0, self.dimension - current_dim), 'constant')
        else:
            # Truncate to match target dimension
            return embedding[:self.dimension]

    def load_history(self):
        """Load and normalize history to include core memory."""
        history = self.storage.load_history()
        if isinstance(history, tuple):
            if len(history) == 2:  # Old format compatibility
                short_term, long_term = history
                core = []
            else:  # New format with core memory
                short_term, long_term, core = history
        else:
            # Handle case where a dictionary or other format is returned
            short_term = history.get("short_term_memory", [])
            long_term = history.get("long_term_memory", [])
            core = history.get("core_memory", [])
        
        return short_term, long_term, core

    def save_memory_to_history(self):
        if not self.in_memory:
            self.storage.save_memory_to_history(self.memory_store)

    def add_interaction(self, prompt, output, embedding, concepts, is_core_memory=False):
        timestamp = time.time()
        interaction_id = str(uuid.uuid4())
        
        interaction = {
            "id": interaction_id,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": output}]}
            ],
            "embedding": embedding.tolist(),
            "timestamp": timestamp,
            "access_count": 1,
            "concepts": list(concepts),
            "decay_factor": 1.0,
        }
        
        if is_core_memory:
            self.memory_store.add_core_memory(interaction)
        else:
            self.memory_store.add_interaction(interaction)
            
        self.save_memory_to_history()

    def _parse_core_memory_with_llm(self, text: str) -> dict:
        """Use LLM to parse text into core memory schema using current memory state."""
        messages = [
            SystemMessage(content=f"""You are a core memory parser. Follow these steps:
            1. Analyze the user input carefully
            2. Check if it matches or updates any existing core memory keys: 
               {json.dumps(self.memory_store.core_memory, indent=2)}
            3. Explain what existing keys should be changed and why
            4. Return a JSON object with 'attribute_key', 'attribute_value', and 'description'"""),
            HumanMessage(content=f"User's specific request: {text}")
        ]

        response = self.llm.invoke(messages)
        print(f"Response from core memory parsing: {response.content}")
        
        attempts = 0
        max_attempts = 7
        while attempts < max_attempts:
            parsed = self._extract_last_json(response.content)
            if parsed and all(k in parsed for k in ["attribute_key", "attribute_value", "description"]):
                # Add embedding and concepts for the new core memory
                embedding = self.get_embedding(f"{parsed['attribute_key']} {parsed['attribute_value']}")
                concepts = self.extract_concepts(f"{parsed['attribute_key']} {parsed['attribute_value']}")
                
                parsed.update({
                    "embedding": embedding.tolist(),
                    "timestamp": time.time(),
                    "concepts": concepts
                })
                return parsed
            
            attempts += 1
            if attempts < max_attempts:
                print(f"Retry {attempts} - Invalid JSON format, retrying...")
                response = self.llm.invoke(messages)
                
        return None

    def store_core_memory(self, text: str, use_semantic: bool = False, semantic_threshold: float = 75.0) -> tuple[bool, str]:
        core_memory_triggers = [
            "remember that",
            "remember this forever",
            "this is core information",
            "store as core memory",
            "this is fundamental",
            "this is essential",
            "core memory",
            "vital information"
        ]
        text_lower = text.lower().strip()
        
        # More flexible matching
        is_exact_match = any(trigger.lower() in text_lower for trigger in core_memory_triggers)
        
        # Semantic match using MemoryStore
        is_semantic_match = False
        if use_semantic:
            trigger_text = " ".join(core_memory_triggers)
            trigger_embedding = self.get_embedding(trigger_text)
            trigger_concepts = self.extract_concepts(trigger_text)
            matches = self.memory_store.retrieve(trigger_embedding, trigger_concepts, similarity_threshold=semantic_threshold)
            is_semantic_match = len(matches) > 0

        if is_exact_match or is_semantic_match:
            # Clean the text by removing the trigger phrases
            cleaned_text = text
            for trigger in core_memory_triggers:
                cleaned_text = cleaned_text.lower().replace(trigger, "").strip()

            # Parse the cleaned text using LLM with current core memory context
            parsed_memory = self._parse_core_memory_with_llm(cleaned_text)
            
            if parsed_memory:
                self.memory_store.add_core_memory(parsed_memory)
                # Save to history immediately after adding core memory
                self.save_memory_to_history()
                #print(f"Stored and saved core memory: {parsed_memory}")
                return True, cleaned_text

            print("Failed to parse core memory structure")
            return False, text
            
        return False, text

    def get_embedding(self, text):
        print(f"Generating embedding for the provided text...")
        if callable(self.embeddings_model):  # If embeddings_model is a function, use it directly
            embedding = self.embeddings_model(text)
        else:  # OpenAI embeddings
            embedding = self.embeddings_model.embed_query(text)
        if embedding is None:
            raise ValueError("Failed to generate embedding.")
        if isinstance(embedding, list):
            embedding_array = np.array(embedding)
        else:
            embedding_array = embedding
        standardized_embedding = self.standardize_embedding(embedding_array)
        return standardized_embedding.reshape(1, -1)

    def _extract_last_json(self, response: str, verbose: bool = True) -> Optional[dict]:
        """Extract the last valid JSON object from a text that may contain both text and JSON."""
        brace_count = 0
        start_index = -1
        
        for i in range(len(response) - 1, -1, -1):
            if response[i] == '}':
                brace_count += 1
                if start_index == -1:
                    start_index = i
            elif response[i] == '{':
                brace_count -= 1
                if brace_count == 0 and start_index != -1:
                    json_str = response[i:start_index + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
        return None

    def extract_concepts(self, text):
        print("Extracting key concepts from the provided text...")

        if isinstance(self.llm, ChatOpenAI):
            messages = [
                SystemMessage(content="You are a precise concept extraction assistant. Analyze the text and return concepts in JSON format."),
                HumanMessage(content=(
                    "Extract key concepts from the text below. Think step by step about the most important concepts, "
                    "then return them in this JSON format: {\"concepts\": [\"concept1\", \"concept2\", ...]}.\n\n"
                    f"Text: {text}"
                ))
            ]
        elif isinstance(self.llm, (ChatOllama, ChatGroq)):
            messages = [
                SystemMessage(content="You are a precise concept extraction assistant. Analyze the text and return concepts in JSON format."),
                HumanMessage(content=(
                    "Extract key concepts from the text below. First, identify the core ideas. "
                    "Then format them as a JSON array using this exact format: {\"concepts\": [\"concept1\", \"concept2\", ...]}.\n\n"
                    f"Text: {text}"
                ))
            ]

        response = self.llm.invoke(messages)
        
        # Add retry logic for JSON extraction
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            json_data = self._extract_last_json(response.content)
            if json_data and "concepts" in json_data:
                return json_data["concepts"]
            attempts += 1
            if attempts < max_attempts:
                print(f"Retry {attempts} - Invalid JSON format, retrying...")
                response = self.llm.invoke(messages)
        
        print(f"Failed to extract concepts after {max_attempts} attempts")
        return []

    def get_predefined_memories(self):
        """Return predefined memories to initialize the system with."""
        predefined_messages = [
            # Assistant identity
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "If I don't say my name, call me User"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "My name is Assistant, but the user can change it to whatever they like."}]}
                ],
                "concepts": ["assistant name", "user name", "identity", "introduction"]
            },
            # Assistant role
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "You are a helpful assistant"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Thank you! I'm here to help you with anything you need."}]}
                ],
                "concepts": ["assistant role", "helpfulness", "persona"]
            }
        ]

        # Use the existing add_interaction mechanism to create properly structured memories
        for msg in predefined_messages:
            # Extract text from messages for embedding
            user_text = msg["messages"][0]["content"][0]["text"]
            assistant_text = msg["messages"][1]["content"][0]["text"]
            embedding = self.get_embedding(f"{user_text} {assistant_text}")
            self.add_interaction(user_text, assistant_text, embedding, msg["concepts"], force_long_term=True)

    def initialize_memory(self):
        """Initialize memory with history and core memory defaults."""
        short_term, long_term, core = self.load_history()
        
        # Initialize core memory with defaults if empty
        if not core:
            default_core = [
                {
                    "attribute_key": "user_name",
                    "attribute_value": "User",
                    "description": "Default user name"
                },
                {
                    "attribute_key": "assistant_name",
                    "attribute_value": "Assistant",
                    "description": "Default assistant name"
                },
                {
                    "attribute_key": "persona",
                    "attribute_value": "helpful and professional",
                    "description": "Default assistant persona"
                },
            ]
            for core_attr in default_core:
                self.memory_store.add_core_memory(core_attr)
        else:
            for core_attr in core:
                self.memory_store.add_core_memory(core_attr)

        # Initialize short-term memory
        for interaction in short_term:
            interaction['embedding'] = self.standardize_embedding(np.array(interaction['embedding']))
            self.memory_store.add_interaction(interaction)
        
        # Initialize long-term memory
        self.memory_store.long_term_memory.extend(long_term)
        
        self.memory_store.cluster_interactions()
        print(f"Memory initialized with {len(self.memory_store.short_term_memory)} interactions in short-term and {len(self.memory_store.long_term_memory)} in long-term.")

    def retrieve_relevant_interactions(self, query, similarity_threshold=40, exclude_last_n=0):
        query_embedding = self.get_embedding(query)
        query_concepts = self.extract_concepts(query)
        return self.memory_store.retrieve(query_embedding, query_concepts, similarity_threshold, exclude_last_n=exclude_last_n)

    def generate_response(self, prompt, last_interactions, retrievals, context_window=3):
        """Generate a response using the Groq model with proper message formatting."""
        # 1. Start with core memory as system context
        system_prompt = (
            f"You are {self.memory_store.core_memory['assistant_name']}, "
            f"with a {self.memory_store.core_memory['persona']} persona. "
            f"The user is known as {self.memory_store.core_memory['user_name']}. "
        )
        
        # Format messages specifically for Groq
        if isinstance(self.llm, ChatGroq):
            messages = []
            # Add system message
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
            # Add context from retrievals and interactions
            processed_retrievals = []
            for retrieval in retrievals:
                for msg in retrieval.messages:
                    # Convert complex content structure to simple string
                    if isinstance(msg.get("content"), list):
                        content = msg["content"][0].get("text", "")
                    else:
                        content = msg.get("content", "")
                    messages.append({
                        "role": msg["role"],
                        "content": content
                    })

            # Add recent interactions
            if last_interactions:
                context_interactions = last_interactions[-context_window:]
                for interaction in context_interactions:
                    print(f"""ALFNAKJ NADKJNSDVJHS: {interaction}""")
                    for msg in interaction["messages"]:
                        if isinstance(msg.get("content"), list):
                            content = msg["content"][0].get("text", "")
                        else:
                            content = msg.get("content", "")
                        messages.append({
                            "role": msg["role"],
                            "content": content
                        })

            # Add current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })

            # Use Groq client directly for chat completion
            try:
                completion = self.groq_client.chat.completions.create(
                    model=self.chat_model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=1024,
                    top_p=.5,
                    stream=False
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Error generating response: {e}")
                return "I apologize, but I encountered an error processing your request."
        
        else:
            # Original message handling for non-Groq models
            messages = [{"role": "system", "content": system_prompt}]
            # ...existing code for other model types...
            response = self.llm.invoke(messages)
            return response.content.strip()

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def check_visual_request(self, prompt: str) -> VisionCheckResponse:
        """Check if the user request requires visual processing."""
        if prompt=="":
            return VisionCheckResponse(is_visual=True, reasoning="Silent prompt")
        
        messages = [
            SystemMessage(content="""You are a request analyzer that determines if the user is asking for visual guidance.
            Consider that requests may come from visually impaired users needing assistance with:
            
            Daily Tasks:
            - Reading labels, expiration dates, or instructions
            - Identifying colors, patterns, or clothing matches
            - Checking if lights are on/off
            - Verifying cleanliness of surfaces
            
            Navigation & Safety:
            - Identifying obstacles or hazards
            - Reading signs or displays
            - Checking weather conditions
            - Confirming door/window states
            
            Social & Personal:
            - Describing people's expressions or gestures
            - Checking personal appearance
            - Reading handwritten notes
            - Identifying incoming mail or documents
            
            Technology:
            - Reading error messages on screens
            - Describing app interfaces
            - Checking display settings
            Output JSON using the following format
{
    "is_visual": true,  # Boolean indicating if request requires visual processing
    "reasoning": "String explaining why this is considered visual assistance."
}"""),
            HumanMessage(content=f"Analyze if this query requires visual processing: {prompt}")
        ]
        
        response = self.llm.invoke(messages)
        try:
            parsed = json.loads(response.content)
            return VisionCheckResponse(**parsed)
        except json.JSONDecodeError:
            # Fallback simple parsing if JSON is malformed
            is_visual = any(word in prompt.lower() for word in ['image', 'picture', 'photo', 'look at', 'see'])
            return VisionCheckResponse(is_visual=is_visual, reasoning="Simple keyword detection")

    def process_visual_request(self, prompt: str, image_path: str) -> str:
        """Process a visual request using the Groq vision model."""
        try:
            # Dynamically get all core memory values
            core_memory = self.memory_store.core_memory
            if prompt=="":
                prompt="User just wants a description"
            descriptive_prompt = f"""As {core_memory.get('assistant_name')}, I am assisting a user named {core_memory.get('user_name')}. 

User's specific request: {prompt}
"""
            
#             Focus on these aspects in your analysis:
# 1. Main subjects and their spatial relationships
# 2. Colors, textures, and lighting
# 3. Important details that might not be immediately obvious
# 4. Text or symbols if present
# 5. Context and setting
# 6. Any potential safety concerns or important warnings

# Return your response in the following JSON format:
# {{
# "description": "A detailed description of what you see in the image",
# "direct_answer": "A focused and concise answer to the user's specific question in their language of choice",
# }}
            print(f"Descriptive prompt: {descriptive_prompt}")
            completion = self.groq_client.chat.completions.create(
                model=self.vision_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": descriptive_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{self.encode_image_to_base64(image_path)}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2,
                max_tokens=1024,
                top_p=.5,
                stream=False
            )
            

            return completion.choices[0].message.content
            
            # Extract JSON from response
            attempts = 0
            max_attempts = 5
            while attempts < max_attempts:
                parsed_response = self._extract_last_json(completion.choices[0].message.content)
                print(f"Parsed response: {completion.choices[0].message.content}")
                if parsed_response and "direct_answer" in parsed_response:
                    return parsed_response["direct_answer"]
                
                attempts += 1
                if attempts < max_attempts:
                    print(f"Retry {attempts} - Invalid JSON format, retrying...")
                    completion = self.groq_client.chat.completions.create(
                        model=self.vision_model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": descriptive_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{self.encode_image_to_base64(image_path)}"
                                        }
                                    }
                                ]
                            }
                        ],
                        temperature=0.2,
                        max_tokens=1024,
                        top_p=.5,
                        stream=False
                    )
            
            return completion.choices[0].message.content

        except Exception as e:
            raise ValueError(f"Vision processing failed: {str(e)}")
