import memoripy.memory_manager as MemoryManager
import memoripy.json_storage as JSONStorage
import whisper.whispermodel as WhisperModel
import os
from colorama import Fore, Style, init
from dotenv import load_dotenv

class SeeProcessor:
    def __init__(self):
        init(autoreset=True)  # Initialize colorama

        load_dotenv()

        # Hard coded path if needed
        image_path = os.getenv('HARD_CODED_PATH')

        # Replace with your actual Groq API key
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("Please set your Groq API key.")

        # Define chat and embedding models for Groq
        chat_model = "groq"            # Use Groq for chat
        chat_model_name = "llama-3.1-70b-versatile" # Use largest parameter model
        
        # Add vision model configuration
        vision_model_name = "llama-3.2-90b-vision-preview"

        # Add speech model configuration
        speech_model_name = "whisper-large-v3"
        
        # Define embedding model for Groq
        embedding_model = "ollama"      # Choose 'openai' or 'ollama' for embeddings
        embedding_model_name = "mxbai-embed-large"  # Specific embedding model name

        # Choose your storage option
        storage_option = JSONStorage.JSONStorage("interaction_history.json")
        # Or use in memory storage by setting to None
        
        # Initialize the MemoryManager with the selected models and storage
        self.memory_manager = MemoryManager.MemoryManager(
            api_key=api_key,
            chat_model=chat_model,
            chat_model_name=chat_model_name,
            vision_model_name=vision_model_name,
            speech_model_name=speech_model_name,
            embedding_model=embedding_model,
            embedding_model_name=embedding_model_name,
            storage=storage_option,
        )

        self.whispermodel = WhisperModel.WhisperTranscriber()

        print("Welcome to the conversation! (Type 'exit' to end)")

    def generate(self, image_path, audio_path):
        prompt, language = self.whispermodel.transcribe_audio(audio_path)

        new_prompt = prompt #input(Fore.GREEN + "\nYou: " + Style.RESET_ALL).strip()

        # Check for core memory update first
        #print("\nCore memory before update:", memory_manager.memory_store.core_memory)
        is_core, processed_prompt = self.memory_manager.store_core_memory(new_prompt)
        print("Is core memory update:", is_core)
        #print("Core memory after update:", memory_manager.memory_store.core_memory)
        
        if is_core:
            # For core memories, just get and show the response
            response = self.memory_manager.generate_response(processed_prompt, [], [])
            print(Fore.CYAN + "\nAssistant: " + response + Style.RESET_ALL)
            return response

        # Check if request requires visual processing
        vision_check = self.memory_manager.check_visual_request(new_prompt)
        print(f"Vision check result: {vision_check.is_visual}")
        print(f"Reasoning: {vision_check.reasoning}")

        # Regular memory processing path
        # Load recent context
        history = self.memory_manager.load_history()
        short_term = history[0]
        last_interactions = short_term[-5:] if len(short_term) >= 5 else short_term

        # Get relevant past interactions
        relevant_interactions = self.memory_manager.retrieve_relevant_interactions(processed_prompt, exclude_last_n=5)

        if vision_check.is_visual:
            try:
                response = self.memory_manager.process_visual_request(new_prompt, image_path)
                print(Fore.CYAN + "\nAssistant (Vision): " + response + Style.RESET_ALL)
                
                # Format the vision interaction with proper JSON structure
                combined_text = f"Visual Request: {new_prompt}\nResponse: {response}"
                concepts = self.memory_manager.extract_concepts(combined_text)
                new_embedding = self.memory_manager.get_embedding(combined_text)
                
                self.memory_manager.add_interaction(
                    prompt=new_prompt,
                    output=response,
                    embedding=new_embedding,
                    concepts=concepts,
                    is_core_memory=False
                )
            except Exception as e:
                print(f"Vision processing failed: {e}")
                # Fall back to regular processing
        else:
            # Generate response
            response = self.memory_manager.generate_response(processed_prompt, last_interactions, relevant_interactions)
            print(Fore.CYAN + "\nAssistant: " + response + Style.RESET_ALL)

            # Process and store regular interaction
            combined_text = f"{processed_prompt} {response}"
            concepts = self.memory_manager.extract_concepts(combined_text)
            new_embedding = self.memory_manager.get_embedding(combined_text)
            
            self.memory_manager.add_interaction(
                processed_prompt, 
                response, 
                new_embedding, 
                concepts,
                is_core_memory=False
            )

        return response
