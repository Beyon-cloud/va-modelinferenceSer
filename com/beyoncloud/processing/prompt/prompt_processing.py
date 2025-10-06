from typing import List, Dict, Any
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from com.beyoncloud.schemas.rag_reqres_data_model import ChatHistory
from com.beyoncloud.processing.prompt.prompt_router import DynamicPromptRouter, run_performance_analysis

class PromptGenerator:

    def __init__(self):
        # Intentionally empty for now.
        # Reason: This class does not require instance state at construction
        # and will initialize attributes lazily when the analysis runs.
        # If future attributes are needed, initialize them here.
        pass

    def prompt_generator(self, query: str, domain_id: str, searchResults: List[Document], chat_history: List[ChatHistory] = None):
        
        # Build chat history string (if available)
        historyPrompt = []
        if chat_history:
            for chat in chat_history[-5:]:
                historyPrompt.append((chat.query, chat.response))


        # Combine context from search results
        fullContext =""
        if searchResults:
            fullContext = '"'
            for i, result in enumerate(searchResults, start=1):
                fullContext += f'{result['chunk']}\\n'
            fullContext = fullContext.rstrip('\\n') + '"'

        result = self._prompt_router(query, fullContext, domain_id, historyPrompt)

        generated_prompt = result.get("generated_prompt")
        print(f" prompt_generator --> generated_prompt --> {generated_prompt}")
        return generated_prompt

    def _prompt_router(self, query: str, text: str= "", domain_id: str="", historyPrompt: List=[]) -> Dict[str,Any] :
        """Comprehensive demonstration of the model-based system"""
        print("🧠 COMPLETE MODEL-BASED DYNAMIC PROMPT SYSTEM")
        print("=" * 80)
    
        # Initialize the complete system
        router = DynamicPromptRouter()
    
        # Generate optimal prompt using complete model ensemble
        result = router.generate_optimal_prompt(query, text, domain_id, historyPrompt)
        
        print(f"\n🎯 SELECTED PROMPT TYPE: {result['selected_prompt_type'].value.upper()}")
        print(f"🏆 ROUTING CONFIDENCE: {result['routing_confidence']:.3f}")
        
        print("\n📝 GENERATED PROMPT:")
        print("-" * 40)
        print(result["generated_prompt"])
        
        print("\n🧠 MODEL PREDICTIONS:")
        for model_name, prediction in result['model_predictions'].items():
            print(f"   • {model_name}: {prediction.model_name} - {prediction.confidence:.3f}")
        
        print("\n💡 ROUTING REASONING:")
        print(result['routing_reasoning'])
        
        if result.get('alternative_prompts'):
            print("\n🔄 ALTERNATIVE PROMPT TYPES:")
            for alt_type in result['alternative_prompts'].keys():
                print(f"   • {alt_type.upper()}")
        
        print("\n" + "="*80)

        run_performance_analysis()
    
        # Save trained models
        try:
            router.save_model_state("model_state.pkl")
            print("💾 Model state saved successfully!")
        except Exception as e:
            print(f"❌ Error saving model state: {e}")
    
        return result