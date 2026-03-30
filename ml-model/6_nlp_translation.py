import pyttsx3
from transformers import pipeline
import warnings

# Suppress some of the verbose huggingface warnings
warnings.filterwarnings("ignore")

class ASLTranslator:
    def __init__(self):
        print("Loading NLP Model (this might take a minute on first run)...")
        # We use a T5 model fine-tuned specifically to correct grammar.
        # This is perfect for turning broken ASL gloss words into fluent English.
        self.nlp_model = pipeline(
            "text2text-generation", 
            model="vennify/t5-base-grammar-correction", 
            device=-1 # Set to 0 if you want to use GPU for this (requires VRAM)
        )
        
        # Initialize Text-to-Speech engine
        self.tts_engine = pyttsx3.init()
        # Optional: Slow down speaking rate slightly for clarity
        self.tts_engine.setProperty('rate', 150) 
        print("✓ NLP and TTS Systems Ready!")

    def translate_gloss_to_english(self, gloss_list):
        """
        Converts a list of ASL signs (glosses) into a fluent English sentence.
        Example: ["name", "I", "john"] -> "My name is John."
        """
        if not gloss_list:
            return ""
            
        # Join the signs into a raw string
        raw_text = " ".join(gloss_list)
        
        # T5 grammar models usually expect the prefix "grammar: "
        input_text = f"grammar: {raw_text}"
        
        try:
            # Generate the corrected fluent sentence
            result = self.nlp_model(input_text, max_length=50, num_beams=4, early_stopping=True)
            fluent_english = result[0]['generated_text']
            return fluent_english
        except Exception as e:
            print(f"Translation Error: {e}")
            return raw_text # Fallback to raw text if model fails

    def speak(self, text):
        """Speaks the text aloud."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()


# --- Quick Test ---
if __name__ == "__main__":
    translator = ASLTranslator()
    
    # Test Case 1
    asl_input_1 = ["I", "name", "John"]
    print(f"\nRaw ASL Input: {asl_input_1}")
    english_output = translator.translate_gloss_to_english(asl_input_1)
    print(f"Translated:    {english_output}")
    translator.speak(english_output)
    
    # Test Case 2
    asl_input_2 = ["store", "he", "go", "now"]
    print(f"\nRaw ASL Input: {asl_input_2}")
    english_output = translator.translate_gloss_to_english(asl_input_2)
    print(f"Translated:    {english_output}")
    translator.speak(english_output)