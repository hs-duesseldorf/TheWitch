import ollama

#init ollama client
client = ollama.Client()

#define function
def paraphrase_sentence(sentence: str):
    ourModel = "llama2"
    
    #add user sentence into prompt
    ourPrompt = (f"Give me 3 differently paraphrased versions of the following sentence: {sentence}") #f for formatted
    
    #send query
    ourResponse = client.generate(
        model=ourModel,
        prompt=ourPrompt
    )

    #return response text
    return ourResponse.response