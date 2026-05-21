import ollama

#init ollama client
client = ollama.Client()

#define function
def paraphrase_sentence(sentence: str):
    ourModel = "llama3.2"
    
    #add user sentence into prompt
    ourPrompt = (f"다음 사주를 가지고 있는 사람은 어떤 사람이고 어떤 미래를 가지고 있을지 이야기해줘: {sentence}") #f for formatted
    
    #send query
    ourResponse = client.generate(
        model=ourModel,
        prompt=ourPrompt
    )

    #return response text
    return ourResponse.response