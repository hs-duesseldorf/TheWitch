import ollama
import TextGeneratePrompts as PT

'''
todo : 
translate all prompt to german
memory system for continuous text generation
apply RAG(Retrieval-Augmented Generation) for base knowledge and consistence chat
apply modelfile that making model that always remember the prompt
use larger LLM

'''

# init ollama client
client = ollama.Client()

# define function
def GenerateText(sentence: str):
    ourModel = "llama3.1"
    #llama2

    # add user sentence into prompt
    ourPrompt = (f""" 
        [System Instruction]
        {PT.prompt_Whip}
        {PT.prompt_Persona}
        {PT.prompt_RespondFormat}
    """)

    # [Current Palm Data]
    # - Heart: {PT.prompt_HeartLine}
    # - Head: {PT.prompt_HeadLine}
    # - Life: {PT.prompt_LifeLine}
    # - Fate: {PT.prompt_FateLine}

    # send query
    ourResponse = client.generate(
        model=ourModel,
        prompt=ourPrompt
    )

    PT.ourMemory = client.generate(
        model=ourModel,
        prompt=f"""
            Summarize the current status of the user's fortune-telling session. 
            You must include:
            1. The key characteristics of the palm lines discussed.
            2. The specific future predictions made in the last response.
            3. The overall mood and tone of the prophecy.
            
            Keep the summary under 150 tokens to save context space.
            Current Memory: {PT.ourMemory}
            New Response: {ourResponse.response}
            """
    )

    PT.chatLog = ourResponse

    # return response text
    return ourResponse.response

print("about HeartLine....")
print(GenerateText(PT.prompt_HeartLine))

print("about HeadLine....")
print(GenerateText(PT.prompt_HeadLine))

print("about LifeLine....")
print(GenerateText(PT.prompt_LifeLine))

print("about FateLine....")
print(GenerateText(PT.prompt_FateLine))