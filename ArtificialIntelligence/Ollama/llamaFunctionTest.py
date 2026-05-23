import ollama

#init ollama client
client = ollama.Client()

#define function
def paraphrase_sentence(sentence: str):
    ourModel = "llama3.2"
    
    #add user sentence into prompt
    ourPrompt = (f"you are the the witch that make prophecy. pharaprasing this senetence only speak korean.: {sentence}") #f for formatted
    
    #send query
    ourResponse = client.generate(
        model=ourModel,
        prompt=ourPrompt
    )

    #return response text
    return ourResponse.response

if __name__ == "__main__":
    print(paraphrase_sentence("""내가 마음껏 자라나고 활동할 수 있는 넓은 땅이 눈앞에 펼쳐진 형국이구나. 
    현실을냉철하게 파악하는 눈과 돈의 흐름을 읽는 감각이 남들보다 발달했어. 부지런히 움직여 기회를 포착하고, 내가 노력한 만큼 
확실한 결실과 재물을 쥐려는 생활력이 아주 강한 팔자구나.다만, 끊임없이 일거리가 보이고 바쁘게 움직이다 보니 늘 마음이 분주할 수 있겠어. 
너무 욕심을 내어 몸을 혹사하기보다, 내 체력이 감당할 수 있는선에서 실속을 챙기는 지혜가 필요하겠군"""))
