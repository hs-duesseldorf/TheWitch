# Ollama

## Base Requirements

- Ollama and Python need to be installed
- matching Ollama models need to be locally installed
```bash
pip install ollama fastapi uvicorn pydantic
```

## Explanation

`llamaFunctionTest.py` shows how we can use Ollama to define simple methods, inside of the paraphrase_sentence method you can see a variable called ourModel. If one doesnt have the required model locally installed the method wont work. For now this method uses the model "llama2", which by default is the 7b version and will take roughly 4gb of space to download.
To download and test the model execute the following command:
```bash
ollama run llama2
```
Note that after downloading the model it will automatically open a chat where you can test it in cmd, you can end the chat by simply writing "/bye".

`fastapiServerTest.py` is a simple webserver that imports and calls the paraphrase_sentence method whenever the user presses the button on the site.
To start the server simply run the following command from inside this folder:
```bash
python -m uvicorn fastapiServerTest:app --reload
```
Note that if you are working on the fastapiServerTest.py class and save your changes whilst the server is running it should automatically restart the server so that you can test your changes easily. Also dont worry when you get the error "GET /favicon.ico HTTP/1.1" 404 Not Found as that is simply the browser looking for the tab icon, thereby nothing we need to worry about. If you get an Internal Server Error on the POST method then that is most likely caused due to not having Ollama running in the background.
Once the server is running you'll be able to access it over http://127.0.0.1:8000/ and over http://127.0.0.1:8000/docs you'll be able to look at it using the swagger UI for FastAPI.



## Additional Info

For Modeltypes be careful not to download too big ones as the more billion parameters a model has the more gb RAM are required to run it well. In general you can go by 7b:8gb, 13b:16gb, 70b:64gb.