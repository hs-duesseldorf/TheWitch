from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
#this is just the script we ourselves made imported 
from llamaFunctionTest import paraphrase_sentence

app = FastAPI()

#input structure
class SentenceInput(BaseModel):
    sentence: str


#API endpoint to get called by the generate button on the site
@app.post("/generate")
def generate(data: SentenceInput):
    result = paraphrase_sentence(data.sentence)

    return {
        "result": result
    }


#simple webpage for testing
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
    <head>
        <title>testing Ollama with FastAPI</title>
    </head>

    <body>
        <h2>Enter a sentence</h2>

        <input id="sentence" type="text" size="50">
        <button onclick="send()">Generate</button>

        <h3>Output:</h3>
        <pre id="output"></pre>

        <script>
            <!-- Function for our button to call onclick -->
            async function send() {

                <!-- save input -->
                const sentence =
                    document.getElementById("sentence").value;

                <!-- use saved input as parameter for our @app.post method and save the result/output -->
                const response = await fetch("/generate", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        sentence: sentence
                    })
                });

                <!-- antwort umformen -->
                const data = await response.json();

                <!-- show result on website -->
                document.getElementById("output")
                    .textContent = data.result;
            }
        </script>
    </body>
</html>
"""