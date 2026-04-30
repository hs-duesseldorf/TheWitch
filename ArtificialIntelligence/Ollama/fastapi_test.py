from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from llamaFunctionTest import paraphrase_sentence

app = FastAPI()


@app.post("/generate")
def generate():

    result = paraphrase_sentence()

    return {
        "result": result
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
    <head>
        <title>testing Ollama with FastAPI</title>
    </head>

    <body>
        <h2>Generate text</h2>

        <button onclick="send()">Generate</button>

        <h3>Output:</h3>
        <pre id="output"></pre>

        <script>
        async function send() {

            const response = await fetch("/generate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                }
            });

            const data = await response.json();

            document.getElementById("output")
                .textContent = data.result;
        }
        </script>

    </body>
</html>
"""