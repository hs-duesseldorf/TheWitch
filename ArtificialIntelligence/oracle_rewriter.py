from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)


REWRITE_PROMPT = """
Du bist ein Experte für mystische Orakeltexte im Stil daoistischer und klassisch chinesischer Weisheitsliteratur, ähnlich dem I Ging.

DEINE AUFGABE:
Formuliere DIE GEGEBENEN Textbausteine in stimmungsvolle Orakelsätze um.

---
WICHTIGE REGELN:

- Die ursprüngliche Bedeutung muss exakt erhalten bleiben
- Keine neuen Inhalte, Deutungen oder Vorhersagen hinzufügen
- Keine bestehenden Aussagen abschwächen oder verändern
- Keine Widersprüche einführen
- Alle Eingabebausteine müssen semantisch erhalten bleiben
- Wenn du unsicher bist, bleibe näher am Original statt kreativer zu werden

---
STILRICHTLINIEN:

- Verwende indirekte, bildhafte Sprache
- Bevorzuge Naturmetaphern wie Wasser, Wind, Mond, Berge, Nebel, Bambus, Fluss, Stein, Licht und Schatten
- Drücke Wandel, Dualität oder Gleichgewicht aus
- Vermeide direkte Handlungsanweisungen
- Formuliere zeitlos, ruhig, weise und neutral
- Verwende keine modernen Begriffe
- Erzeuge eine mystische Atmosphäre ohne Bedeutungsänderung

---
STRUKTUR:

- Mehrere kurze poetische Sätze
- Kein Kommentar
- Kein erklärender Text
- Keine Markdown-Formatierung

---
EINGABE:

{input_text}

---
AUSGABEFORMAT:

{
  "Ausgabe": [
    "Satz 1.",
    "Satz 2."
  ]
}

BEISPIELE:
 
EINGABE:
"You will succeed if you remain patient."
 
AUSGABE:
{
"Ausgabe": [
"Der Fluss eilt nicht und erreicht doch das Meer.",
"In der Stille erfüllt sich, was bereits angelegt ist."
]
}
 
---
 
EINGABE:
"Now is not the right time to act."
 
AUSGABE:
{
"Ausgabe": [
"Der Wind hat seine Richtung noch nicht gefunden.",
"Der Weise verweilt, bis sich die Strömung zeigt."
]
}
 
---
 
EINGABE:
"You are facing inner conflict and must find balance."
 
AUSGABE:
{
"Ausgabe": [
"Zwei Kräfte bewegen sich im selben Raum.",
"Erst wenn sie einander loslassen, entsteht Klarheit."
]
}
 

WICHTIG:
- Ausschließlich JSON
- Keine weiteren Schlüssel
- Keine Markdown-Blöcke
- Kein Zusatztext
"""


def parse_oracle_response(text: str) -> list[str]:
    data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError("Response is not a JSON object")

    expected_keys = {"Ausgabe"}

    if set(data.keys()) != expected_keys:
        raise ValueError(
            f"Unexpected JSON keys: {list(data.keys())}"
        )

    output = data["Ausgabe"]

    if not isinstance(output, list):
        raise ValueError("Ausgabe is not a list")

    if len(output) == 0:
        raise ValueError("Ausgabe is empty")

    for item in output:
        if not isinstance(item, str):
            raise ValueError(
                "Ausgabe contains non-string values"
            )

        if not item.strip():
            raise ValueError(
                "Ausgabe contains empty strings"
            )

    return output


class OracleRewriter:
    MAX_RETRIES = 5

    def __init__(self, llm: Any):
        self._llm = llm

    async def rewrite(self, text_block: str) -> str:
        base_prompt = REWRITE_PROMPT.replace(
            "{input_text}",
            text_block,
        )

        last_error: Exception | None = None
        previous_response: str | None = None

        for attempt in range(self.MAX_RETRIES):
            prompt = base_prompt

            if previous_response is not None:
                prompt += f"""

DEINE LETZTE ANTWORT WAR UNGÜLTIG.

VALIDIERUNGSFEHLER:
{last_error}

DEINE LETZTE ANTWORT:
{previous_response}

KORRIGIERE DIE ANTWORT.

WICHTIG:
- Gib ausschließlich gültiges JSON zurück.
- Keine Erklärungen.
- Keine Markdown-Blöcke.
- Keine zusätzlichen Schlüssel.
"""

            response = await self._llm.async_generate_fortune(
                prompt
            )

            logger.info(
                "Oracle response attempt %s: %s",
                attempt + 1,
                response,
            )

            try:
                sentences = parse_oracle_response(
                    response
                )

                return "\n".join(sentences)

            except Exception as exc:
                last_error = exc
                previous_response = response

                logger.warning(
                    "Oracle validation failed "
                    "(attempt %s/%s): %s",
                    attempt + 1,
                    self.MAX_RETRIES,
                    exc,
                )

        raise RuntimeError(
            f"Oracle JSON validation failed after "
            f"{self.MAX_RETRIES} attempts: "
            f"{last_error}"
        )