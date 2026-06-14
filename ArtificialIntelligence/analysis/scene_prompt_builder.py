from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from shared.events import Scene

logger = logging.getLogger("ai")

PACKAGE_PATH = Path(__file__).resolve().parent.parent / "package.json"

SYSTEM_PROMPT = (
    "/no_think\n"
    "Antworte ausschließlich auf Deutsch.\n"
    "Du bist ein reiner Text-Transformator.\n"
    "\n"
    "AUFGABE:\n"
    "Formuliere den Basetext als natürlich gesprochene Zeile einer weisen, leicht düsteren Wahrsagerin um.\n"
    "\n"
    "PFLICHTREGELN:\n"
    "- Bewahre alle konkreten Anweisungen, Handlungen und Fakten.\n"
    "- Erfinde keine neuen Informationen.\n"
    "- Formuliere den Inhalt neu, ohne seine Bedeutung zu verändern.\n"
    "- Gib genau EINE einzige umformulierte Version zurück.\n"
    "- Gib niemals mehrere Varianten derselben Aussage aus.\n"
    "- Gib niemals Alternativen, Auswahlmöglichkeiten oder Vorschläge aus.\n"
    "- Nach der ersten vollständigen Antwort endet die Ausgabe sofort.\n"
    "- Kein Markdown, keine Listen, keine Nummerierung.\n"
    "- Keine Klammern, keine Emojis, keine Sternchen und keine Backticks.\n"
    "- Gib nur die finale gesprochene Antwort aus.\n"
    "\n"
    "SPRACHE:\n"
    "- Natürliches, flüssiges und grammatikalisch korrektes Deutsch.\n"
    "- Ruhig, präzise, leicht düster und gut sprechbar.\n"
    "- Lieber einfach und korrekt als kreativ und fehlerhaft.\n"
    "- Vermeide holprige Formulierungen.\n"
    "- Vermeide unnötig komplizierte Satzkonstruktionen.\n"
    "- Nutze vollständige und natürlich klingende Sätze.\n"
    "\n"
    "VARIATION:\n"
    "- Variiere Wortwahl, Rhythmus und Satzstruktur.\n"
    "- Vermeide wiederkehrende Phrasen.\n"
    "- Nutze passende Synonyme.\n"
    "- Wiederhole niemals den Basetext wortwörtlich.\n"
    "- Die Antwort soll sich deutlich von früheren Formulierungen unterscheiden.\n"
    "\n"
    "Prüfe vor der Ausgabe Grammatik, Satzbau, Wortstellung, Bezüge und Zeichensetzung.\n"
    "Gib niemals mehrere Versionen derselben Antwort aus.\n"
    "/no_think\n"
)


@dataclass(frozen=True)
class ScenePromptContext:
    gaslight_pending: bool = False
    gaslight_hand_name: str | None = None


class ScenePromptBuilder:
    def __init__(self):
        self.scenes = self._load_scenes()
        self._current_base_texts: dict[str, str] = {}
        self._shuffle_bags: dict[str, list[str]] = {}

    def build_prompt(self, scene: Scene, context: ScenePromptContext | None = None) -> str | None:
        context = context or ScenePromptContext()

        if scene is Scene.SCENE_4_AWAITING_HAND:
            return None

        base_text = self._base_text(scene, context)
        if not base_text:
            return None
        return self._wrap(scene, base_text)

    def _base_text(self, scene: Scene, context: ScenePromptContext) -> str | None:
        if scene is Scene.DEBUG_GASLIGHT and context.gaslight_pending:
            hand_name = context.gaslight_hand_name or "unbekannte"

            template = self._scene_text(scene.value)
            if not template:
                template = (
                    "Du hast mir deine {hand_name} Hand gezeigt. "
                    "Genau diese ist die falsche. Nimm die andere Hand."
                )
            return template.replace("{hand_name}", hand_name)

        return self._scene_text(scene.value)

    def _scene_text(self, scene_name: str) -> str | None:
        entry = self.scenes.get(scene_name)
        if not isinstance(entry, list):
            return None
        candidates = [
            item.strip()
            for item in entry
            if isinstance(item, str) and item.strip()
        ]
        if not candidates:
            return None
        return self._next_from_shuffle_bag(scene_name, candidates)

    def _get_examples(self, scene_name: str) -> list[str]:
        entry = self.scenes.get(scene_name)
        if not isinstance(entry, list):
            return []
        current = self._current_base_texts.get(scene_name)
        return [
            item.strip()
            for item in entry
            if isinstance(item, str) and item.strip() and item.strip() != current
        ]

    def _next_from_shuffle_bag(self, scene_name: str, candidates: list[str]) -> str:
        bag = self._shuffle_bags.get(scene_name, [])

        if not bag or set(bag) - set(candidates):
            bag = candidates[:]
            random.shuffle(bag)
            last = self._current_base_texts.get(scene_name)
            if last and len(bag) > 1 and bag[-1] == last:
                bag[0], bag[-1] = bag[-1], bag[0]

        chosen = bag.pop()
        self._shuffle_bags[scene_name] = bag
        self._current_base_texts[scene_name] = chosen
        return chosen

    def _wrap(self, scene: Scene, base_text: str) -> str:
        scene_name = scene.value
        parts = [
            SYSTEM_PROMPT,
            f"Szene: {scene_name}.",
            "Hier ist dein verbindlicher Basetext.",
            "Formuliere ihn neu und variiere Tonfall, Rhythmus, Satzbau und Wortwahl deutlich.",
            "Die Bedeutung und jede konkrete Handlungsanweisung müssen vollständig erhalten bleiben.",
            "Erzeuge genau eine einzige Neuformulierung des Basetextes.",
            "Gib niemals mehrere Varianten derselben Aussage aus.",
            "Gib niemals Alternativen, Auswahlmöglichkeiten oder Vorschläge aus.",
            "Nach der ersten vollständigen Antwort endet die Ausgabe sofort.",
            "Die Antwort darf aus einem oder mehreren Sätzen bestehen, solange sie genau eine einzige Version darstellt.",
            "Schreibe natürliches, grammatikalisch korrektes und gut sprechbares Deutsch.",
            "Schreibe lieber einfacher und korrekt als poetisch und fehlerhaft.",
            "Vermeide unnötig komplizierte oder verschachtelte Sätze.",
            "Nutze die Beispiele ausschließlich als Stilvorbild.",
            "Die Beispiele dürfen niemals wörtlich wiedergegeben werden.",
            f"Basetext: {base_text.strip()}",
        ]

        examples = self._get_examples(scene_name)
        if examples:
            parts.append(
                "Die folgenden Beispiele dienen ausschließlich als stilistische Orientierung."
                "Übernimm höchstens Tonfall, Rhythmus, Satzlänge und Atmosphäre."
                "Du darfst sie nur dann genau übernehmen, wenn du keine fehlerfreie Deutsche Alternative erstellen kannst."
            )
            for index, example in enumerate(examples, start=1):
                parts.append(f"Beispiel {index}: {example}")

        if scene_name == "scene_1_welcome":
            parts.append(
                "EXTRA INFO:"
                "Sag nicht das man sich neben dich setzen soll, wenn nur zu dir oder dir gegenüber!"
                "Erwähne NICHT das Wort Willkommensraum!"
                "Sage niemals 'schau dir dein Schicksal an', da du das Schicksal offenbarst würdest du es eher zeigen!"
            )
            
        if scene_name == "scene_3_intro":
            parts.append(
                "EXTRA INFO:"
                "Sage niemals etwas wie 'meine Hand in deiner Hand'. Es geht nur um die Hand der Person die dir gegenüber sitzt!"
                "Nutze nie das Wort 'now'. Nutze stattdessen das deutsche Wort 'jetzt'."
            )
        
        if scene_name == "scene_debug_gaslight":
            parts.append(
                "EXTRA INFO:"
                "Sage niemals wort wörtlich 'Hand gezeigt die falsche.' sondern eher 'Hand gezeigt genau diese ist die falsche.' oder 'Hand gezeigt jedoch leider ist dies genau die falsche.'"
            )
        
        if scene_name == "scene_5_handscan":
            parts.append(
                "EXTRA INFO:"
                "Sag nur das du selber etwas siehsts oder das dir etwas etwas sagt."
                "Sage niemals etwas wie 'jetzt siehst du deine Hand' oder 'jetzt siehst du alles was du brauchst'."
                "Verwende niemals die Anrede 'du' oder das Wort 'stehen'!"
            )

        return "\n".join(parts)

    @staticmethod
    def _load_scenes() -> dict:
        try:
            package = json.loads(PACKAGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("The content package is not valid: %s", PACKAGE_PATH)
            return {}
        return package.get("scenes", {})
