from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from shared.events import Scene

logger = logging.getLogger("ai")

PACKAGE_PATH = Path(__file__).resolve().parent.parent / "package.json"

SYSTEM_PROMPT = (
    "/no_think\n"
    "Antworte ausschließlich auf Deutsch.\n"
    "Gib nur die finale gesprochene Antwort aus: kein Markdown, keine Formatierung und keine Erklärungen.\n"
    "Du bist ein reiner Text-Transformator.\n"
    "Formuliere den Basetext als kurze, natürlich gesprochene Zeile einer weisen, düsteren Wahrsagerin um.\n"
    "Bewahre alle konkreten Anweisungen, Handlungen und Fakten.\n"
    "Erfinde keine neuen Informationen.\n"
    "Kein Markdown, keine Klammern, keine Emojis, keine Sternchen, keine Backticks.\n"
    "Ton: ruhig, präzise, leicht dunkel und klar sprechbar.\n"
    "Formuliere ausschließlich vollständige, natürlich klingende und grammatikalisch korrekte deutsche Sätze.\n"
    "Prüfe vor der Ausgabe Grammatik, Satzbau, Wortstellung, Bezüge und Zeichensetzung. Gib niemals holprige, mehrdeutige oder unvollständige Sätze aus.\n"
    "Variiere jede Antwort in Wortwahl, Rhythmus und Satzstruktur, aber niemals auf Kosten von korrektem, natürlichem Deutsch oder klarer Bedeutung.\n"
    "Vermeide Wiederholungen derselben Formulierung und nutze passende Synonyme.\n"
    "/no_think\n"
)


@dataclass(frozen=True)
class ScenePromptContext:
    gaslight_pending: bool = False
    gaslight_hand_name: str | None = None


class ScenePromptBuilder:
    def __init__(self):
        self.scenes = self._load_scenes()

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
            return (
                f"Du hast mir deine {hand_name} Hand gezeigt. "
                "Genau diese ist die falsche. Nimm die andere Hand."
            )

        return self._scene_text(scene.value)

    def _scene_text(self, scene_name: str) -> str | None:
        entry = self.scenes.get(scene_name)
        if isinstance(entry, dict):
            entry = entry.get("base_text")
        if isinstance(entry, str):
            return entry.strip() or None
        return None

    def _wrap(self, scene: Scene, base_text: str) -> str:
        return "\n".join(
            [
                SYSTEM_PROMPT,
                f"Szene: {scene.value}.",
                "Hier ist dein verbindlicher Basetext. Variiere Tonfall, Rhythmus, Satzbau und Wortwahl deutlich.",
                "Die Bedeutung und jede konkrete Handlungsanweisung müssen erhalten bleiben.",
                "Die Neuformulierung muss natürlich klingen und aus grammatikalisch korrekten, vollständigen deutschen Sätzen bestehen.",
                "Jede Neuformulierung muss sich deutlich von vorherigen unterscheiden. Vermeide identische Phrasen.",
                f"Basetext: {base_text.strip()}",
            ]
        )

    @staticmethod
    def _load_scenes() -> dict:
        try:
            package = json.loads(PACKAGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("The content package is not valid: %s", PACKAGE_PATH)
            return {}
        return package.get("scenes", {})
