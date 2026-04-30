import ollama
import random as r

#init ollama client
client = ollama.Client()

phrase = {
  "einstieg": [
    "Setz dich. Der Raum hat dich bereits bemerkt.",
    "Du bist nicht zufällig hier erschienen.",
    "Dein Atem trägt Geschichten mit sich.",
    "Ich habe dich erwartet, auch wenn du es nicht wusstest.",
    "Sprich nicht. Ich sehe bereits genug.",
    "Der Moment hat dich zu mir geführt."
  ],
  "allgemeine_aussagen": [
    "Du trägst eine Entscheidung in dir, die du noch nicht anerkannt hast.",
    "Ein Teil von dir hält fest, während ein anderer längst loslassen will.",
    "Du suchst Klarheit, aber fürchtest, was sie dir zeigen könnte.",
    "Etwas Unausgesprochenes begleitet dich schon lange.",
    "Du stehst näher an einer Veränderung, als du glaubst.",
    "Dein Weg ist nicht blockiert, nur verzögert."
  ],
  "elemente": {
    "holz": {
      "zu_stark": [
        "Dein Holz wächst unkontrolliert. Es drängt nach vorne, ohne Richtung.",
        "Zu viel Holz in dir lässt dich rastlos werden.",
        "Dein Wachstum ist schnell, aber nicht stabil."
      ],
      "zu_schwach": [
        "Dein Holz ist schwach. Es fehlt dir an Aufbruch.",
        "Das Wachstum in dir ist gehemmt.",
        "Du hältst dich zurück, obwohl Bewegung nötig wäre."
      ],
      "in_balance": [
        "Dein Holz ist im Gleichgewicht. Wachstum und Richtung sind im Einklang.",
        "Du entwickelst dich in deinem eigenen Tempo.",
        "Dein Weg entfaltet sich natürlich."
      ],
      "blockiert": [
        "Dein Holz ist blockiert. Etwas hindert dich am Voranschreiten.",
        "Dein Wachstum stößt auf unsichtbaren Widerstand.",
        "Du willst dich bewegen, aber etwas hält dich fest."
      ]
    },
    "feuer": {
      "zu_stark": [
        "Dein Feuer brennt zu hell. Es verzehrt mehr, als es wärmt.",
        "Zu viel Feuer bringt Unruhe in dein Herz.",
        "Deine Energie ist stark, aber schwer zu kontrollieren."
      ],
      "zu_schwach": [
        "Dein Feuer ist schwach. Es fehlt dir an Antrieb.",
        "Die Wärme in dir ist zurückgegangen.",
        "Du zögerst, wo Entschlossenheit gebraucht wird."
      ],
      "in_balance": [
        "Dein Feuer ist ausgeglichen. Es gibt dir Kraft und Klarheit.",
        "Deine Energie trägt dich, ohne dich zu verbrennen.",
        "Du handelst mit Wärme und Bewusstsein."
      ],
      "blockiert": [
        "Dein Feuer ist eingeschlossen. Es kann sich nicht entfalten.",
        "Deine Energie findet keinen Ausdruck.",
        "Etwas unterdrückt deine innere Kraft."
      ]
    },
    "erde": {
      "zu_stark": [
        "Deine Erde ist zu schwer. Sie hält dich fest.",
        "Zu viel Erde macht dich unbeweglich.",
        "Du klammerst dich an Sicherheit."
      ],
      "zu_schwach": [
        "Deine Erde ist schwach. Dir fehlt Halt.",
        "Du findest keinen festen Stand.",
        "Stabilität entzieht sich dir."
      ],
      "in_balance": [
        "Deine Erde ist im Gleichgewicht. Sie trägt dich sicher.",
        "Du bist geerdet und stabil.",
        "Du findest Ruhe in dir selbst."
      ],
      "blockiert": [
        "Deine Erde ist blockiert. Stabilität ist gestört.",
        "Du suchst Halt, aber findest ihn nicht.",
        "Etwas verhindert, dass du zur Ruhe kommst."
      ]
    },
    "metall": {
      "zu_stark": [
        "Dein Metall ist zu scharf. Es trennt mehr, als es klärt.",
        "Zu viel Metall macht dich hart.",
        "Du urteilst zu schnell und zu streng."
      ],
      "zu_schwach": [
        "Dein Metall ist schwach. Dir fehlt Klarheit.",
        "Grenzen verschwimmen für dich.",
        "Du lässt zu viel durch, was dich schwächt."
      ],
      "in_balance": [
        "Dein Metall ist im Gleichgewicht. Klarheit begleitet dich.",
        "Du erkennst, was bleiben darf und was gehen muss.",
        "Deine Entscheidungen sind präzise."
      ],
      "blockiert": [
        "Dein Metall ist blockiert. Klarheit bleibt verborgen.",
        "Du siehst, aber erkennst nicht.",
        "Etwas verhindert deine Unterscheidungskraft."
      ]
    },
    "wasser": {
      "zu_stark": [
        "Dein Wasser ist zu tief. Du verlierst dich darin.",
        "Zu viel Wasser bringt Unsicherheit.",
        "Du ziehst dich zu weit zurück."
      ],
      "zu_schwach": [
        "Dein Wasser ist schwach. Dir fehlt Tiefe.",
        "Du meidest, was unter der Oberfläche liegt.",
        "Deine Intuition ist leise geworden."
      ],
      "in_balance": [
        "Dein Wasser ist ruhig und klar.",
        "Du vertraust deiner inneren Tiefe.",
        "Deine Intuition führt dich sicher."
      ],
      "blockiert": [
        "Dein Wasser ist gestaut. Es kann nicht fließen.",
        "Gefühle bleiben eingeschlossen.",
        "Etwas hindert dich daran, loszulassen."
      ]
    }
  },
  "verbindung": [
    "Du stehst zwischen Bewegung und Stillstand.",
    "In dir kämpfen zwei Kräfte um Richtung.",
    "Du suchst Kontrolle, doch etwas will sich entfalten.",
    "Deine Stärke und deine Unsicherheit sind enger verbunden, als du denkst.",
    "Was dich schützt, hält dich gleichzeitig zurück."
  ],
  "zukunft": [
    "Der richtige Moment nähert sich, aber er ist noch nicht da.",
    "Wenn du jetzt handelst, veränderst du mehr als geplant.",
    "Ein neuer Weg wird sich zeigen, wenn du bereit bist.",
    "Du wirst wählen müssen, auch wenn du es vermeiden willst.",
    "Das, was kommt, hängt von deinem nächsten Schritt ab."
  ],
  "abschluss": [
    "Der Kreis ist noch nicht geschlossen.",
    "Wir werden uns wiedersehen.",
    "Der Schatten folgt dir noch.",
    "Achte auf das, was sich wiederholt.",
    "Nicht alles ist gesagt."
  ]
}

#zufällige phrasen auswählen
def random_phrases():
    
    p1 = r.choice(phrase["einstieg"])
    p2 = r.choice(phrase["allgemeine_aussagen"])
    
    anordnung = list(phrase["elemente"].keys())
    r.shuffle(anordnung)
    p3_1 = r.choice(phrase["elemente"][anordnung[0]]["zu_stark"])
    p3_2 = r.choice(phrase["elemente"][anordnung[1]]["zu_schwach"])
    p3_3 = r.choice(phrase["elemente"][anordnung[2]]["in_balance"])
    p3_4 = r.choice(phrase["elemente"][anordnung[3]]["blockiert"])
    
    p4 = r.choice(phrase["verbindung"])
    p5 = r.choice(phrase["zukunft"])
    p6 = r.choice(phrase["abschluss"])
    
    return [p1, p2, p3_1, p3_2, p3_3, p3_4, p4, p5, p6]

#define function
def paraphrase_sentence():
    ourModel = "gemma2:2b"
    
    sentence = "\n".join(random_phrases())
    print(sentence)
    #add user sentence into prompt
    ourPrompt = (f"Formuliere alle diese Sätze einzeln mystisch um. Behalte die Reihenfolge bei. gib dein Ergebnis auf deutsch ohne zusätzliche Kommentare in einem einheitlichen Block aus: {sentence}") #f for formatted
    
    #send query
    ourResponse = client.generate(
        model=ourModel,
        prompt=ourPrompt,
        options={
        "temperature": 0.5,
        "num_predict": 300
    }
    )

    #return response text
    return ourResponse.response