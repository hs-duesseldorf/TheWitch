# TO START STREAMLIT: 
# navigate to project (ArtificalIntelligence) Terminal PowerShell
# .venv\Scripts\Activate.ps1
# streamlit run StateMachine.py


import streamlit as app
import time

from enum import Enum, auto

#-------------------# STATE

# (Enum) Assigns iterative numbering to every state (auto())
# Includes all States used by StateMachine
class State(Enum):
    IDLE = auto()
    HAND_DETECTED = auto()
    NO_HAND = auto()
    READING_HAND = auto()
    DIALOG = auto()

#-------------------# STATE MACHINE

class StateMachine:

    def __init__(self):
        self.current_state = State.IDLE

    # PER FRAME UPDATE

    def update(self, hand_detected : bool):
        match self.current_state:
            case State.IDLE:
                self.idle_state(hand_detected)
            case State.HAND_DETECTED:
                self.hand_detected_state(hand_detected)
            case State.NO_HAND:
                self.debug_no_hand(hand_detected)

    # STATE FUNCTIONS

    def idle_state(self, hand_detected):
        if hand_detected:
            print("Hand erkannt | IDLE -> HAND_DETECTED")
            self.current_state = State.HAND_DETECTED
            return

        if not hand_detected:
            print("Keine Hand | IDLE -> NO_HAND")
            self.current_state = State.NO_HAND

    def hand_detected_state(self, hand_detected):
        if not hand_detected:
            print(" keine Hand | HAND_DETECTED -> IDLE")
            self.current_state = State.IDLE
            return
        
        if hand_detected:
            print("Hand ist erkannt")
    
    def debug_no_hand(self, hand_detected):
        if hand_detected:
            print("Hand wieder gefunden | NO_HAND -> HAND_DETECTED")
            self.current_state = State.HAND_DETECTED
            return
        
        if not hand_detected:
            print("Hand bitte vorhalten | NO_HAND -> IDLE")
            self.current_state = State.IDLE


#-------------------# TESTING 

app.title("The Witch - State Machine UI")
hand_detected = app.checkbox("Sehen wir eine Hand?", False)

if "running" not in app.session_state:
    app.session_state.running = False
if "machine" not in app.session_state:
    app.session_state.machine = StateMachine()
if "state" not in app.session_state:
    app.session_state.state = State.IDLE

column_1, column_2 = app.columns(2)
with column_1:
    if app.button("Start"):
        app.session_state.running = True
with column_2:
    if app.button("Stop"):
        app.session_state.running = False

if app.session_state.running:
    app.session_state.state = app.session_state.machine.update(hand_detected)
    time.sleep(0.5)
    app.rerun()
