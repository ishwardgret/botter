
import logging
from rasa_core.agent import Agent
from rasa_core.domain import Domain
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_nlu.model import Trainer, Metadata, Interpreter

interpreter = Interpreter.load('./models/nlu/default/chat')
agent = Agent.load('./models/dialogue', interpreter=interpreter)


def predict(msg):
    print("  BOT :: ", agent.handle_text(msg)[0]["text"])


if __name__ == '__main__':
    while 1 == 1:
        msg = input('you :: ')
        predict(msg)
