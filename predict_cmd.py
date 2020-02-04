import logging
from rasa_core.agent import Agent
from rasa_nlu.model import Trainer, Metadata, Interpreter
import sys

from contextlib import contextmanager
import sys
import os


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


loggerTf = logging.getLogger('tensorflow')
loggerTf.setLevel(logging.ERROR)

with suppress_stdout():
    interpreter = Interpreter.load('./models/nlu/default/chat')
    agent = Agent.load('./models/dialogue', interpreter=interpreter)

# interpreter = Interpreter.load('./models/nlu/default/chat')
# agent = Agent.load('./models/dialogue', interpreter=interpreter)


def predict(msg):
    return agent.handle_text(msg)[0]["text"]


def main():

    if(len(sys.argv) != 2):
        print("Invalid params", file=sys.stderr)
    else:
        text = sys.argv[1]
        listStr = text.split("-")
        text = " ".join(text.split("-"))
        print(predict(sys.argv[1]))

main()
