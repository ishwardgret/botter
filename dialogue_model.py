import logging
from rasa_core.agent import Agent
from rasa_core.domain import Domain
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_nlu.model import Trainer, Metadata, Interpreter


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    dialog_training_data_file = './data/stories.md'
    path_to_model = './models/dialogue'
    agent = Agent('domain.yml', policies=[
                  MemoizationPolicy(max_history=1),
                  KerasPolicy(epochs=50, batch_size=5,)])
    data = agent.load_data(dialog_training_data_file)
    agent.train(data,
                augmentation_factor=50,
                validation_split=0.2)
    agent.persist(path_to_model)





