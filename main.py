from runners.runner import TwoStreamCNNrunner
from tensorboardX import SummaryWriter
import yaml

from utils.utils import dict_to_namespace


def main():
    log_dir = './logs'
    logger = SummaryWriter(log_dir)

    config_path = './configs/config.yml'

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config = dict_to_namespace(config)  # Convert dictionary to Namespace object

    runner = TwoStreamCNNrunner(config = config, logger = logger)
    runner.train()
    runner.test()


if __name__ == "__main__":
    main()