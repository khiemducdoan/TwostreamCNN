from runners.runner import TwoStreamCNNrunner
from tensorboardX import SummaryWriter
import yaml
import torch 
from utils.utils import dict_to_namespace
from torchvision.transforms import v2,InterpolationMode

def main():
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(226, 226), interpolation= InterpolationMode.NEAREST),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
    ])
    log_dir = './logs'    
    logger = SummaryWriter(log_dir)

    config_path = './configs/config.yml'

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config = dict_to_namespace(config)  # Convert dictionary to Namespace object

    runner = TwoStreamCNNrunner(config = config, logger = logger,transform= transform)
    runner.train()
    runner.test()


if __name__ == "__main__":
    main()