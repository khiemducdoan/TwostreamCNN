from runners.runner import TwoStreamCNNrunner
from tensorboardX import SummaryWriter
import yaml

from utils.utils import dict_to_namespace
from torchvision import transforms
from torchvision.transforms import v2 

def main():
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    log_dir = './logs'
    logger = SummaryWriter(log_dir)

    config_path = './configs/config.yml'

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config = dict_to_namespace(config)  # Convert dictionary to Namespace object

    runner = TwoStreamCNNrunner(config = config, logger = logger,transform= transforms)
    runner.train()
    runner.test()


if __name__ == "__main__":
    main()