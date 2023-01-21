import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=256, help='INSERT IMAGE_SIZE')
parser.add_argument('--aug_factor', type=int, default=2, help='INSERT AUGUMENTATION FACTOR')
parser.add_argument('--upscale_factor', type=int, default=4, help='INSERT UPSCALE FACTOR')
parser.add_argument('--batch_size', type=int, default=64, help='INSERT BATCH SIZE')
parser.add_argument('--epochs', type=int, default=1000, help='INSERT EPOCHS')
parser.add_argument('--step_size', type=int, default=200, help='INSERT STEP SIZE')
parser.add_argument('--load_num', type=int, default=-1, help='INSERT LOAD NUM')
parser.add_argument('--on_local', default=False, action='store_true', help='INSERT ON LOCAL')
parser.add_argument('--display_num', type=int, default=0, help='INSERT DISPLAY NUM')
# Random Degradation
parser.add_argument('--rd', default=False, action='store_true', help='INSERT RANDOM DEGRADATION ON/OFF')
args = parser.parse_args()
