import os, shutil

from torch.utils.tensorboard import SummaryWriter


class Tensorboard:
    def __init__(self, args) -> None:
        # os.makedirs(f'runs/{args.current_time}-{args.task}-{args.mode}')
        # os.makedirs(f'runs/{args.current_time}-{args.task}-{args.mode}/')

        self.writer = SummaryWriter(f"runs/{args.current_time}-{args.task}-{args.mode}")
        
        self.yaml_copy(args)

    def yaml_copy(self, args):
        filename = f'{args.task}_config.yaml'

        shutil.copyfile(
            f"configs/{filename}",
            f"runs/{args.current_time}-{args.task}-{args.mode}/{filename}",
        )
    
    def add_scalar(self, train_mode, acc, loss, f1, epoch) -> None:
        self.writer.add_scalar(f"Acc/{train_mode}", acc, epoch)
        self.writer.add_scalar(f"Loss/{train_mode}", loss, epoch)
        self.writer.add_scalar(f"F1-score/{train_mode}", f1, epoch)
    
    def add_lr_scalar(self, lr, epoch) -> None:
        self.writer.add_scalar("lr", lr, epoch)
        

    def __del__(self):
        self.writer.close()


if __name__ == "__main__":
    Tensorboard()
