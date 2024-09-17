import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import torch.multiprocessing as mp


def train(rank, args, model, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed + rank)
    # training code for mnist hogwild

    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, train_loader, optimizer)


def main():
    parser = argparse.ArgumentParser(description="MNIST Training Script")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,  
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=2,
        metavar="N",
        help="how many training processes to use (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--save-dir", default="./", help="checkpoint will be saved in this directory"
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=1, 
        metavar='N', 
        help='number of epochs to train (default: 10)'
    ) 
    parser.add_argument('--save_model', action='store_true', default=False,
                    help='save the trained model to state_dict')



    args, unknown = parser.parse_known_args()                


    #args = parser.parse_args()

    torch.manual_seed(args.seed)
    mp.set_start_method("spawn", force=True)

    model = Net()
    # create model and setup mp

    model.share_memory()

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": 1,
        "pin_memory": True,
        "shuffle": True,
    }

    # create mnist train dataset
                    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    # dataset2 = datasets.MNIST('../data', train=False,
    #                    transform=transform)

    # mnist hogwild training process
    processes = []
                     
    for rank in range(args.num_processes):
        #p = mp.Process(target=train, args=(rank, args, model, device,
        p = mp.Process(target=train, args=(rank, args, model, dataset1, kwargs))                       
        
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

 # save model check point   
    if args.save_model: 
        torch.save(model.state_dict(), "mnist_cnn.pt")  
          
        
if __name__ == "__main__":
    main()