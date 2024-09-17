import json
import time
import random
import torch
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from model import Net


def infer(model, dataset, save_dir, num_samples=5):
    model.eval()
    results_dir = Path(save_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), num_samples)
    for idx in indices:
        image, _ = dataset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1, keepdim=True).item()

        img = Image.fromarray(image.squeeze().numpy() * 255).convert("L")
        img.save(results_dir / f"{pred}.png")


def main():
    save_dir = "/opt/mount"
    
    # init model and load checkpoint here
    model = Net()  #  model architecture defined in model.py
    checkpoint_path = "mnist_cnn.pt"  # Path to the saved model checkpoint
    model.load_state_dict(torch.load(checkpoint_path))
    

	# create transforms and test dataset for mnist
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalization values for MNIST
    ])

    # Load the MNIST test dataset
    dataset = datasets.MNIST(
        "/opt/mount", train=False, download=True, transform=transform
    )

    infer(model, dataset, save_dir)
    print("Inference completed. Results saved in the 'results' folder.")


if __name__ == "__main__":
    main()
