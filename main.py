import model
import data_loader
import trainer
import evaluate

import argparse
import torch

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epoch", help = "Set training eopchops", type = int, default = 10)
    parser.add_argument("-b", "--batch_size", help = "Set batch size", type = int, default = 16)
    parser.add_argument("-l", "--learning_rate", help = "Set learning rate", type = float, default = 0.001)

    args = parser.parse_args()

    # Load data
    data_file = "data\\plume2d"
    train_loader, test_loader = data_loader.load(data_file, args.batch_size)

    # Initialize model
    unet_model = model.UNet()

    # Train model
    trained_model = trainer.train_fluid_simulation_model(unet_model, train_loader, epochs=args.epoch, lr=args.learning_rate)

    torch.save({
        'epoch': args.epoch,
        'model_state_dict': trained_model.state_dict(),
        'loss': 0.0008 # Dummy loss value for example TODO: replace with actual loss
    }, f"unet_model_epoch_{args.epoch}.pth")

    evaluate.run_model_testing(trained_model, test_loader, visualize=False)

if __name__ == "__main__":
    main()