import torch
import argparse
import datetime
from load_model import load_model
from networks import compressor, decompressor
from JPEGCompressor import jpeg_analysis
from QuantizedCompressor import quantization_analysis
from calculate_metrics import calculate_metrics
from utils import plot_image, save_image, metrics
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms

def main():
    """
    Main function to train latent vector or run evaluation metrics
    :return: None
    """

    parser = argparse.ArgumentParser(description="Train model or evaluate model (default)")
    parser.add_argument("--train-model", action="store_true", default=False)

    arg_parser = parser.parse_args()

    if arg_parser.train_model:
        PATH = f"../results/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        cuda0 = torch.device('cuda:0')
        generator = load_model()
        #inputLatent = torch.randn(1, 512).cuda()
        #original_image = generator(inputLatent).detach().clone()
        #print(original_image)
        image = cv2.imread('./bovik.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([transforms.ToTensor()])
        image2 = transform(image)
        original_image = torch.reshape(image2, (1, 3, 512, 512)).to(cuda0)

        #print(original_image)
        #plot_image(original_image)
        save_image(original_image, PATH, "")

        image_compressed_vector = compressor(generator, original_image, PATH)
        torch.save(image_compressed_vector, f"{PATH}/ICV.pt")

        quantization_analysis(generator, image_compressed_vector.clone(), PATH)
        jpeg_analysis(original_image, PATH)

    else:
        calculate_metrics()


if __name__ == "__main__":
    main()
