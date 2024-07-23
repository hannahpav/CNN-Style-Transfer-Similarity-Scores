import torchvision.transforms as transforms

def load_image(image_name, image_size=256):
    loader = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)