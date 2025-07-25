import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import csv

# Open the CSV file
with open('C:\\Users\\sivas\\Downloads\\archive_logo_fake\\file_mapping.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Iterate over the rows and replace backslashes with forward slashes
for row in rows:
    row[0] = row[0].replace('\\', '/')

# Save the modified CSV file
'''with open('C:\\Users\\sivas\\Downloads\\archive_logo_fake\\modified_csv_file.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)'''
class LogoDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data['Filename'][idx])
        image = Image.open(img_path).convert('RGB')
        label_name = self.data['Label'][idx]
        # tagline = self.data['Tagline'][idx]
        
        if label_name == "Genuine":
            label = 0
        else:
            label = 1
            
        if self.transform:
            image = self.transform(image)
            
        return image, label 
        
dataset_path = 'C:\\Users\\sivas\\Downloads\\archive_logo_fake'
csv_file = 'C:\\Users\\sivas\\Downloads\\archive_logo_fake\\modified_csv_file.csv'

image_transforms = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(30),
    transforms.ToTensor(),

])

logo_dataset = LogoDataset(csv_file, dataset_path, transform=image_transforms) #to label

batch_size = 32
dataloader = DataLoader(logo_dataset, batch_size=batch_size, shuffle=True)

model = models.resnet50(pretrained=False)
num_classes = 2  # Adjust this based on the number of classes in your dataset
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)

model.fc = torch.nn.Linear(in_features, num_classes)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set the device for training
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model = model.to(device)

num_epochs = 20
save_interval = 4

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    if (epoch + 1) % save_interval == 0:
            # Save the trained model
            torch.save(model.state_dict(), f'classification_model_epoch{epoch+1}.pth')
