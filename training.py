# requirements for training, this is for my ref:
# 1. need to reshape according to the vit input image size wich is 224*224
# 2. dataset, which is imagenet
# 3. then we need to construct the ds accordingly like 224*224, need to augment, and smoe speciffic things to imagenet like the mean and sd fro images, some normalization as i remeber - need to refer something or the other
# 4. need to decide upon the hyperparameters - no. of epochs, batch_size, learning rate, and model hyperparamets such as in_classes, image_Size, num_layers, embedding_dims, dropout(one normal and one for attention), num_heads(fuck this i dont like this, why to parallelize it man, shit)
# 5. train, test, valid split the ds
# 6. and optimizer for optimizing the lr in backward prop
# 7. tqdm lib for tracking beautrifully the training
# 8. and some matrices like train_loss, train_acc, val_acc, val_loss
# 9. and yeah the device thing need to implement it correctly

# need to look upon some libs first:
# 1. torch.optim -> adamw optimizer(enhanced version of adam, where normalization needed after, here it is solved within -> uses learning rate to upgrade the wights), scheduler(which is adjusting thing for learning rate and suggested to use something like cosine annealing lr which goes by some formular like lr(new_) = lr(min) + ((lrmax-lrmin)/2)*(1+cos((curr epoch no./no. of epochs)*pi)), weird ass formula), loss thing(obviously cross entrpoy loss)

# 2. gem of a lib from torchvision -> transforms, god tier lib covering the secret of augmentation taht roboflow provides us -> compose(composes every type of augementation we provide), resize(obvious!), randomhorizontalflip, randomrotation, colr jitter(some useless things like randomize the intensity values, play with the values literally), to tensor(most important, as the models cannot comprhend pil directly so we convert the rgb), and imagenet normalize thing which is specifed before -> mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

# 3. torchvision.datasets lib to provide cifar10, cifar100, and imagenet lib(need to check for imagenet alone)

# 4. sklearn metrics which gives us these metrics calc -> acc_score, precison_recall_fscore, confusion_matrix

# this is too much, i'll start gently through cifar 100, no. of epochs : 50, learning rate = 3e-4, batch_size = 32, no_of_workers = 4(if needed, extra bs)

# Vision Transformers (ViTs)
# From scratch: 1e-4 to 3e-4 (0.0001 to 0.0003)

# Fine-tuning: 1e-5 to 5e-6 (0.00001 to 0.000005)

# Requires learning rate warm-up for stable training

# learning rate is inspired from this

# so to decide upon the functions now, and neatly implement this inside a class
# 1. constructor - inintlaize teh model configs and everything like self., self. evrything
# 2. prepare_dataset - load and train test val split
# 3. train - train the model with all the hyperparams
# 4. validate - validate and calculate the scores
# 5. train_epoch - for a single epoch decompose it(inspired from microservices, hehe)
# 6. save model - in models folder
# 7. load model - from models thing for testing afterward
# 8. plot things are remaining, bs, i'll just print the accurachy and eveything lilke loss in terminal as of now, 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from main import VisionTransformer
import time
import os

class VitTrainer:
    def __init__(self, noepochs=50, batch_size=32, learning_rate=3e-4, device = None, model_config = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noepochs = noepochs  
        self.bs = batch_size
        self.lr = learning_rate

        if model_config is None:
            model_config = {
                'image_size' : 224,
                'patch_size' : 32,
                'in_channels' : 3,
                'embedding_dims' : 768,
                'num_layers' : 12,
                'num_heads' : 12,
                'mlp_ratio' : 4.0,
                'num_classes' : 100,
                'dropout' : 0.1,
                'attention_dropout' : 0.1
            }

        self.model_config = model_config
        self.model = VisionTransformer(**model_config).to(self.device)

        self.train_loader, self.val_loader, self.test_loader = self.prepare_dataset()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.1)  # Fixed: was optimzer
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=noepochs)  # Fixed: optimizer reference
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.train_accuracies = []  # Fixed: was train_acc
        self.val_losses = []
        self.val_accuracies = []  # Fixed: was val_acc

        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters on {self.device}")

    def prepare_dataset(self):
        train_transform = transforms.Compose([
            transforms.Resize((self.model_config['image_size'], self.model_config['image_size'])),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        
        ])

        val_transform = transforms.Compose([
            transforms.Resize((self.model_config['image_size'], self.model_config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        trainds = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        valds = CIFAR100(root='./data', train=False, download=True, transform=val_transform)  # Fixed: line break issue
        testds = valds

        train_loader = DataLoader(trainds, batch_size=self.bs, shuffle=True, num_workers=4)
        val_loader = DataLoader(valds, batch_size=self.bs, shuffle=False, num_workers=4)
        test_loader = DataLoader(testds, batch_size=self.bs, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training")

        for batch_idx, (data, targets) in progress_bar:
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()  
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()  

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()

            progress_bar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'accuracy': f'{100 * correct / ((batch_idx + 1) * self.bs):.2f}%'
            })

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100 * correct / len(self.train_loader.dataset)

        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validating"):
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()

        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100 * correct / len(self.val_loader.dataset)

        return epoch_loss, epoch_acc
    
    def train(self):
        print("starting the training, gear up.....") 
        best_val_acc = 0  

        for epoch in range(self.noepochs): 
            start_time = time.time()

            print(f"\nEpoch {epoch + 1}/{self.noepochs}")
            print('-' * 50)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc) 
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc) 

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

            if val_acc > best_val_acc: 
                best_val_acc = val_acc 
                self.save_model(f"best_model_epoch_{epoch + 1}.pth")
                print(f"Best model saved at epoch {epoch + 1} with accuracy {best_val_acc:.2f}%")  

        print(f"Best Validation Accuracy: {best_val_acc:.2f}%") 

    def test(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, targets in tqdm(self.test_loader, desc="Testing"):
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds) * 100
        conf_matrix = confusion_matrix(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')

        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_model(self, filename):
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, os.path.join('models', filename))

    def load_model(self, filename):
        checkpoint = torch.load(os.path.join('models', filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_config = checkpoint['model_config']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        print(f"Model loaded from {filename}")

def main():
    trainer = VitTrainer(noepochs=50, batch_size=32, learning_rate=3e-4)
    trainer.train()
    results = trainer.test()
    print("\nFinal Test Results:")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"F1 Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()







