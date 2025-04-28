import os
import random
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

import fiftyone as fo

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import faiss
import numpy as np
import torch


class TripletFODataset(Dataset):
    def __init__(self, samples, transform=None, label_to_idx=None):
        """
        Параметры:
            samples (list): Список кортежей (filepath, label) – путь к изображению и его строковая метка.
            transform: Трансформации для изображения.
            label_to_idx (dict): Словарь для отображения строковой метки в числовой индекс.
                            Если None, он будет вычислен по списку samples.
        """
        self.transform = transform
        # Если не передан mapping, вычисляем его из всех меток
        if label_to_idx is None:
            labels = sorted({label for _, label in samples})
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        # Преобразуем метки в числовые индексы
        self.samples = [(filepath, self.label_to_idx[label]) for filepath, label in samples]

        # Построим словарь: для каждого класса список индексов образцов данного класса
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.samples):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Возвращает кортеж:
        (anchor_img, positive_img, negative_img, anchor_label, negative_label)
        """
        filepath, anchor_label = self.samples[index]
        try:
            anchor_img = Image.open(filepath).convert("RGB")
        except:
            anchor_img = Image.open('/kavesnin/delete_this/256_ObjectCategories/198.spider/198_0001.jpg').convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)

        # Выбираем позитив: другое изображение того же класса
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(self.class_to_indices[anchor_label])
        positive_filepath, _ = self.samples[positive_index]
        try:
            positive_img = Image.open(positive_filepath).convert("RGB")
        except:
            positive_img = Image.open('/kavesnin/delete_this/256_ObjectCategories/198.spider/198_0001.jpg').convert(
                "RGB")
        if self.transform:
            positive_img = self.transform(positive_img)

        # Выбираем негатив: изображение из другого класса
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.class_to_indices.keys()))
        negative_index = random.choice(self.class_to_indices[negative_label])
        negative_filepath, negative_label = self.samples[negative_index]
        try:
            negative_img = Image.open(negative_filepath).convert("RGB")
        except:
            negative_img = Image.open('/kavesnin/delete_this/256_ObjectCategories/198.spider/198_0001.jpg').convert(
                "RGB")
        if self.transform:
            negative_img = self.transform(negative_img)

        # Приводим метки к тензорам
        return (anchor_img, positive_img, negative_img,
                torch.tensor(anchor_label), torch.tensor(negative_label))


class EmbeddingNet(nn.Module):
    def __init__(self, backbone_name="resnet18", embedding_dim=128, pretrained=True):
        """
        Модель-эмбеддер, использующая бэкбон из timm и дополнительный FC слой.
        Параметры:
            backbone_name (str): Имя модели-бэкбона (например, "resnet18").
            embedding_dim (int): Размерность выходного эмбеддинга.
            pretrained (bool): Использовать ли предобученные веса.
        """
        super(EmbeddingNet, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        backbone_features = self.backbone.num_features
        self.fc = nn.Linear(backbone_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


def train_one_epoch(model, dataloader, optimizer, device,
                    margin_criterion, proxy_criterion, proxy_coef=1.0, sampling='none'):
    model.train()
    losses = []
    margin = getattr(margin_criterion, 'margin', 1.0)

    for batch_idx, (anchor, positive, negative, anchor_label, negative_label) in enumerate(dataloader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        anchor_label = anchor_label.to(device)
        negative_label = negative_label.to(device)

        optimizer.zero_grad()
        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        if sampling == 'none':
            A, P, N = anchor_out, positive_out, negative_out

        elif sampling == 'semi-hard':
            B = anchor_out.size(0)
            cand_emb = torch.cat([anchor_out, negative_out], dim=0)
            cand_lbl = torch.cat([anchor_label, negative_label], dim=0)
            A_list, P_list, N_list = [], [], []

            # создаём индекс-тензор на том же устройстве
            idx_tensor = torch.arange(cand_emb.size(0), device=device)

            for i in range(B):
                d_ap = torch.norm(anchor_out[i] - positive_out[i], p=2)
                mask = (cand_lbl != anchor_label[i])
                if mask.sum() > 0:
                    d_an_all = torch.norm(
                        anchor_out[i].unsqueeze(0) - cand_emb[mask],
                        p=2, dim=1
                    )
                    semi_mask = (d_an_all > d_ap) & (d_an_all < d_ap + margin)
                    if semi_mask.any():
                        # индексы среди всех кандидатов
                        valid_idxs = idx_tensor[mask][semi_mask]
                        chosen_idx = valid_idxs[torch.argmin(d_an_all[semi_mask])]
                        chosen_neg = cand_emb[chosen_idx]
                    else:
                        chosen_neg = negative_out[i]
                else:
                    chosen_neg = negative_out[i]

                A_list.append(anchor_out[i])
                P_list.append(positive_out[i])
                N_list.append(chosen_neg)

            A = torch.stack(A_list, dim=0)
            P = torch.stack(P_list, dim=0)
            N = torch.stack(N_list, dim=0)

        elif sampling == 'batch-hard':
            emb = anchor_out
            lbl = anchor_label
            dist_mat = torch.cdist(emb, emb, p=2)
            A_idx, P_idx, N_idx = [], [], []
            B = lbl.size(0)

            for i in range(B):
                pos_idx = (lbl == lbl[i]).nonzero(as_tuple=True)[0]
                neg_idx = (lbl != lbl[i]).nonzero(as_tuple=True)[0]
                if pos_idx.numel() < 2:
                    continue
                pdist = dist_mat[i][pos_idx]
                pdist[pos_idx == i] = -1.0
                j = pos_idx[torch.argmax(pdist)]
                k = neg_idx[torch.argmin(dist_mat[i][neg_idx])]
                A_idx.append(i);
                P_idx.append(j.item());
                N_idx.append(k.item())

            if len(A_idx) == 0:
                continue

            A = emb[A_idx]
            P = emb[P_idx]
            N = emb[N_idx]

        else:
            raise ValueError(f"Unknown sampling mode: {sampling}")

        margin_loss = margin_criterion(A, P, N)

        loss = margin_loss

        if proxy_criterion:
            proxy_emb = torch.cat([anchor_out, positive_out], dim=0)
            proxy_lbl = torch.cat([anchor_label, anchor_label], dim=0)
            proxy_loss = proxy_criterion(proxy_emb, proxy_lbl)
            loss += proxy_coef * proxy_loss

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            losses.append(loss.item())
            print(f"Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}")

    return losses


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            anchor, positive, negative, _, _ = batch

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate_recall_at_k(model, dataloader, k, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Из батча берём только anchor и его метку
            anchor, _, _, labels, _ = batch
            anchor = anchor.to(device)
            emb = model(anchor)
            embeddings_list.append(emb)
            labels_list.append(labels.to(device))

    embeddings_all = torch.cat(embeddings_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    distances = torch.cdist(embeddings_all, embeddings_all, p=2)
    sorted_indices = torch.argsort(distances, dim=1)

    hits = 0
    N = embeddings_all.size(0)
    for i in range(N):
        neighbors = sorted_indices[i, 1:k + 1]
        if (labels_all[neighbors] == labels_all[i]).any():
            hits += 1

    recall_at_k = hits / N
    return recall_at_k


class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, embedding_size, margin=0.1, alpha=32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.margin = margin
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, embeddings, labels):
        e = F.normalize(embeddings, p=2, dim=1)
        p = F.normalize(self.proxies, p=2, dim=1)
        sim = e @ p.t()
        labels_onehot = F.one_hot(labels, self.num_classes).float()
        pos_mask = labels_onehot
        neg_mask = 1.0 - labels_onehot

        pos_term = torch.exp(-self.alpha * (sim - self.margin)) * pos_mask
        neg_term = torch.exp(self.alpha * (sim + self.margin)) * neg_mask

        # среднее по представленным классам в батче
        present = labels.unique().numel()
        pos_loss = (torch.log(1 + pos_term.sum(0))).sum() / present
        neg_loss = (torch.log(1 + neg_term.sum(0))).sum() / self.num_classes

        return pos_loss + neg_loss


def launch_experiment(experiment_name, margin=1.0, use_proxy=False, proxy_coef=1.0, margin_lr=1e-4, proxy_lr=1e-3):
    sampling_type = {
        'triple_semi_hard': 'semi-hard',
        'triple_batch_hard': 'batch-hard',
        'triple': 'none',
    }
    margin_criterions = {
        'triple_semi_hard': nn.TripletMarginLoss(margin=margin, p=2),
        'triple_batch_hard': nn.TripletMarginLoss(margin=margin, p=2),
        'triple': nn.TripletMarginLoss(margin=margin, p=2),
    }
    num_classes = len(label_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_proxy:
        proxy_criterion = ProxyAnchorLoss(num_classes, embedding_size=128,
                                          margin=0.1, alpha=32).to(device)
    else:
        proxy_criterion = None

    save_path = f"model_{experiment_name}_{str(margin).replace('.', '_')}_lr{str(margin_lr).replace('.', '_')}"
    if use_proxy:
        save_path += f"_proxy_{str(proxy_coef).replace('.', '_')}_lr{str(proxy_lr).replace('.', '_')}"

    model = EmbeddingNet(backbone_name="levit_128", embedding_dim=128, pretrained=True).to(device)
    if use_proxy:
        optimizer = optim.Adam([
            {"params": model.parameters(), "lr": margin_lr},
            {"params": proxy_criterion.parameters(), "lr": proxy_lr}
        ], weight_decay=1e-5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=margin_lr, weight_decay=1e-5)
    val_criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    num_epochs = 2
    k = 1
    all_losses = []

    for epoch in range(num_epochs):
        print(f"\n=== Эпоха {epoch + 1}/{num_epochs} ===")
        epoch_losses = train_one_epoch(
            model, train_loader, optimizer, device, margin_criterion=margin_criterions[experiment_name],
            sampling=sampling_type[experiment_name], proxy_criterion=proxy_criterion, proxy_coef=proxy_coef,
        )
        all_losses.extend(epoch_losses)

        val_loss = validate(model, val_loader, val_criterion, device)
        recall_at_k = validate_recall_at_k(model, val_loader, k, device)
        print(f"Train Loss(mean@10steps)={sum(epoch_losses) / len(epoch_losses):.4f} | "
              f"Val Loss={val_loss:.4f} | Recall@{k}={recall_at_k:.4f}")

        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")

    plt.figure()
    plt.plot(all_losses)
    plt.xlabel("Step (каждые 10 батчей)")
    plt.ylabel("Loss")
    plt.title("Train loss (batch-hard) за 2 эпохи")
    plt.show()

def build_faiss_index(model, dataloader, device, embedding_dim):
    model.eval()
    all_embs = []
    all_labels = []
    with torch.no_grad():
        for anchor, _, _, anchor_label, _ in dataloader:
            anchor = anchor.to(device)
            emb = model(anchor)
            emb = emb.cpu().numpy().astype('float32')
            all_embs.append(emb)
            all_labels.extend(anchor_label.cpu().numpy())

    all_embs = np.vstack(all_embs)
    train_labels = np.array(all_labels)

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(all_embs)

    return index, train_labels

def evaluate_faiss(model, index, train_labels, dataloader, device, k=5):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for anchor, _, _, anchor_label, _ in dataloader:
            anchor = anchor.to(device)
            emb = model(anchor)
            emb = emb.cpu().numpy().astype('float32')

            distances, indices = index.search(emb, k)

            true = anchor_label.cpu().numpy()
            for i, t in enumerate(true):
                neigh = train_labels[indices[i]]
                vals, counts = np.unique(neigh, return_counts=True)
                pred = vals[np.argmax(counts)]
                if pred == t:
                    correct += 1
                total += 1

    accuracy = correct / total
    return accuracy


dataset = fo.Dataset.from_dir(
    dataset_dir="/kavesnin/delete_this/256_ObjectCategories",
    dataset_type=fo.types.ImageClassificationDirectoryTree,
)
print(f"Загружен Caltech256: {len(dataset)} образцов")

# Читаем CSV с валидационными сэмплами (в столбце filename)
# val_df = pd.read_csv("homework/val.csv")
val_df = pd.read_csv("val.csv")
val_filenames = set(val_df["filename"].tolist())

train_samples = []
val_samples = []

for sample in dataset:
    filename = os.path.basename(sample.filepath)
    # Предполагается, что метка хранится в поле ground_truth с ключом "label"
    if "ground_truth" in sample and sample["ground_truth"] is not None:
        label = sample["ground_truth"]["label"]
    else:
        # Если поле отсутствует, можно попробовать sample["label"]
        label = sample.get("label", None)
    if label is None:
        continue
    if filename in val_filenames:
        val_samples.append((sample.filepath, label))
    else:
        train_samples.append((sample.filepath, label))

print(f"Обучающих сэмплов: {len(train_samples)}")
print(f"Валидационных сэмплов: {len(val_samples)}")

# Вычисляем общее отображение меток (label -> числовой индекс)
all_labels = {label for _, label in (train_samples + val_samples)}
labels_sorted = sorted(all_labels)
label_to_idx = {label: idx for idx, label in enumerate(labels_sorted)}

# Определяем трансформации для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Создаем PyTorch-датасеты
train_dataset = TripletFODataset(train_samples, transform=transform, label_to_idx=label_to_idx)
val_dataset = TripletFODataset(val_samples, transform=transform, label_to_idx=label_to_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# choose and run experiment
launch_experiment('triple_semi_hard') # 0.8051
launch_experiment('triple_semi_hard', 0.2) # 0.8120
launch_experiment('triple_batch_hard') # 0.6986
launch_experiment('triple_batch_hard', 0.2) # 0.7429
launch_experiment('triple_semi_hard', 0.2, use_proxy=True, proxy_coef=0.1) # 0.7988
launch_experiment('triple_semi_hard', 0.2, use_proxy=True, proxy_coef=1.0, margin_lr=1e-4, proxy_lr=1e-4) # 0.7881
launch_experiment('triple_semi_hard', 0.2, use_proxy=True, proxy_coef=1.0, margin_lr=1e-4, proxy_lr=1e-3) # 0.8229
launch_experiment('triple_semi_hard', 0.2, use_proxy=True, proxy_coef=1.0, margin_lr=1e-4, proxy_lr=1e-2) # 0.8136
launch_experiment('triple_semi_hard', 0.2, use_proxy=True, proxy_coef=1.0, margin_lr=1e-4, proxy_lr=3e-3) # 0.7645
launch_experiment('triple_semi_hard', 0.2, use_proxy=True, proxy_coef=0.5, margin_lr=1e-4, proxy_lr=1e-3) # 0.8254

# inference
model_path = \
    'model_triple_semi_hard_0_2_lr0_0001_proxy_0_5_lr0_001/model_epoch_2.pth'
backbone_name = 'levit_128'
embedding_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingNet(backbone_name=backbone_name, embedding_dim=embedding_dim, pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# index
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 128
index, train_labels = build_faiss_index(model, train_loader, device, embedding_dim)

# evaluate - choose k
faiss_acc = evaluate_faiss(model, index, train_labels, val_loader, device, k=5)
print(f"FAISS top 5 accuracy: {faiss_acc * 100:.2f}%")

faiss_acc = evaluate_faiss(model, index, train_labels, val_loader, device, k=3)
print(f"FAISS top 3 accuracy: {faiss_acc * 100:.2f}%")

faiss_acc = evaluate_faiss(model, index, train_labels, val_loader, device, k=10)
print(f"FAISS top 10 accuracy: {faiss_acc * 100:.2f}%")

faiss_acc = evaluate_faiss(model, index, train_labels, val_loader, device, k=1)
print(f"FAISS top 1 accuracy: {faiss_acc * 100:.2f}%")

