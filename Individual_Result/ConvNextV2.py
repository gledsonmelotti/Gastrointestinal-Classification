import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix,
                              roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import timm

# =========================================================
# CONFIG
# =========================================================
img_size         = 384
batch_size_train = 3 # 23
batch_size_valid = 4 # 11
num_classes      = 2
num_epochs       = 200
lr               = 1e-3
weight_decay     = 1e-5
warmup_epochs    = 5
patience         = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# =========================================================
# PATHS
# =========================================================
base_path = "/media/gledson/Dados1/Classificação_GastroIntestinal/Dataset_Final/"
train_dir = os.path.join(base_path, "train")
valid_dir = os.path.join(base_path, "validation")
test_dir  = os.path.join(base_path, "test")

save_path = "/media/gledson/Dados1/Classificação_GastroIntestinal/classificar_original/convnextv2_pretrain/"
os.makedirs(save_path, exist_ok=True)

best_weights_path = os.path.join(save_path, "best_weights.pth")

# =========================================================
# DATA
# ConvNeXt V2 usa normalização ImageNet padrão
# =========================================================
norm = "gastrointestinal" # "padrao" # 
if norm=="padrao":
    print("Normalização padrão do modelo original")
    normalizacao_mean = (0.485, 0.456, 0.406)
    normalizacao_std  = (0.229, 0.224, 0.225)
else:
    print("Normalização gastrointestinal")
    normalizacao_mean = (0.60931299, 0.46328843, 0.3963334) # gastrointestinal_
    normalizacao_std  = (0.27045652, 0.23730823, 0.2296206) # gastrointestinal_

train_tfms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

val_tfms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

test_tfms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
val_ds   = datasets.ImageFolder(valid_dir, transform=val_tfms)
test_ds  = datasets.ImageFolder(test_dir,  transform=test_tfms)

train_loader = DataLoader(train_ds, batch_size=batch_size_train,
                          shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size_valid,
                          shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=1,
                          shuffle=False, num_workers=4, pin_memory=True)

print(f'Classes: {train_ds.class_to_idx}')
print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

# =========================================================
# ATTENTION POOLING
# ConvNeXt V2 retorna feature map 2D (B, C, H, W)
# precisamos converter para sequência antes do pooling
# =========================================================
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x pode chegar como (B, C, H, W) ou (B, N, C)
        if x.ndim == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        w = self.attn(x)
        w = torch.softmax(w, dim=1)
        return torch.sum(w * x, dim=1)


# =========================================================
# MODEL — ConvNeXt V2 Large 384
# convnextv2_large — pré-treinado com FCMAE + fine-tune ImageNet
# global_pool='' desativa o pooling interno para usar AttentionPooling
# =========================================================
class ConvNeXtV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'convnextv2_large',
            pretrained=True,
            num_classes=0,
            global_pool=''   # desativa pooling interno
        )
        dim = self.backbone.num_features   # 1536 para Large

        # FREEZE BACKBONE
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.pool = AttentionPooling(dim)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)   # (B, 1536, H, W)
        x = self.pool(x)       # (B, 1536)
        return self.head(x)


model = ConvNeXtV2Classifier().to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Parâmetros treináveis: {trainable:,} / {total:,} '
      f'({100 * trainable / total:.1f}%)')

# =========================================================
# LOSS + OPTIMIZER
# =========================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=weight_decay
)

# =========================================================
# SCHEDULER — warmup + cosine decay
# =========================================================
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# =========================================================
# AMP
# =========================================================
scaler = torch.amp.GradScaler("cuda")

# =========================================================
# MÉTRICAS
# =========================================================
def compute_metrics(y_true, y_prob):
    y_true    = np.array(y_true)
    y_prob    = np.array(y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    prauc     = average_precision_score(y_true, y_prob)
    preds     = (y_prob > 0.5).astype(int)
    acc       = (y_true == preds).mean()
    return acc, auc_score, prauc

# =========================================================
# TRAIN STATE
# =========================================================
best_prauc = 0
counter    = 0

train_losses, val_losses   = [], []
train_praucs, val_praucs   = [], []
val_aucs                   = []

# =========================================================
# LOOP DE TREINAMENTO
# =========================================================
print('\n================================================')
print('PRÉ-TREINO — ConvNeXt V2 Large (backbone congelado)')
print('================================================\n')

for epoch in range(num_epochs):

    model.train()
    train_loss = 0
    y_true_tr, y_prob_tr = [], []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_true_tr.extend(y.cpu().numpy())
        y_prob_tr.extend(probs)

    train_loss /= len(train_loader)
    _, tr_auc, tr_prauc = compute_metrics(y_true_tr, y_prob_tr)

    model.eval()
    val_loss = 0
    y_true_v, y_prob_v = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss   = criterion(logits, y)

            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_true_v.extend(y.cpu().numpy())
            y_prob_v.extend(probs)

    val_loss /= len(val_loader)
    val_acc, val_auc, val_prauc = compute_metrics(y_true_v, y_prob_v)

    scheduler.step()

    if val_prauc > best_prauc:
        best_prauc = val_prauc
        counter    = 0
        torch.save(model.state_dict(), best_weights_path)
        print(f'  → Melhor PR-AUC: {best_prauc:.4f} — pesos salvos')
    else:
        counter += 1

    if counter >= patience:
        print(f'\nEarly stop na época {epoch + 1}')
        break

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_praucs.append(tr_prauc)
    val_praucs.append(val_prauc)
    val_aucs.append(val_auc)

    print(f'Epoch {epoch + 1}/{num_epochs} | '
          f'Train Loss: {train_loss:.4f} PR-AUC: {tr_prauc:.4f} | '
          f'Val Loss: {val_loss:.4f} PR-AUC: {val_prauc:.4f} '
          f'AUC: {val_auc:.4f}')

# =========================================================
# CURVAS
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.title('Loss — Pré-treino ConvNeXt V2 Large')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_path, 'loss.png'), dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_praucs, label='Train PR-AUC')
plt.plot(val_praucs,   label='Val PR-AUC')
plt.plot(val_aucs,     label='Val ROC-AUC')
plt.title('AUC — Pré-treino ConvNeXt V2 Large')
plt.xlabel('Época')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_path, 'auc.png'), dpi=600, bbox_inches='tight')
plt.show()

print(f'\nTreinamento finalizado.')
print(f'Melhor PR-AUC: {best_prauc:.4f}')
print(f'Melhor AUC:    {max(val_aucs):.4f}')
print(f'Pesos salvos em: {best_weights_path}')

# =========================================================
# TESTE
# =========================================================
print('\n================================================')
print('AVALIAÇÃO NO TESTE')
print('================================================\n')

model = ConvNeXtV2Classifier().to(device)
model.load_state_dict(torch.load(best_weights_path, map_location=device))
model.eval()

y_true_test, y_prob_test, y_pred_test = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda"):
            logits = model(x)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_true_test.extend(y.cpu().numpy())
        y_prob_test.extend(probs)
        y_pred_test.extend(preds)

y_true_test = np.array(y_true_test)
y_prob_test = np.array(y_prob_test)
y_pred_test = np.array(y_pred_test)

class_names = list(test_ds.class_to_idx.keys())

print(classification_report(y_true_test, y_pred_test,
                             target_names=class_names, digits=4))

y_prob_all       = np.stack([1 - y_prob_test, y_prob_test], axis=1)
y_true_bin       = np.zeros((len(y_true_test), 2))
y_true_bin[:, 1] = y_true_test
y_true_bin[:, 0] = 1 - y_true_test

colors = ['steelblue', 'crimson']

plt.figure(figsize=(7, 6))
for i in range(2):
    fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_prob_all[:, i])
    roc_auc_val           = auc(fpr, tpr)
    youden_idx            = np.argmax(tpr - fpr)
    optimal_threshold     = thresholds[youden_idx]
    optimal_fpr           = fpr[youden_idx]
    optimal_tpr           = tpr[youden_idx]

    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'Classe "{class_names[i]}" (AUC = {roc_auc_val:.4f})')
    plt.scatter(optimal_fpr, optimal_tpr, color=colors[i], zorder=5, s=70,
                label=f'  Threshold = {optimal_threshold:.3f} '
                      f'| Sens={optimal_tpr:.3f} Espec={1 - optimal_fpr:.3f}')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Aleatório')
plt.xlabel('1 - Especificidade (FPR)', fontsize=11)
plt.ylabel('Sensibilidade (TPR)',       fontsize=11)
plt.title(f'Curva ROC — {class_names[0]} vs {class_names[1]}', fontsize=12)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'roc_duas_classes.png'),
            dpi=600, bbox_inches='tight')
plt.show()

cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão — Pré-treino ConvNeXt V2 Large')
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'confusion_matrix.png'),
            dpi=600, bbox_inches='tight')
plt.show()



print('\n================================================')
print('FINE TUNING — ConvNeXt V2 Large')
print('================================================\n')

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix,
                              roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import timm

# =========================================================
# CONFIG
# =========================================================
img_size          = 384
batch_size_train  = 8 # 23
batch_size_valid  = 4 # 11
num_classes       = 2
epochs_finetuning = 50
fine_tune_at      = 2    # estágios 0..1 congelados, 2..3 treináveis
                         # ConvNeXt V2 Large tem 4 estágios
                         # cada estágio tem blocos nomeados stages.X.blocks.Y
lr_finetuning     = 1e-5
weight_decay      = 1e-5
patience          = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# =========================================================
# PATHS
# =========================================================
base_path = "/media/gledson/Dados1/Classificação_GastroIntestinal/Dataset_Final/"
train_dir = os.path.join(base_path, "train")
valid_dir = os.path.join(base_path, "validation")
test_dir  = os.path.join(base_path, "test")

save_path             = "/media/gledson/Dados1/Classificação_GastroIntestinal/classificar_original/"
best_weights_pretrain = save_path + "convnextv2_pretrain/best_weights.pth"

save_path_finetuning    = save_path + "convnextv2_finetuning/"
os.makedirs(save_path_finetuning, exist_ok=True)

best_weights_finetuning = os.path.join(save_path_finetuning, "best_weights_finetuning.pth")

# =========================================================
# DATA
# =========================================================
norm = "gastrointestinal" # "padrao" # 

if norm=="padrao":
    print("Normalização padrão do modelo original")
    normalizacao_mean = (0.485, 0.456, 0.406)
    normalizacao_std  = (0.229, 0.224, 0.225)
else:
    print("Normalização gastrointestinal")
    normalizacao_mean = (0.60931299, 0.46328843, 0.3963334) # gastrointestinal_
    normalizacao_std  = (0.27045652, 0.23730823, 0.2296206) # gastrointestinal_

train_tfms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

val_tfms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

test_tfms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
val_ds   = datasets.ImageFolder(valid_dir, transform=val_tfms)
test_ds  = datasets.ImageFolder(test_dir,  transform=test_tfms)

train_loader = DataLoader(train_ds, batch_size=batch_size_train,
                          shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size_valid,
                          shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=1,
                          shuffle=False, num_workers=4, pin_memory=True)

print(f'Classes: {train_ds.class_to_idx}')
print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

# =========================================================
# ATTENTION POOLING
# =========================================================
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        if x.ndim == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        w = self.attn(x)
        w = torch.softmax(w, dim=1)
        return torch.sum(w * x, dim=1)


# =========================================================
# MODEL — mesma arquitetura do pré-treino
# =========================================================
class ConvNeXtV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'convnextv2_large',
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        dim = self.backbone.num_features

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.pool = AttentionPooling(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


# =========================================================
# CARREGAR PESOS DO PRÉ-TREINO
# =========================================================
print('\nCarregando pesos do pré-treino...')
model = ConvNeXtV2Classifier().to(device)
model.load_state_dict(torch.load(best_weights_pretrain, map_location=device))

# =========================================================
# FINE-TUNING — descongelar por estágio
#
# ConvNeXt V2 Large no timm tem estrutura:
#   backbone.stem          → entrada
#   backbone.stages.0      → estágio 0  (3 blocos)
#   backbone.stages.1      → estágio 1  (3 blocos)
#   backbone.stages.2      → estágio 2  (27 blocos) ← mais profundo
#   backbone.stages.3      → estágio 3  (3 blocos)
#   backbone.norm_pre      → LayerNorm final
#
# Com fine_tune_at=2: congela estágios 0 e 1,
# treina estágios 2 e 3 + norm_pre
# ConvNeXt usa LayerNorm (não BatchNorm) → pode descongelar sem restrição
# =========================================================
print(f'\nDescongelando estágios >= {fine_tune_at} do backbone ConvNeXt V2 Large...')

for name, param in model.backbone.named_parameters():
    unfreeze = False

    for i in range(fine_tune_at, 4):
        if f'stages.{i}.' in name:
            unfreeze = True
            break

    # Descongelar LayerNorm final
    if 'norm_pre' in name:
        unfreeze = True

    param.requires_grad = unfreeze

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Parâmetros treináveis: {trainable:,} / {total:,} '
      f'({100 * trainable / total:.1f}%)')

# =========================================================
# OPTIMIZER
# =========================================================
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr_finetuning,
    weight_decay=weight_decay
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
scaler    = torch.amp.GradScaler("cuda")

# =========================================================
# MÉTRICAS
# =========================================================
def compute_metrics(y_true, y_prob):
    y_true    = np.array(y_true)
    y_prob    = np.array(y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    prauc     = average_precision_score(y_true, y_prob)
    preds     = (y_prob > 0.5).astype(int)
    acc       = (y_true == preds).mean()
    return acc, auc_score, prauc

# =========================================================
# TREINAMENTO — FINE-TUNING
# =========================================================
best_prauc = 0
counter    = 0

train_losses, val_losses   = [], []
train_praucs, val_praucs   = [], []
val_aucs                   = []

print('\n================================================')
print('FINE-TUNING — ConvNeXt V2 Large')
print('================================================\n')

for epoch in range(epochs_finetuning):

    model.train()
    train_loss = 0
    y_true_tr, y_prob_tr = [], []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = model(x)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_true_tr.extend(y.cpu().numpy())
        y_prob_tr.extend(probs)

    train_loss /= len(train_loader)
    _, tr_auc, tr_prauc = compute_metrics(y_true_tr, y_prob_tr)

    model.eval()
    val_loss = 0
    y_true_v, y_prob_v = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss   = criterion(logits, y)

            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_true_v.extend(y.cpu().numpy())
            y_prob_v.extend(probs)

    val_loss /= len(val_loader)
    val_acc, val_auc, val_prauc = compute_metrics(y_true_v, y_prob_v)

    if val_prauc > best_prauc:
        best_prauc = val_prauc
        counter    = 0
        torch.save(model.state_dict(), best_weights_finetuning)
        print(f'  → Melhor PR-AUC: {best_prauc:.4f} — pesos salvos')
    else:
        counter += 1

    if counter >= patience:
        print(f'\nEarly stop na época {epoch + 1}')
        break

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_praucs.append(tr_prauc)
    val_praucs.append(val_prauc)
    val_aucs.append(val_auc)

    print(f'Epoch {epoch + 1}/{epochs_finetuning} | '
          f'Train Loss: {train_loss:.4f} PR-AUC: {tr_prauc:.4f} | '
          f'Val Loss: {val_loss:.4f} PR-AUC: {val_prauc:.4f} '
          f'AUC: {val_auc:.4f}')

# =========================================================
# CURVAS
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.title('Loss — Fine-tuning ConvNeXt V2 Large')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_path_finetuning, 'loss_finetuning.png'),
            dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_praucs, label='Train PR-AUC')
plt.plot(val_praucs,   label='Val PR-AUC')
plt.plot(val_aucs,     label='Val ROC-AUC')
plt.title('AUC — Fine-tuning ConvNeXt V2 Large')
plt.xlabel('Época')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_path_finetuning, 'auc_finetuning.png'),
            dpi=600, bbox_inches='tight')
plt.show()

print(f'\nMelhor PR-AUC (val): {best_prauc:.4f}')
print(f'Melhor ROC-AUC (val): {max(val_aucs):.4f}')

# =========================================================
# TESTE
# =========================================================
print('\n================================================')
print('AVALIAÇÃO NO TESTE')
print('================================================\n')
save_path             = "/media/gledson/Dados1/Classificação_GastroIntestinal/classificar_original/"
save_path_finetuning    = save_path + "convnextv2_finetuning/"
best_weights_finetuning = os.path.join(save_path_finetuning, "best_weights_finetuning.pth")
model = ConvNeXtV2Classifier().to(device)
model.load_state_dict(torch.load(best_weights_finetuning, map_location=device))
model.eval()

y_true_test, y_prob_test, y_pred_test = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda"):
            logits = model(x)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_true_test.extend(y.cpu().numpy())
        y_prob_test.extend(probs)
        y_pred_test.extend(preds)

y_true_test = np.array(y_true_test)
y_prob_test = np.array(y_prob_test)
y_pred_test = np.array(y_pred_test)

class_names = list(test_ds.class_to_idx.keys())

print(classification_report(y_true_test, y_pred_test,
                             target_names=class_names, digits=4))

y_prob_all       = np.stack([1 - y_prob_test, y_prob_test], axis=1)
y_true_bin       = np.zeros((len(y_true_test), 2))
y_true_bin[:, 1] = y_true_test
y_true_bin[:, 0] = 1 - y_true_test

import scipy.io as sio
test_labels = y_true_test
sio.savemat(save_path_finetuning + 'test_labels.mat', {'test_labels': test_labels})

predictions_test = y_prob_all
sio.savemat(save_path_finetuning + 'predictions_test.mat', {'predictions_test': predictions_test})

test_predict = y_pred_test 
sio.savemat(save_path_finetuning + 'test_predict.mat', {'test_predict': test_predict})


colors = ['steelblue', 'crimson']

plt.figure(figsize=(7, 6))
for i in range(2):
    fpr, tpr, thresholds = roc_curve(y_true_bin[:, i], y_prob_all[:, i])
    roc_auc_val           = auc(fpr, tpr)
    youden_idx            = np.argmax(tpr - fpr)
    optimal_threshold     = thresholds[youden_idx]
    optimal_fpr           = fpr[youden_idx]
    optimal_tpr           = tpr[youden_idx]

    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'Classe "{class_names[i]}" (AUC = {roc_auc_val:.4f})')
    plt.scatter(optimal_fpr, optimal_tpr, color=colors[i], zorder=5, s=70,
                label=f'  Threshold = {optimal_threshold:.3f} '
                      f'| Sens={optimal_tpr:.3f} Espec={1 - optimal_fpr:.3f}')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Aleatório')
plt.xlabel('1 - Especificidade (FPR)', fontsize=11)
plt.ylabel('Sensibilidade (TPR)',       fontsize=11)
plt.title(f'Curva ROC — {class_names[0]} vs {class_names[1]}', fontsize=12)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(save_path_finetuning, 'roc_duas_classes.png'),
            dpi=600, bbox_inches='tight')
plt.show()

cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão — Fine-tuning ConvNeXt V2 Large')
plt.tight_layout()
plt.savefig(os.path.join(save_path_finetuning, 'confusion_matrix.png'),
            dpi=600, bbox_inches='tight')
plt.show()
