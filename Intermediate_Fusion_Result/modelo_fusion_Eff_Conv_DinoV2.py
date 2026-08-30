import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             classification_report, confusion_matrix, 
                             roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import timm

# Silencia o aviso de falta de token do Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# =========================================================
# PATHS
# =========================================================
base_path = "/media/gledson/Dados1/Classificação_GastroIntestinal/Dataset_Final/"
train_dir = os.path.join(base_path, "train")
valid_dir = os.path.join(base_path, "validation")
test_dir  = os.path.join(base_path, "test")

save_path = "/media/gledson/Dados1/Classificação_GastroIntestinal/classificar_original/"
save_path_fusion = os.path.join(save_path, "fusion_model_eff_conv_dinoV2/")
os.makedirs(save_path_fusion, exist_ok=True)
best_weights_fusion = os.path.join(save_path_fusion, "best_weights_fusion.pth")

# =========================================================
# CONFIGURAÇÕES GERAIS
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device detectado: {device}\n')

batch_size_train  = 8
batch_size_valid  = 4
batch_size_test   = 1
num_classes       = 2
epochs_finetuning = 200
patience          = 25  # Early stopping patience
lr_finetuning     = 1e-3
weight_decay      = 1e-5

# Normalização gastrointestinal personalizada
normalizacao_mean = (0.60931299, 0.46328843, 0.3963334)
normalizacao_std  = (0.27045652, 0.23730823, 0.2296206) 

# =========================================================
# TRANSFORMS PARA CADA RESOLUÇÃO
# =========================================================
train_tfms_effic = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

train_tfms_conv = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

train_tfms_dino = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

val_tfms_effic = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

val_tfms_conv = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

val_tfms_dino = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

test_tfms_effic = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

test_tfms_conv = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

test_tfms_dino = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(normalizacao_mean, normalizacao_std)
])

# =========================================================
# DATASET CUSTOMIZADO Triplo
# =========================================================
class DatasetTriploGastro(Dataset):
    def __init__(self, dir_caminho, transform_effic, transform_conv, transform_dino):
        self.base_dataset = datasets.ImageFolder(dir_caminho)
        self.transform_effic = transform_effic
        self.transform_conv  = transform_conv
        self.transform_dino  = transform_dino
        
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        path, label = self.base_dataset.samples[idx]
        img_original = self.base_dataset.loader(path)
        
        img_effic = self.transform_effic(img_original)
        img_conv  = self.transform_conv(img_original)
        img_dino  = self.transform_dino(img_original)
        
        return img_effic, img_conv, img_dino, label

# Instanciando DataLoaders Unificados
train_ds = DatasetTriploGastro(train_dir, train_tfms_effic, train_tfms_conv, train_tfms_dino)
val_ds   = DatasetTriploGastro(valid_dir, val_tfms_effic, val_tfms_conv, val_tfms_dino)
test_ds  = DatasetTriploGastro(test_dir, test_tfms_effic, test_tfms_conv, test_tfms_dino)

train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size_valid, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

# =========================================================
# POOLING DE ATENÇÃO ORIGINAL
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
            x = x.flatten(2).transpose(1, 2)
        w = self.attn(x)
        w = torch.softmax(w, dim=1)
        return torch.sum(w * x, dim=1)

# =========================================================
# BACKBONES ORIGINAIS
# =========================================================
class EfficientNetV2MClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_m',
            pretrained=True,
            num_classes=0,
            global_pool=''   # desativa pooling interno
        )
        dim = self.backbone.num_features   # 1280 para EfficientNetV2-M

        # FREEZE BACKBONE — pré-treino com base congelada
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
        x = self.backbone(x)   # (B, 1280, H, W)
        x = self.pool(x)       # (B, 1280)
        return self.head(x)

class ConvNeXtV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('convnextv2_large', pretrained=True, num_classes=0, global_pool='')
        dim = self.backbone.num_features
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

class DinoV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=0, dynamic_img_size=True)
        dim = self.backbone.num_features
        self.pool = AttentionPooling(dim)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.backbone.forward_features(x)
        if isinstance(x, tuple): x = x[0]
        if x.ndim == 3: x = self.pool(x)
        return self.head(x)

# Carregando pesos pré-treinados
weights_effic_path = os.path.join(save_path, "efficientnetv2m_finetuning/best_weights_finetuning.pth")
weights_conv_path  = os.path.join(save_path, "convnextv2_finetuning/best_weights_finetuning.pth")
weights_dino_path  = os.path.join(save_path, "dinov2_finetuning/best_weights_finetuning.pth")

model_effic = EfficientNetV2MClassifier().to(device)
model_effic.load_state_dict(torch.load(weights_effic_path, map_location=device, weights_only=True))

model_conv = ConvNeXtV2Classifier().to(device)
model_conv.load_state_dict(torch.load(weights_conv_path, map_location=device, weights_only=True))

model_dinov2 = DinoV2Classifier().to(device)
model_dinov2.load_state_dict(torch.load(weights_dino_path, map_location=device, weights_only=True))

# =========================================================
# MODELO MULTI-ENTRADA DE FUSÃO TARDIA (LATE FUSION)
# =========================================================
class LateFusionClassifier(nn.Module):
    def __init__(self, model_effic, model_conv, model_dinov2, num_classes=2):
        super().__init__()
        self.model_effic  = model_effic
        self.model_conv   = model_conv
        self.model_dinov2 = model_dinov2

        # Congelar todos os backbones — apenas novo_head é treinável
        for param in self.model_effic.parameters():
            param.requires_grad = False
        for param in self.model_conv.parameters():
            param.requires_grad = False
        for param in self.model_dinov2.parameters():
            param.requires_grad = False

        # 256 (EfficientNetV2-M) + 256 (ConvNeXtV2) + 256 (DINOv2)
        in_features_combinado = 256 + 256 + 256

        self.novo_head = nn.Sequential(
            nn.LayerNorm(in_features_combinado),  # estabiliza antes da classificação
            nn.GELU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features_combinado, num_classes)
        )

    def forward(self, x_effic, x_conv, x_dino):  # três entradas
        with torch.no_grad():

            # --- EfficientNetV2-M ---
            # backbone retorna (B, 1280, H, W) com global_pool=''
            feat_effic = self.model_effic.backbone(x_effic)
            feat_effic = self.model_effic.pool(feat_effic)      # (B, 1280)
            out_effic  = self.model_effic.head[0](feat_effic)   # LayerNorm
            out_effic  = self.model_effic.head[1](out_effic)    # Linear → (B, 256)

            # --- ConvNeXtV2 ---
            # backbone retorna (B, 1536, H, W) com global_pool=''
            feat_conv = self.model_conv.backbone(x_conv)
            feat_conv = self.model_conv.pool(feat_conv)          # (B, 1536)
            out_conv  = self.model_conv.head[0](feat_conv)       # LayerNorm
            out_conv  = self.model_conv.head[1](out_conv)        # Linear → (B, 256)

            # --- DINOv2 ---
            # forward_features retorna (B, N, 768) — tokens de atenção
            feat_dino = self.model_dinov2.backbone.forward_features(x_dino)
            if isinstance(feat_dino, tuple):
                feat_dino = feat_dino[0]
            if feat_dino.ndim == 3:
                feat_dino = self.model_dinov2.pool(feat_dino)   # (B, 768)
            out_dino  = self.model_dinov2.head[0](feat_dino)    # LayerNorm
            out_dino  = self.model_dinov2.head[1](out_dino)     # Linear → (B, 256)

        # Concatenar as três representações
        x_combinado = torch.cat((out_effic, out_conv, out_dino), dim=1)  # (B, 768)
        return self.novo_head(x_combinado)

modelo_fusao = LateFusionClassifier(model_effic, model_conv, model_dinov2, num_classes=num_classes).to(device)

# =========================================================
# OPTIMIZER / CRITERION / SCALER (ADAPTADO)
# =========================================================
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, modelo_fusao.parameters()),
    lr=lr_finetuning,
    weight_decay=weight_decay
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
scaler    = torch.amp.GradScaler("cuda")

# =========================================================
# MÉTRICAS ORIGINAL
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
# TREINAMENTO — ADAPTADO PARA MULTI-ENTRADA E PR-AUC
# =========================================================
best_prauc = 0
counter    = 0

train_losses, val_losses   = [], []
train_praucs, val_praucs   = [], []
val_aucs                   = []

print('\n================================================')
print('LATE FUSION TRAINING (EfficientNetV2M + ConvNeXtV2 + DinoV2)')
print('================================================\n')

best_prauc = 0
counter    = 0

train_losses, val_losses   = [], []
train_praucs, val_praucs   = [], []
val_aucs                   = []

criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, modelo_fusao.parameters()),
    lr=lr_finetuning, weight_decay=weight_decay)
scaler = torch.amp.GradScaler("cuda")

for epoch in range(epochs_finetuning):

    modelo_fusao.train()
    train_loss = 0
    y_true_tr, y_prob_tr = [], []

    for x_effic, x_conv, x_dino, y in train_loader:   # 4 elementos
        x_effic = x_effic.to(device)
        x_conv  = x_conv.to(device)
        x_dino  = x_dino.to(device)
        y       = y.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            logits = modelo_fusao(x_effic, x_conv, x_dino)
            loss   = criterion(logits, y)

        if not torch.isfinite(loss):
            print(f'  ⚠ Loss NaN/Inf na época {epoch+1} — abortando')
            break

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_true_tr.extend(y.cpu().numpy())
        y_prob_tr.extend(probs)

    train_loss /= len(train_loader)
    _, tr_auc, tr_prauc = compute_metrics(y_true_tr, y_prob_tr)

    # --- VALIDAÇÃO ---
    modelo_fusao.eval()
    val_loss = 0
    y_true_v, y_prob_v = [], []

    with torch.no_grad():
        for x_effic, x_conv, x_dino, y in val_loader:   # 4 elementos
            x_effic = x_effic.to(device)
            x_conv  = x_conv.to(device)
            x_dino  = x_dino.to(device)
            y       = y.to(device)

            with torch.amp.autocast("cuda"):
                logits = modelo_fusao(x_effic, x_conv, x_dino)
                loss   = criterion(logits, y)

            val_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_true_v.extend(y.cpu().numpy())
            y_prob_v.extend(probs)

    val_loss /= len(val_loader)

    if not np.all(np.isfinite(y_prob_v)):
        print(f'  ⚠ Probabilidades NaN na validação época {epoch+1} — abortando')
        break

    val_acc, val_auc, val_prauc = compute_metrics(y_true_v, y_prob_v)

    if val_prauc > best_prauc:
        best_prauc = val_prauc   # corrigido: lógica simplificada e correta
        counter    = 0
        torch.save(modelo_fusao.state_dict(), best_weights_fusion)
        print(f'  → Melhor PR-AUC: {best_prauc:.4f} — pesos salvos')
    else:
        counter += 1

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_praucs.append(tr_prauc)
    val_praucs.append(val_prauc)
    val_aucs.append(val_auc)

    print(f'Epoch {epoch + 1}/{epochs_finetuning} | '
          f'Train Loss: {train_loss:.4f} PR-AUC: {tr_prauc:.4f} | '
          f'Val Loss: {val_loss:.4f} PR-AUC: {val_prauc:.4f} '
          f'AUC: {val_auc:.4f}')

    if counter >= patience:
        print(f'\nEarly stop na época {epoch + 1}')
        break

print("\nTreinamento de Fusão Concluído!")

# =========================================================
# CURVAS DE TREINAMENTO
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.title('Loss — Late Fusion')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_path_fusion, 'loss_fusion.png'), dpi=600, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_praucs, label='Train PR-AUC')
plt.plot(val_praucs,   label='Val PR-AUC')
plt.plot(val_aucs,     label='Val ROC-AUC')
plt.title('AUC — Late Fusion')
plt.xlabel('Época')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_path_fusion, 'auc_fusion.png'), dpi=600, bbox_inches='tight')
plt.show()

# =========================================================
# TESTE — MODELO DE FUSÃO
# =========================================================
print('\n================================================')
print('AVALIAÇÃO NO TESTE — MODELO DE FUSÃO')
print('================================================\n')

# Corrigido: três modelos passados, igual ao treino
modelo_test_fusion = LateFusionClassifier(
    model_effic, model_conv, model_dinov2,
    num_classes=num_classes).to(device)

modelo_test_fusion.load_state_dict(
    torch.load(best_weights_fusion, map_location=device, weights_only=True))
modelo_test_fusion.eval()

y_true_test, y_prob_test, y_pred_test = [], [], []

with torch.no_grad():
    for x_effic, x_conv, x_dino, y in test_loader:   # 4 elementos
        x_effic = x_effic.to(device)
        x_conv  = x_conv.to(device)
        x_dino  = x_dino.to(device)
        y       = y.to(device)

        with torch.amp.autocast("cuda"):
            logits = modelo_test_fusion(x_effic, x_conv, x_dino)

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

# Salvar .mat
y_prob_all       = np.stack([1 - y_prob_test, y_prob_test], axis=1)
y_true_bin       = np.zeros((len(y_true_test), 2))
y_true_bin[:, 1] = y_true_test
y_true_bin[:, 0] = 1 - y_true_test

import scipy.io as sio
sio.savemat(save_path_fusion + 'test_labels.mat',
            {'test_labels': y_true_test})
sio.savemat(save_path_fusion + 'predictions_test.mat',
            {'predictions_test': y_prob_all})
sio.savemat(save_path_fusion + 'test_predict.mat',
            {'test_predict': y_pred_test})

# =========================================================
# CURVA ROC
# =========================================================
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
plt.title(f'Curva ROC — {class_names[0]} vs {class_names[1]} (Fusão)', fontsize=12)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(save_path_fusion, 'roc_duas_classes_fusion.png'),
            dpi=600, bbox_inches='tight')
plt.show()

# =========================================================
# MATRIZ DE CONFUSÃO
# =========================================================
cm = confusion_matrix(y_true_test, y_pred_test)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão — Late Fusion')
plt.tight_layout()
plt.savefig(os.path.join(save_path_fusion, 'confusion_matrix_fusion.png'),
            dpi=600, bbox_inches='tight')
plt.show()