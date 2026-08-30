import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from itertools import combinations
import scipy.io as sio

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

print('Save in late_fusion_post_hoc')
save_path = "/media/gledson/Dados1/Classificação_GastroIntestinal/classificar_original/late_fusion_post_hoc/fusion_3_model/prod_norm"

# =========================================================
# CAMINHOS — ajuste conforme o experimento
# =========================================================

local = ["convnextv2", "dinov2", "efficientnetv2m"]

for comb in combinations(local, 3):
    
    print("######################")
    
    print(comb)
    
    # =========================================================
    # CAMINHOS — ajuste conforme o experimento
    # =========================================================
    result_path_model_1 = '/media/gledson/Dados1/Classificação_GastroIntestinal/classificar_original/' + comb[0] + '_finetuning' + '/'
       
    path_test_labels_model_1      = result_path_model_1 + 'test_labels.mat'
    path_test_predict_model_1     = result_path_model_1 + 'test_predict.mat'
    path_predictions_test_model_1 = result_path_model_1 + 'predictions_test.mat'
    
    # =========================================================
    # CARREGAR ARQUIVOS .mat
    # =========================================================
    test_labels           = sio.loadmat(path_test_labels_model_1)['test_labels'].ravel().astype(int)
    test_predict_model_1     = sio.loadmat(path_test_predict_model_1)['test_predict'].ravel().astype(int)
    predictions_test_model_1 = sio.loadmat(path_predictions_test_model_1)['predictions_test']
    
    # predictions_test shape: (n_amostras, 2)
    # coluna 0 → score classe lesion (0)
    # coluna 1 → score classe normal (1)
    prob_lesion_model_1 = predictions_test_model_1[:, 0]
    prob_normal_model_1 = predictions_test_model_1[:, 1]
    
    class_names = ['lesion', 'normal']
    
    print(f'Total de amostras de teste : {len(test_labels)}')
    print(f'Classes verdadeiras únicas : {np.unique(test_labels)}')
    print(f'Classes preditas únicas    : {np.unique(test_predict_model_1)}')
    
    # =========================================================
    # CAMINHOS — ajuste conforme o experimento
    # =========================================================
    result_path_model_2 = '/media/gledson/Dados1/Classificação_GastroIntestinal/classificar_original/' + comb[1] + '_finetuning' + '/'
    
    path_test_labels_model_2      = result_path_model_2 + 'test_labels.mat'
    path_test_predict_model_2     = result_path_model_2 + 'test_predict.mat'
    path_predictions_test_model_2 = result_path_model_2 + 'predictions_test.mat'
    
    # =========================================================
    # CARREGAR ARQUIVOS .mat
    # =========================================================
    test_labels             = sio.loadmat(path_test_labels_model_2)['test_labels'].ravel().astype(int)
    test_predict_model_2     = sio.loadmat(path_test_predict_model_2)['test_predict'].ravel().astype(int)
    predictions_test_model_2 = sio.loadmat(path_predictions_test_model_2)['predictions_test']
    
    # predictions_test shape: (n_amostras, 2)
    # coluna 0 → score classe lesion (0)
    # coluna 1 → score classe normal (1)
    prob_lesion_model_2 = predictions_test_model_2[:, 0]
    prob_normal_model_2 = predictions_test_model_2[:, 1]
    
    class_names = ['lesion', 'normal']
    
    print(f'Total de amostras de teste : {len(test_labels)}')
    print(f'Classes verdadeiras únicas : {np.unique(test_labels)}')
    print(f'Classes preditas únicas    : {np.unique(test_predict_model_2)}')
    
    # =========================================================
    # CAMINHOS — ajuste conforme o experimento
    # =========================================================
    result_path_model_3 = '/media/gledson/Dados1/Classificação_GastroIntestinal/classificar_original/' + comb[2] + '_finetuning' + '/'
       
    path_test_labels_model_3      = result_path_model_3 + 'test_labels.mat'
    path_test_predict_model_3     = result_path_model_3 + 'test_predict.mat'
    path_predictions_test_model_3 = result_path_model_3 + 'predictions_test.mat'
    
    # =========================================================
    # CARREGAR ARQUIVOS .mat
    # =========================================================
    test_labels              = sio.loadmat(path_test_labels_model_3)['test_labels'].ravel().astype(int)
    test_predict_model_3     = sio.loadmat(path_test_predict_model_3)['test_predict'].ravel().astype(int)
    predictions_test_model_3 = sio.loadmat(path_predictions_test_model_3)['predictions_test']
    
    # predictions_test shape: (n_amostras, 2)
    # coluna 0 → score classe lesion (0)
    # coluna 1 → score classe normal (1)
    prob_lesion_model_3 = predictions_test_model_3[:, 0]
    prob_normal_model_3 = predictions_test_model_3[:, 1]
    
    class_names = ['lesion', 'normal']
    
    print(f'Total de amostras de teste : {len(test_labels)}')
    print(f'Classes verdadeiras únicas : {np.unique(test_labels)}')
    print(f'Classes preditas únicas    : {np.unique(test_predict_model_3)}')
    
    print("Fusion Post-Hoc")
    num = np.array([prob_lesion_model_1*prob_lesion_model_2*prob_lesion_model_3,prob_normal_model_1*prob_normal_model_2*prob_normal_model_3]).transpose()
    
    model_1 = np.array([prob_lesion_model_1,prob_normal_model_1]).transpose()
    model_1 = 1-model_1
    model_2 = np.array([prob_lesion_model_2,prob_normal_model_2]).transpose()
    model_2 = 1-model_2
    model_3 = np.array([prob_lesion_model_3,prob_normal_model_3]).transpose()
    model_3 = 1-model_3
    
    den = model_1*model_2*model_3
    
    fusion = num/(num+den)
    sio.savemat(save_path + '/predictions_test' + "_" + comb[0] + "_" + comb[1] + "_" + comb[2] +'.mat',
                {'predictions_test': fusion})
    
    # Máximo e seus índices ao longo das linhas (axis=1)
    argvalue = np.max(fusion, axis=1)
    test_predict = np.argmax(fusion, axis=1)
    sio.savemat(save_path + '/test_predict' + "_" + comb[0] + "_" + comb[1] + "_" + comb[2] +'.mat',
                {'test_predict': test_predict})
    
    sio.savemat(save_path + '/test_labels.mat',
                {'test_labels': test_labels})
    
    # =========================================================
    # MATRIZ DE CONFUSÃO GLOBAL
    # =========================================================
    cm = confusion_matrix(test_labels, test_predict)
    print(f'\nMatriz de Confusão (global):\n{cm}')
    
    # =========================================================
    # MÉTRICAS POR CLASSE (One-vs-Rest)
    # =========================================================
    def calcular_metricas_por_classe(y_true, y_pred, class_idx):
        y_true_bin = (y_true == class_idx).astype(int)
        y_pred_bin = (y_pred == class_idx).astype(int)
    
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin)
    
        TN = cm_bin[0, 0]
        FP = cm_bin[0, 1]
        FN = cm_bin[1, 0]
        TP = cm_bin[1, 1]
    
        total = TP + TN + FP + FN
    
        accuracy    = (TP + TN) / total if total > 0 else 0
        sensitivity = TP / (TP + FN)   if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP)   if (TN + FP) > 0 else 0
        precision   = TP / (TP + FP)   if (TP + FP) > 0 else 0
        fpr         = FP / (FP + TN)   if (FP + TN) > 0 else 0
        f1          = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        mcc         = matthews_corrcoef(y_true_bin, y_pred_bin)
        kappa       = cohen_kappa_score(y_true_bin, y_pred_bin)
    
        return {
            'TP':                          TP,
            'FP':                          FP,
            'FN':                          FN,
            'TN':                          TN,
            'Accuracy':                    accuracy,
            'Sensitivity (Recall/TPR)':    sensitivity,
            'Specificity':                 specificity,
            'Precision':                   precision,
            'FalsePositiveRate (FPR)':     fpr,
            'F1_score':                    f1,
            'MatthewsCorrelationCoef':     mcc,
            'Kappa':                       kappa,
        }
    
    
    metricas = {}
    for i, nome in enumerate(class_names):
        metricas[nome] = calcular_metricas_por_classe(test_labels, test_predict, i)
    
    print('\n' + '=' * 60)
    print('MÉTRICAS POR CLASSE')
    print('=' * 60)
    for nome, m in metricas.items():
        print(f'\n--- Classe "{nome}" (índice {class_names.index(nome)}) ---')
        for metrica, valor in m.items():
            if isinstance(valor, (int, np.integer)):
                print(f'  {metrica:<35}: {valor}')
            else:
                print(f'  {metrica:<35}: {valor:.4f}')
    
    # =========================================================
    # CURVA ROC + AUC POR CLASSE
    # =========================================================
    y_true_bin_lesion = (test_labels == 0).astype(int)
    y_true_bin_normal = (test_labels == 1).astype(int)
    
    # predictions_test shape: (n_amostras, 2)
    # coluna 0 → score classe lesion (0)
    # coluna 1 → score classe normal (1)
    prob_lesion = fusion[:, 0]
    prob_normal = fusion[:, 1]
    
    fpr_lesion, tpr_lesion, thr_roc_lesion = roc_curve(y_true_bin_lesion, prob_lesion)
    fpr_normal, tpr_normal, thr_roc_normal = roc_curve(y_true_bin_normal, prob_normal)
    
    auc_roc_lesion = auc(fpr_lesion, tpr_lesion)
    auc_roc_normal = auc(fpr_normal, tpr_normal)
    
    youden_lesion  = np.argmax(tpr_lesion - fpr_lesion)
    youden_normal  = np.argmax(tpr_normal - fpr_normal)
    thr_opt_roc_lesion = thr_roc_lesion[youden_lesion]
    thr_opt_roc_normal = thr_roc_normal[youden_normal]
    
    print('\n' + '=' * 60)
    print('CURVA ROC — AUC POR CLASSE')
    print('=' * 60)
    print(f'  Classe "lesion" — AUC ROC: {auc_roc_lesion:.4f} | Threshold ótimo (Youden): {thr_opt_roc_lesion:.4f}')
    print(f'    Sens={tpr_lesion[youden_lesion]:.4f} | Espec={1 - fpr_lesion[youden_lesion]:.4f}')
    print(f'  Classe "normal" — AUC ROC: {auc_roc_normal:.4f} | Threshold ótimo (Youden): {thr_opt_roc_normal:.4f}')
    print(f'    Sens={tpr_normal[youden_normal]:.4f} | Espec={1 - fpr_normal[youden_normal]:.4f}')
    
    colors = ['steelblue', 'crimson']
    
    plt.figure(figsize=(7, 6))
    for i, (nome, fpr_c, tpr_c, roc_auc_c, youden_idx, thr_roc_c) in enumerate([
        ('lesion', fpr_lesion, tpr_lesion, auc_roc_lesion, youden_lesion, thr_roc_lesion),
        ('normal', fpr_normal, tpr_normal, auc_roc_normal, youden_normal, thr_roc_normal),
    ]):
        plt.plot(fpr_c, tpr_c, color=colors[i], lw=2,
                 label=f'Classe "{nome}" (AUC = {roc_auc_c:.4f})')
        plt.scatter(fpr_c[youden_idx], tpr_c[youden_idx],
                    color=colors[i], zorder=5, s=70,
                    label=f'  Threshold = {thr_roc_c[youden_idx]:.3f}'
                          f' | Sens={tpr_c[youden_idx]:.3f}'
                          f' Espec={1 - fpr_c[youden_idx]:.3f}')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Aleatório')
    plt.xlabel('1 - Especificidade (FPR)', fontsize=11)
    plt.ylabel('Sensibilidade (TPR)',       fontsize=11)
    plt.title('Curva ROC por Classe' + " - " + comb[0] + "_" + comb[1] + '_' + comb[2], fontsize=12)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, comb[0] + '_' + comb[1] + '_' + comb[2] + '_' + 'ROC_por_classe.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # =========================================================
    # CURVA PR + AP + THRESHOLD ÓTIMO POR CLASSE
    # Threshold ótimo na curva PR = maximiza F1-score
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # =========================================================
    prec_lesion, rec_lesion, thr_pr_lesion = precision_recall_curve(y_true_bin_lesion, prob_lesion)
    prec_normal, rec_normal, thr_pr_normal = precision_recall_curve(y_true_bin_normal, prob_normal)
    
    ap_lesion = average_precision_score(y_true_bin_lesion, prob_lesion)
    ap_normal = average_precision_score(y_true_bin_normal, prob_normal)
    
    # precision_recall_curve retorna n+1 pontos mas apenas n thresholds
    # o último ponto (Precision=1, Recall=0) não tem threshold correspondente
    # por isso usamos [:-1] para alinhar os arrays
    def threshold_otimo_pr(precision, recall, thresholds):
        """
        Encontra o threshold que maximiza o F1-score na curva PR.
        precision e recall têm n+1 elementos; thresholds tem n.
        """
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / \
                    (precision[:-1] + recall[:-1] + 1e-8)
        idx_opt   = np.argmax(f1_scores)
        return thresholds[idx_opt], precision[idx_opt], recall[idx_opt], f1_scores[idx_opt], idx_opt
    
    thr_opt_pr_lesion, prec_opt_lesion, rec_opt_lesion, f1_opt_lesion, idx_lesion = \
        threshold_otimo_pr(prec_lesion, rec_lesion, thr_pr_lesion)
    
    thr_opt_pr_normal, prec_opt_normal, rec_opt_normal, f1_opt_normal, idx_normal = \
        threshold_otimo_pr(prec_normal, rec_normal, thr_pr_normal)
    
    print('\n' + '=' * 60)
    print('CURVA PR — AVERAGE PRECISION E THRESHOLD ÓTIMO POR CLASSE')
    print('=' * 60)
    print(f'  Classe "lesion" — AP: {ap_lesion:.4f} | Threshold ótimo (F1): {thr_opt_pr_lesion:.4f}')
    print(f'    Precision={prec_opt_lesion:.4f} | Recall={rec_opt_lesion:.4f} | F1={f1_opt_lesion:.4f}')
    print(f'  Classe "normal" — AP: {ap_normal:.4f} | Threshold ótimo (F1): {thr_opt_pr_normal:.4f}')
    print(f'    Precision={prec_opt_normal:.4f} | Recall={rec_opt_normal:.4f} | F1={f1_opt_normal:.4f}')
    
    plt.figure(figsize=(7, 6))
    
    # Classe lesion
    plt.plot(rec_lesion, prec_lesion, color=colors[0], lw=2,
             label=f'Classe "lesion" (AP = {ap_lesion:.4f})')
    plt.scatter(rec_opt_lesion, prec_opt_lesion,
                color=colors[0], zorder=5, s=70,
                label=f'  Threshold = {thr_opt_pr_lesion:.3f}'
                      f' | Prec={prec_opt_lesion:.3f}'
                      f' Rec={rec_opt_lesion:.3f}'
                      f' F1={f1_opt_lesion:.3f}')
    
    # Classe normal
    plt.plot(rec_normal, prec_normal, color=colors[1], lw=2,
             label=f'Classe "normal" (AP = {ap_normal:.4f})')
    plt.scatter(rec_opt_normal, prec_opt_normal,
                color=colors[1], zorder=5, s=70,
                label=f'  Threshold = {thr_opt_pr_normal:.3f}'
                      f' | Prec={prec_opt_normal:.3f}'
                      f' Rec={rec_opt_normal:.3f}'
                      f' F1={f1_opt_normal:.3f}')
    
    # Linhas de baseline
    baseline_lesion = y_true_bin_lesion.mean()
    baseline_normal = y_true_bin_normal.mean()
    plt.axhline(y=baseline_lesion, color=colors[0], lw=1, linestyle='--',
                alpha=0.4, label=f'  Baseline lesion = {baseline_lesion:.3f}')
    plt.axhline(y=baseline_normal, color=colors[1], lw=1, linestyle='--',
                alpha=0.4, label=f'  Baseline normal = {baseline_normal:.3f}')
    
    plt.xlabel('Recall',    fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.title('Curva PR por Classe' + " - " + comb[0] + "_" + comb[1] + "_" + comb[2], fontsize=12)
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, comb[0] + '_' + comb[1] + "_" + comb[2] + '_' + 'PR_por_classe.png'), dpi=600, bbox_inches='tight')
    plt.show()
    
    # =========================================================
    # SALVAR MÉTRICAS EM ARQUIVO TEXTO
    # =========================================================
    with open(os.path.join(save_path, comb[0] + '_' + comb[1] + '_' + comb[2] + '_' + 'metricas_por_classe.txt'), 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('MÉTRICAS POR CLASSE\n')
        f.write('=' * 60 + '\n')
    
        for nome, m in metricas.items():
            f.write(f'\n--- Classe "{nome}" ---\n')
            for metrica, valor in m.items():
                if isinstance(valor, (int, np.integer)):
                    f.write(f'  {metrica:<35}: {valor}\n')
                else:
                    f.write(f'  {metrica:<35}: {valor:.4f}\n')
    
        f.write('\n' + '=' * 60 + '\n')
        f.write('CURVA ROC\n')
        f.write('=' * 60 + '\n')
        f.write(f'  Classe "lesion" — AUC ROC: {auc_roc_lesion:.4f}'
                f' | Threshold ótimo (Youden): {thr_opt_roc_lesion:.4f}'
                f' | Sens: {tpr_lesion[youden_lesion]:.4f}'
                f' | Espec: {1 - fpr_lesion[youden_lesion]:.4f}\n')
        f.write(f'  Classe "normal" — AUC ROC: {auc_roc_normal:.4f}'
                f' | Threshold ótimo (Youden): {thr_opt_roc_normal:.4f}'
                f' | Sens: {tpr_normal[youden_normal]:.4f}'
                f' | Espec: {1 - fpr_normal[youden_normal]:.4f}\n')
    
        f.write('\n' + '=' * 60 + '\n')
        f.write('CURVA PR\n')
        f.write('=' * 60 + '\n')
        f.write(f'  Classe "lesion" — AP: {ap_lesion:.4f}'
                f' | Threshold ótimo (F1): {thr_opt_pr_lesion:.4f}'
                f' | Prec: {prec_opt_lesion:.4f}'
                f' | Rec: {rec_opt_lesion:.4f}'
                f' | F1: {f1_opt_lesion:.4f}\n')
        f.write(f'  Classe "normal" — AP: {ap_normal:.4f}'
                f' | Threshold ótimo (F1): {thr_opt_pr_normal:.4f}'
                f' | Prec: {prec_opt_normal:.4f}'
                f' | Rec: {rec_opt_normal:.4f}'
                f' | F1: {f1_opt_normal:.4f}\n')
    
    print(f'\nMétricas salvas em : {os.path.join(save_path, "metricas_por_classe.txt")}')
    print(f'Curva ROC salva em : {os.path.join(save_path, "roc_por_classe.png")}')
    print(f'Curva PR  salva em : {os.path.join(save_path, "pr_por_classe.png")}')