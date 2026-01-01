"""
YOLOv11 X-Ray Security System - Metrics Module
Performans metrikleri hesaplama fonksiyonları
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from collections import defaultdict


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    İki bounding box arasındaki IoU (Intersection over Union) hesapla
    
    Args:
        box1: [x1, y1, x2, y2] formatında box
        box2: [x1, y1, x2, y2] formatında box
    
    Returns:
        float: IoU değeri (0-1 arası)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Kesişim alanı
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    # Her box'ın alanı
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Birleşim alanı
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def calculate_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Average Precision (AP) hesapla
    
    Args:
        recall: Recall değerleri array
        precision: Precision değerleri array
    
    Returns:
        float: AP değeri
    """
    # Recall'u 0 ve 1 ile genişlet
    recall = np.concatenate(([0], recall, [1]))
    precision = np.concatenate(([0], precision, [0]))
    
    # Precision değerlerini monoton azalan yap
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # Recall değişimlerini bul
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    
    # AP hesapla
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return ap


def calculate_map(predictions: List[Dict], 
                  ground_truths: List[Dict],
                  iou_threshold: float = 0.5,
                  class_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Mean Average Precision (mAP) hesapla
    
    Args:
        predictions: Model tahminleri listesi
        ground_truths: Ground truth annotations listesi
        iou_threshold: IoU threshold değeri
        class_names: Sınıf isimleri listesi
    
    Returns:
        Dict: mAP ve sınıf bazında AP değerleri
    """
    if not class_names:
        # Tüm sınıfları topla
        all_classes = set()
        for gt in ground_truths:
            all_classes.update([box['class'] for box in gt.get('boxes', [])])
        class_names = sorted(list(all_classes))
    
    ap_per_class = {}
    
    for class_name in class_names:
        # Bu sınıf için tüm tahminleri ve GT'leri topla
        class_predictions = []
        class_ground_truths = defaultdict(list)
        
        for img_id, pred in enumerate(predictions):
            for box in pred.get('boxes', []):
                if box['class'] == class_name:
                    class_predictions.append({
                        'image_id': img_id,
                        'confidence': box['confidence'],
                        'bbox': box['bbox']
                    })
        
        for img_id, gt in enumerate(ground_truths):
            for box in gt.get('boxes', []):
                if box['class'] == class_name:
                    class_ground_truths[img_id].append(box['bbox'])
        
        if len(class_predictions) == 0:
            ap_per_class[class_name] = 0.0
            continue
        
        # Confidence'a göre sırala
        class_predictions = sorted(class_predictions, 
                                   key=lambda x: x['confidence'], 
                                   reverse=True)
        
        # TP, FP hesapla
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))
        total_gt = sum(len(boxes) for boxes in class_ground_truths.values())
        
        detected = defaultdict(set)
        
        for i, pred in enumerate(class_predictions):
            img_id = pred['image_id']
            pred_box = pred['bbox']
            
            gt_boxes = class_ground_truths.get(img_id, [])
            
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= iou_threshold and max_gt_idx not in detected[img_id]:
                tp[i] = 1
                detected[img_id].add(max_gt_idx)
            else:
                fp[i] = 1
        
        # Kümülatif toplamlar
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Recall ve Precision
        recall = tp_cumsum / (total_gt + 1e-16)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        
        # AP hesapla
        ap = calculate_ap(recall, precision)
        ap_per_class[class_name] = ap
    
    # mAP hesapla
    map_value = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
    
    return {
        'mAP': map_value,
        'AP_per_class': ap_per_class
    }


def calculate_precision_recall_f1(predictions: List[Dict],
                                   ground_truths: List[Dict],
                                   iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Precision, Recall ve F1-Score hesapla
    
    Args:
        predictions: Model tahminleri
        ground_truths: Ground truth annotations
        iou_threshold: IoU threshold
    
    Returns:
        Dict: Precision, Recall, F1 değerleri
    """
    tp = 0
    fp = 0
    fn = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred.get('boxes', [])
        gt_boxes = gt.get('boxes', [])
        
        matched_gt = set()
        
        for pred_box in pred_boxes:
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_box['class'] != pred_box['class']:
                    continue
                
                iou = calculate_iou(pred_box['bbox'], gt_box['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= iou_threshold and max_gt_idx not in matched_gt:
                tp += 1
                matched_gt.add(max_gt_idx)
            else:
                fp += 1
        
        fn += len(gt_boxes) - len(matched_gt)
    
    precision = tp / (tp + fp + 1e-16)
    recall = tp / (tp + fn + 1e-16)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def calculate_confusion_matrix(predictions: List[Dict],
                               ground_truths: List[Dict],
                               class_names: List[str],
                               iou_threshold: float = 0.5) -> np.ndarray:
    """
    Confusion matrix hesapla
    
    Args:
        predictions: Model tahminleri
        ground_truths: Ground truth annotations
        class_names: Sınıf isimleri
        iou_threshold: IoU threshold
    
    Returns:
        np.ndarray: Confusion matrix
    """
    n_classes = len(class_names)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Background sınıfı ekle
    confusion_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int32)
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred.get('boxes', [])
        gt_boxes = gt.get('boxes', [])
        
        matched_gt = set()
        
        for pred_box in pred_boxes:
            pred_class_idx = class_to_idx.get(pred_box['class'], -1)
            if pred_class_idx == -1:
                continue
            
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box['bbox'], gt_box['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            if max_iou >= iou_threshold and max_gt_idx != -1:
                gt_class_idx = class_to_idx.get(gt_boxes[max_gt_idx]['class'], -1)
                if gt_class_idx != -1:
                    confusion_matrix[gt_class_idx, pred_class_idx] += 1
                    matched_gt.add(max_gt_idx)
            else:
                # False positive (background)
                confusion_matrix[n_classes, pred_class_idx] += 1
        
        # False negatives
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                gt_class_idx = class_to_idx.get(gt_box['class'], -1)
                if gt_class_idx != -1:
                    confusion_matrix[gt_class_idx, n_classes] += 1
    
    return confusion_matrix


def save_metrics_report(metrics: Dict, 
                       output_path: str,
                       experiment_name: str = "experiment") -> None:
    """
    Metrikleri JSON dosyasına kaydet
    
    Args:
        metrics: Metrik dictionary
        output_path: Çıktı dosya yolu
        experiment_name: Deney adı
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'experiment_name': experiment_name,
        'metrics': metrics,
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Metrics raporu kaydedildi: {output_path}")


def compare_metrics(baseline_metrics: Dict,
                   new_metrics: Dict) -> Dict[str, float]:
    """
    İki metrik setini karşılaştır ve iyileştirme oranlarını hesapla
    
    Args:
        baseline_metrics: Baseline metrikler
        new_metrics: Yeni metrikler
    
    Returns:
        Dict: İyileştirme oranları
    """
    improvements = {}
    
    for key in baseline_metrics:
        if isinstance(baseline_metrics[key], (int, float)) and isinstance(new_metrics.get(key), (int, float)):
            baseline_val = baseline_metrics[key]
            new_val = new_metrics[key]
            
            if baseline_val != 0:
                improvement = ((new_val - baseline_val) / baseline_val) * 100
                improvements[f"{key}_improvement_%"] = improvement
            
            improvements[f"{key}_baseline"] = baseline_val
            improvements[f"{key}_new"] = new_val
    
    return improvements


if __name__ == "__main__":
    # Test fonksiyonları
    print("Metrics module test")
    
    # IoU test
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = calculate_iou(box1, box2)
    print(f"IoU: {iou:.3f}")



