"""
YOLOv11 X-Ray Security System - Visualization Module
Görselleştirme ve plotting fonksiyonları
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd


# Renk paleti (BGR format - OpenCV için)
COLOR_PALETTE = {
    'Baton': (138, 43, 226),      # Blue Violet
    'Bullet': (255, 105, 180),    # Hot Pink
    'Gun': (255, 0, 0),           # Blue
    'Hammer': (255, 0, 255),      # Magenta
    'HandCuffs': (255, 255, 0),   # Cyan
    'Knife': (0, 0, 255),         # Red
    'Lighter': (0, 165, 255),     # Orange
    'Pliers': (255, 0, 0),        # Blue
    'Powerbank': (42, 42, 165),   # Brown
    'Scissors': (255, 255, 0),    # Cyan
    'Sprayer': (0, 255, 255),     # Yellow
    'Wrench': (0, 255, 0),        # Green
}


def draw_bounding_boxes(image: np.ndarray,
                        boxes: List[Dict],
                        show_conf: bool = True,
                        show_label: bool = True,
                        line_thickness: int = 2,
                        font_scale: float = 0.5) -> np.ndarray:
    """
    Görüntü üzerine bounding box çiz
    
    Args:
        image: OpenCV görüntüsü (BGR)
        boxes: Box listesi [{'class': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}]
        show_conf: Confidence değerini göster
        show_label: Sınıf etiketini göster
        line_thickness: Çizgi kalınlığı
        font_scale: Font boyutu
    
    Returns:
        np.ndarray: Bounding box'lar çizilmiş görüntü
    """
    img = image.copy()
    
    for box in boxes:
        class_name = box['class']
        confidence = box.get('confidence', 0)
        x1, y1, x2, y2 = map(int, box['bbox'])
        
        # Renk seç
        color = COLOR_PALETTE.get(class_name, (0, 255, 0))
        
        # Box çiz
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
        
        # Label hazırla
        if show_label or show_conf:
            label = ""
            if show_label:
                label += class_name
            if show_conf:
                label += f" {confidence:.2f}" if show_label else f"{confidence:.2f}"
            
            # Label arka planı
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                   font_scale, line_thickness)
            y1_label = max(y1, label_size[1] + 10)
            
            cv2.rectangle(img, 
                         (x1, y1_label - label_size[1] - 10),
                         (x1 + label_size[0], y1_label + baseline - 10),
                         color, -1)
            
            # Label text
            cv2.putText(img, label, (x1, y1_label - 7), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       (255, 255, 255), line_thickness)
    
    return img


def visualize_predictions(image_path: str,
                          predictions: List[Dict],
                          output_path: Optional[str] = None,
                          show: bool = False) -> np.ndarray:
    """
    Model tahminlerini görselleştir
    
    Args:
        image_path: Görüntü dosya yolu
        predictions: Tahminler listesi
        output_path: Çıktı dosya yolu (None ise kaydetme)
        show: Görüntüyü ekranda göster
    
    Returns:
        np.ndarray: Görselleştirilmiş görüntü
    """
    # Görüntüyü oku
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Görüntü okunamadı: {image_path}")
    
    # Bounding box'ları çiz
    img_with_boxes = draw_bounding_boxes(img, predictions)
    
    # Kaydet
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img_with_boxes)
        print(f"[OK] Goruntu kaydedildi: {output_path}")
    
    # Göster
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Detections: {len(predictions)}")
        plt.tight_layout()
        plt.show()
    
    return img_with_boxes


def plot_confusion_matrix(confusion_matrix: np.ndarray,
                          class_names: List[str],
                          output_path: Optional[str] = None,
                          show: bool = False,
                          normalize: bool = True) -> None:
    """
    Confusion matrix çiz
    
    Args:
        confusion_matrix: Confusion matrix
        class_names: Sınıf isimleri
        output_path: Çıktı dosya yolu
        show: Grafiği göster
        normalize: Normalize et (yüzdelik)
    """
    # Normalize
    if normalize:
        cm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1, keepdims=True) + 1e-10)
    else:
        cm = confusion_matrix
    
    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', xticklabels=class_names + ['Background'],
                yticklabels=class_names + ['Background'],
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Confusion matrix kaydedildi: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(baseline_metrics: Dict,
                            new_metrics: Dict,
                            output_path: Optional[str] = None,
                            show: bool = False) -> None:
    """
    İki metrik setini karşılaştırmalı olarak göster
    
    Args:
        baseline_metrics: Baseline metrikler
        new_metrics: Yeni metrikler
        output_path: Çıktı dosya yolu
        show: Grafiği göster
    """
    # Ortak metrikleri bul
    common_metrics = set(baseline_metrics.keys()) & set(new_metrics.keys())
    numeric_metrics = [k for k in common_metrics 
                      if isinstance(baseline_metrics[k], (int, float))]
    
    if not numeric_metrics:
        print("⚠️ Karşılaştırılabilir metrik bulunamadı")
        return
    
    # Veri hazırla
    metrics_data = {
        'Metric': [],
        'Baseline': [],
        'New': [],
        'Improvement': []
    }
    
    for metric in numeric_metrics:
        baseline_val = baseline_metrics[metric]
        new_val = new_metrics[metric]
        improvement = ((new_val - baseline_val) / (baseline_val + 1e-10)) * 100
        
        metrics_data['Metric'].append(metric)
        metrics_data['Baseline'].append(baseline_val)
        metrics_data['New'].append(new_val)
        metrics_data['Improvement'].append(improvement)
    
    df = pd.DataFrame(metrics_data)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Karşılaştırma bar chart
    x = np.arange(len(df))
    width = 0.35
    
    ax1.bar(x - width/2, df['Baseline'], width, label='Baseline', alpha=0.8)
    ax1.bar(x + width/2, df['New'], width, label='New', alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Value')
    ax1.set_title('Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Metric'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # İyileştirme yüzdeleri
    colors = ['green' if x > 0 else 'red' for x in df['Improvement']]
    ax2.barh(df['Metric'], df['Improvement'], color=colors, alpha=0.7)
    ax2.set_xlabel('Improvement (%)')
    ax2.set_title('Performance Improvement')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='x', alpha=0.3)
    
    # Değerleri göster
    for i, v in enumerate(df['Improvement']):
        ax2.text(v, i, f' {v:+.1f}%', va='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Karşılaştırma grafiği kaydedildi: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_class_performance(ap_per_class: Dict[str, float],
                           class_counts: Optional[Dict[str, int]] = None,
                           output_path: Optional[str] = None,
                           show: bool = False) -> None:
    """
    Sınıf bazında performans grafiği
    
    Args:
        ap_per_class: Sınıf başına AP değerleri
        class_counts: Sınıf başına örnek sayıları
        output_path: Çıktı dosya yolu
        show: Grafiği göster
    """
    classes = list(ap_per_class.keys())
    ap_values = list(ap_per_class.values())
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # AP bar chart
    x = np.arange(len(classes))
    bars = ax1.bar(x, ap_values, alpha=0.7, label='AP')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Average Precision (AP)', color='b')
    ax1.set_title('Per-Class Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # Renklendirme (performansa göre)
    colors = ['green' if v >= 0.8 else 'orange' if v >= 0.6 else 'red' for v in ap_values]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Değerleri göster
    for i, v in enumerate(ap_values):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Eğer sınıf sayıları varsa, ikinci y ekseni
    if class_counts:
        ax2 = ax1.twinx()
        counts = [class_counts.get(c, 0) for c in classes]
        ax2.plot(x, counts, 'r--o', label='Sample Count', alpha=0.6)
        ax2.set_ylabel('Sample Count', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sınıf performans grafiği kaydedildi: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_precision_recall_curve(precision: np.ndarray,
                                recall: np.ndarray,
                                ap: float,
                                class_name: str = "All Classes",
                                output_path: Optional[str] = None,
                                show: bool = False) -> None:
    """
    Precision-Recall curve çiz
    
    Args:
        precision: Precision değerleri
        recall: Recall değerleri
        ap: Average Precision değeri
        class_name: Sınıf adı
        output_path: Çıktı dosya yolu
        show: Grafiği göster
    """
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.fill_between(recall, precision, alpha=0.2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {class_name}\nAP = {ap:.3f}')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ PR curve kaydedildi: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


def create_detection_grid(image_paths: List[str],
                          predictions_list: List[List[Dict]],
                          titles: Optional[List[str]] = None,
                          output_path: Optional[str] = None,
                          show: bool = False,
                          grid_size: Tuple[int, int] = None) -> None:
    """
    Birden fazla tespit sonucunu grid layout'ta göster
    
    Args:
        image_paths: Görüntü dosya yolları
        predictions_list: Her görüntü için tahminler
        titles: Her görüntü için başlık
        output_path: Çıktı dosya yolu
        show: Grafiği göster
        grid_size: (rows, cols) tuple, None ise otomatik
    """
    n_images = len(image_paths)
    
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, (img_path, predictions) in enumerate(zip(image_paths, predictions_list)):
        if idx >= len(axes):
            break
        
        img = cv2.imread(str(img_path))
        img_with_boxes = draw_bounding_boxes(img, predictions)
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img_rgb)
        axes[idx].axis('off')
        
        title = titles[idx] if titles and idx < len(titles) else f"Image {idx+1}"
        axes[idx].set_title(f"{title}\n({len(predictions)} detections)")
    
    # Boş subplot'ları gizle
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Detection grid kaydedildi: {output_path}")
    
    if show:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Visualization module test")


