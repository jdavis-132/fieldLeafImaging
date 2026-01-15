#!/usr/bin/env python3
"""
V3.1 NEBRASKA ONLY - Phenotype processing for Nebraska
Runs in parallel with the main Alabama/Georgia process
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from datetime import datetime
import sys

# Importar todas las funciones del script principal
sys.path.append('/home/preethi/leaf_segmentation_project/04_scripts')

# Copiar todas las funciones necesarias aquí
def calculate_leaf_width_corrected_v3_1(mask, original_image):
    """V2.9/V3.0 method - PROVEN, no changes"""
    height, width = mask.shape

    crop_px = 500
    if width < crop_px * 2 + 100:
        crop_px = max(100, width // 4)

    mask_cropped = mask[:, crop_px:-crop_px].copy()

    contours, _ = cv2.findContours(mask_cropped, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (center), (w, h), angle = rect

    leaf_width_px = min(w, h)
    leaf_length_px = max(w, h)

    # Visualization with RED box (5px thick)
    vis = original_image.copy()

    # Crop lines (yellow)
    cv2.line(vis, (crop_px, 0), (crop_px, height), (0, 255, 255), 3)
    cv2.line(vis, (width-crop_px, 0), (width-crop_px, height), (0, 255, 255), 3)

    # Bounding box (BRIGHT RED, 5px thick)
    box = cv2.boxPoints(rect)
    box[:, 0] += crop_px
    box = np.int32(box)
    cv2.drawContours(vis, [box], 0, (0, 0, 255), 5)

    return {
        'leaf_width_corrected_px': leaf_width_px,
        'leaf_length_px': leaf_length_px,
        'crop_px': crop_px
    }, vis


def detect_midrib_single_stage_v3_1(image, mask, s_max, v_min, score_threshold, stage):
    """V3.1 single stage detection"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    candidate_mask = (s < s_max) & (v > v_min) & (mask > 0)
    num_candidates_original = np.sum(candidate_mask > 0)

    if num_candidates_original < 50:
        return None, {'success': False, 'reason': f'insufficient_candidates_stage{stage}', 'stage': stage}

    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
    candidate_mask_closed = cv2.morphologyEx(
        candidate_mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        kernel_horizontal
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate_mask_closed, connectivity=8
    )

    if num_labels <= 1:
        return None, {'success': False, 'reason': f'no_components_stage{stage}', 'stage': stage}

    leaf_area = np.sum(mask > 0)
    image_height, image_width = mask.shape

    valid_components = []

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        aspect_ratio = h / max(w, 1)
        area_ratio = area / leaf_area
        horizontal_span = w / image_width

        component_mask_this = (labels == i).astype(np.uint8)
        midrib_pixels_hsv = hsv[component_mask_this > 0]

        if len(midrib_pixels_hsv) == 0:
            continue

        mean_V = np.mean(midrib_pixels_hsv[:, 2])
        mean_S = np.mean(midrib_pixels_hsv[:, 1])

        # Permissive pre-filters (V2.9 style)
        if (aspect_ratio > 0.50 or aspect_ratio < 0.05 or area < 200 or
            area_ratio > 0.30 or horizontal_span < 0.15):
            continue

        touches_top = np.any(component_mask_this[:100, :] > 0)
        touches_bottom = np.any(component_mask_this[-100:, :] > 0)
        if touches_top or touches_bottom:
            continue

        if mean_V > 180 or mean_S < 30:
            continue

        valid_components.append(i)

    if len(valid_components) == 0:
        return None, {'success': False, 'reason': f'no_valid_components_stage{stage}', 'stage': stage}

    # Graduated scoring
    best_component = None
    best_score = 0
    all_scores = []

    for i in valid_components:
        x, y, w, h, area = stats[i]

        aspect_ratio = h / max(w, 1)
        area_ratio = area / leaf_area
        horizontal_span = w / image_width
        vertical_span = h / image_height
        center_y = y + h/2
        vertical_offset = abs(center_y - image_height/2) / image_height

        component_mask_this = (labels == i).astype(np.uint8)
        midrib_pixels_hsv = hsv[component_mask_this > 0]

        mean_H = np.mean(midrib_pixels_hsv[:, 0])
        mean_S = np.mean(midrib_pixels_hsv[:, 1])
        mean_V = np.mean(midrib_pixels_hsv[:, 2])
        std_V = np.std(midrib_pixels_hsv[:, 2])

        score = 0
        score_breakdown = {}

        # Centering
        if vertical_offset < 0.08:
            score += 50
            score_breakdown['centering'] = 50
        elif vertical_offset < 0.15:
            score += 35
            score_breakdown['centering'] = 35
        elif vertical_offset < 0.20:
            score += 20
            score_breakdown['centering'] = 20
        elif vertical_offset < 0.30:
            score += 5
            score_breakdown['centering'] = 5
        else:
            score -= 15
            score_breakdown['centering'] = -15

        # Span
        if horizontal_span > 0.75:
            score += 50
        elif horizontal_span > 0.65:
            score += 40
        elif horizontal_span > 0.55:
            score += 30
        elif horizontal_span > 0.45:
            score += 20
        elif horizontal_span > 0.35:
            score += 10

        # Aspect ratio
        if 0.08 < aspect_ratio < 0.11:
            score += 60
        elif 0.07 < aspect_ratio < 0.13:
            score += 45
        elif 0.06 < aspect_ratio < 0.15:
            score += 30
        elif 0.05 < aspect_ratio < 0.20:
            score += 15
        else:
            score += 5

        # Solidity
        contours_comp, _ = cv2.findContours(component_mask_this, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solidity = 0
        if len(contours_comp) > 0:
            contour = max(contours_comp, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            if solidity > 0.88:
                score += 35
            elif solidity > 0.80:
                score += 25
            elif solidity > 0.70:
                score += 15
            elif solidity > 0.60:
                score += 5
            else:
                score -= 20

        # Color
        if 10 < mean_H < 35 and 60 < mean_S < 180 and 85 < mean_V < 150:
            score += 35
        elif 5 < mean_H < 40 and 50 < mean_S < 200 and 80 < mean_V < 160:
            score += 20
        elif mean_V > 100:
            score += 10

        if mean_V < 70:
            score -= 30
        elif mean_V < 80:
            score -= 15
        elif mean_V < 90:
            score -= 5

        if std_V < 18:
            score += 25
        elif std_V < 28:
            score += 15
        elif std_V < 38:
            score += 5

        if std_V > 50:
            score -= 25
        elif std_V > 40:
            score -= 15

        # Edge proximity
        left_region = component_mask_this[:, :100]
        right_region = component_mask_this[:, -100:]
        touches_left = np.any(left_region > 0)
        touches_right = np.any(right_region > 0)

        if touches_left and touches_right:
            score += 30
        elif touches_left or touches_right:
            score += 10

        if area_ratio > 0.25:
            score -= 30
        elif area_ratio > 0.20:
            score -= 15

        all_scores.append({
            'component_id': i,
            'score': score,
            'breakdown': score_breakdown,
            'vertical_offset': vertical_offset,
            'solidity': solidity,
            'horizontal_span': horizontal_span
        })

        if score > best_score:
            best_score = score
            best_component = i

    # Final validation
    if best_component is not None:
        best_data = next(s for s in all_scores if s['component_id'] == best_component)

        if best_data['vertical_offset'] > 0.30:
            return None, {'success': False, 'reason': f'best_not_centered', 'stage': stage}
        if best_data['solidity'] < 0.50:
            return None, {'success': False, 'reason': f'best_fragmented', 'stage': stage}
        if best_data['horizontal_span'] < 0.40:
            return None, {'success': False, 'reason': f'best_short_span', 'stage': stage}

    if best_score < score_threshold:
        return None, {'success': False, 'reason': f'score_below_threshold_stage{stage}', 'stage': stage}

    midrib_mask = np.zeros_like(mask, dtype=np.uint8)
    midrib_mask[labels == best_component] = 255

    x, y, w, h, area = stats[best_component]
    midrib_pixels_hsv = hsv[midrib_mask > 0]
    midrib_pixels_bgr = image[midrib_mask > 0]

    return midrib_mask, {
        'success': True,
        'stage': stage,
        'threshold_used': f'S<{s_max}_V>{v_min}',
        'score': best_score,
        'score_breakdown': best_data['breakdown'],
        'aspect_ratio': h / max(w, 1),
        'vertical_span': h / image_height,
        'horizontal_span': w / image_width,
        'vertical_offset': best_data['vertical_offset'],
        'solidity': best_data['solidity'],
        'midrib_bbox': [x, y, w, h],
        'color_H': np.mean(midrib_pixels_hsv[:, 0]),
        'color_S': np.mean(midrib_pixels_hsv[:, 1]),
        'color_V': np.mean(midrib_pixels_hsv[:, 2]),
        'color_R': np.mean(midrib_pixels_bgr[:, 2]),
        'color_G': np.mean(midrib_pixels_bgr[:, 1]),
        'color_B': np.mean(midrib_pixels_bgr[:, 0]),
    }


def detect_midrib_v3_1_multistage(image, mask):
    """V3.1 multi-stage detection"""
    # Stage 1
    midrib_mask, metadata = detect_midrib_single_stage_v3_1(image, mask, s_max=220, v_min=100, score_threshold=75, stage=1)
    if metadata['success']:
        return midrib_mask, metadata

    # Stage 2
    midrib_mask, metadata = detect_midrib_single_stage_v3_1(image, mask, s_max=255, v_min=80, score_threshold=70, stage=2)
    if metadata['success']:
        return midrib_mask, metadata

    # Stage 3
    midrib_mask, metadata = detect_midrib_single_stage_v3_1(image, mask, s_max=255, v_min=60, score_threshold=65, stage=3)
    if metadata['success']:
        return midrib_mask, metadata

    return None, {'success': False, 'reason': 'no_midrib_after_all_stages', 'stages_attempted': 3}


def calculate_midrib_width(midrib_mask):
    """V2.9 method"""
    contours, _ = cv2.findContours(midrib_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    midrib_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(midrib_contour)

    width_bbox_px = h
    area_px = np.sum(midrib_mask > 0)
    length_px = w
    width_area_px = area_px / length_px if length_px > 0 else 0

    widths_sampled = []
    for i in range(10):
        x_sample = x + (i+1) * w // 11
        if x_sample < midrib_mask.shape[1]:
            col_slice = midrib_mask[:, x_sample]
            white_pixels = np.where(col_slice > 0)[0]
            if len(white_pixels) > 0:
                width_at_point = white_pixels[-1] - white_pixels[0] + 1
                widths_sampled.append(width_at_point)

    width_sampled_px = np.mean(widths_sampled) if widths_sampled else width_bbox_px

    return {
        'midrib_width_bbox_px': width_bbox_px,
        'midrib_width_area_px': width_area_px,
        'midrib_width_sampled_px': width_sampled_px,
        'midrib_length_px': length_px,
        'midrib_area_px': area_px
    }


def save_midrib_outputs(midrib_mask, original_image, mask, output_base, image_name, location):
    """Save midrib outputs"""
    outputs = {}
    if midrib_mask is None:
        return outputs

    mask_dir = os.path.join(output_base, 'midrib_masks', location)
    image_dir = os.path.join(output_base, 'midrib_images', location)
    highlight_dir = os.path.join(output_base, 'midrib_highlighted', location)

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(highlight_dir, exist_ok=True)

    base_name = image_name.replace('.jpg', '')

    mask_path = os.path.join(mask_dir, f'{base_name}_midrib_mask.png')
    cv2.imwrite(mask_path, midrib_mask)
    outputs['midrib_mask_path'] = mask_path

    midrib_img = np.zeros_like(original_image)
    midrib_img[midrib_mask > 0] = original_image[midrib_mask > 0]
    img_path = os.path.join(image_dir, f'{base_name}_midrib.png')
    cv2.imwrite(img_path, midrib_img)
    outputs['midrib_image_path'] = img_path

    highlighted = original_image.copy()
    highlighted[mask == 0] = 0
    highlighted[midrib_mask > 0] = [0, 0, 255]
    highlight_path = os.path.join(highlight_dir, f'{base_name}_highlighted.png')
    cv2.imwrite(highlight_path, highlighted)
    outputs['midrib_highlighted_path'] = highlight_path

    return outputs


def create_visualization_v3_1(original_image, mask, leaf_width_vis,
                               candidate_mask_original, candidate_mask_closed,
                               midrib_mask, metadata, output_path):
    """V3.1 visualization"""
    h, w = original_image.shape[:2]
    display_h = 500
    scale = display_h / h
    display_w = int(w * scale)

    panel1 = cv2.resize(original_image, (display_w, display_h))
    panel2 = cv2.resize(leaf_width_vis, (display_w, display_h)) if leaf_width_vis is not None else panel1.copy()

    panel3 = original_image.copy()
    panel3[mask == 0] = 0
    panel3[candidate_mask_original > 0] = [0, 255, 255]
    panel3 = cv2.resize(panel3, (display_w, display_h))

    panel4 = original_image.copy()
    panel4[mask == 0] = 0
    panel4[candidate_mask_closed > 0] = [255, 255, 0]
    panel4 = cv2.resize(panel4, (display_w, display_h))

    panel5 = original_image.copy()
    panel5[mask == 0] = 0
    if midrib_mask is not None and metadata.get('success', False):
        panel5[midrib_mask > 0] = [0, 0, 255]
    panel5 = cv2.resize(panel5, (display_w, display_h))

    label_h = 80
    label_bg = np.ones((label_h, display_w, 3), dtype=np.uint8) * 250

    def make_label(text, color=(0, 0, 0), font_scale=0.8):
        lbl = label_bg.copy()
        cv2.putText(lbl, text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        return lbl

    lbl1 = make_label('Original Image')
    lbl2 = make_label('Leaf Width (RED box)')
    lbl3 = make_label('Candidates (YELLOW)')
    lbl4 = make_label('After Closing (CYAN)')

    if metadata.get('success', False):
        score = metadata.get('score', 0)
        stage = metadata.get('stage', 1)
        lbl5 = make_label(f'Midrib (Score: {score:.0f}, Stage: {stage})', (0, 150, 0))
    else:
        reason = metadata.get('reason', 'unknown')
        if len(reason) > 35:
            reason = reason[:32] + '...'
        lbl5 = make_label(f'Failed: {reason}', (0, 0, 200), 0.7)

    h_spacing = 20
    h_spacer = np.ones((display_h + label_h, h_spacing, 3), dtype=np.uint8) * 30

    row1 = np.hstack([
        np.vstack([lbl1, panel1]),
        h_spacer,
        np.vstack([lbl2, panel2]),
        h_spacer,
        np.vstack([lbl3, panel3])
    ])

    row2 = np.hstack([
        np.vstack([lbl4, panel4]),
        h_spacer,
        np.vstack([lbl5, panel5]),
        h_spacer,
        np.ones((display_h + label_h, display_w, 3), dtype=np.uint8) * 30
    ])

    v_spacing = 20
    v_spacer = np.ones((v_spacing, row1.shape[1], 3), dtype=np.uint8) * 30
    combined = np.vstack([row1, v_spacer, row2])

    title_h = 100
    title_bg = np.ones((title_h, combined.shape[1], 3), dtype=np.uint8) * 255
    image_name = metadata.get('image_name', 'Unknown')
    if len(image_name) > 65:
        image_name = image_name[:62] + '...'

    cv2.putText(title_bg, f'V3.1 NEBRASKA: {image_name}', (30, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    final = np.vstack([title_bg, combined])
    cv2.imwrite(output_path, final)


def main():
    print("\n" + "="*80)
    print("PHENOTYPE EXTRACTION V3.1 - NEBRASKA ONLY")
    print("="*80 + "\n")

    # Paths
    base_dir = '/home/preethi/leaf_segmentation_project/hybrid_segmentation_final'
    masks_dir = os.path.join(base_dir, 'final_masks')
    metadata_file = os.path.join(base_dir, 'metadata', 'hybrid_results.json')
    output_dir = '/home/preethi/leaf_segmentation_project/phenotype_extraction_v3_1_FINAL'

    # Load metadata
    print("Loading metadata...")
    with open(metadata_file, 'r') as f:
        all_metadata = json.load(f)

    # Filter to Nebraska only and successful
    nebraska_images = [item for item in all_metadata
                       if 'Nebraska' in item.get('location', '')
                       and item.get('success', False)]

    total_to_process = len(nebraska_images)

    print(f"Nebraska images to process: {total_to_process:,}")
    print(f"Expected runtime: ~{total_to_process * 3 / 3600:.1f} hours")
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...\n")

    # Track metrics
    results = []
    failure_modes = defaultdict(int)
    success_categories = defaultdict(int)
    stage_success = {1: 0, 2: 0, 3: 0}
    location_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'stage1': 0, 'stage2': 0, 'stage3': 0})

    start_time = datetime.now()

    for idx, item in enumerate(nebraska_images):
        if (idx + 1) % 100 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (idx + 1) / elapsed
            remaining = (total_to_process - idx - 1) / rate
            success_count = sum(1 for r in results if r['midrib_detected'])

            print(f"[{idx+1:5d}/{total_to_process}] "
                  f"Rate: {rate:.2f} img/s | ETA: {remaining/3600:.1f} hrs | "
                  f"Success: {success_count}/{idx+1} ({100*success_count/(idx+1):.1f}%)")

        location = item['location']
        location_stats[location]['total'] += 1

        # Get paths
        original_path = item['image_path']
        mask_path = item['final_mask_path']

        if not os.path.exists(original_path) or not os.path.exists(mask_path):
            continue

        # Load
        image = cv2.imread(original_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        # Process
        leaf_metrics, leaf_vis = calculate_leaf_width_corrected_v3_1(mask, image)
        if leaf_metrics is None:
            continue

        midrib_mask, midrib_meta = detect_midrib_v3_1_multistage(image, mask)

        midrib_width_metrics = None
        if midrib_meta['success']:
            midrib_width_metrics = calculate_midrib_width(midrib_mask)
            stage = midrib_meta.get('stage', 1)
            stage_success[stage] += 1
            location_stats[location][f'stage{stage}'] += 1
            location_stats[location]['success'] += 1

        midrib_outputs = {}
        if midrib_meta['success']:
            midrib_outputs = save_midrib_outputs(midrib_mask, image, mask, output_dir, item['image_name'], location)

        # Visualization
        vis_filename = f'{idx+1:05d}_{item["image_name"]}'
        vis_subdir = os.path.join(output_dir, 'visualizations', location)
        os.makedirs(vis_subdir, exist_ok=True)
        vis_path = os.path.join(vis_subdir, vis_filename)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if midrib_meta['success']:
            stage = midrib_meta['stage']
            if stage == 1:
                s_thresh, v_thresh = 220, 100
            elif stage == 2:
                s_thresh, v_thresh = 255, 80
            else:
                s_thresh, v_thresh = 255, 60
        else:
            s_thresh, v_thresh = 220, 100

        candidate_original = ((s < s_thresh) & (v > v_thresh) & (mask > 0)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        candidate_closed = cv2.morphologyEx(candidate_original, cv2.MORPH_CLOSE, kernel)

        create_visualization_v3_1(image, mask, leaf_vis, candidate_original, candidate_closed,
                                 midrib_mask, {**midrib_meta, 'image_name': item['image_name']}, vis_path)

        # Store result
        result = {
            'image_name': item['image_name'],
            'location': location,
            'segmentation_method': item.get('method_used', 'unknown'),
            **leaf_metrics,
            'midrib_detected': midrib_meta['success'],
        }

        if midrib_width_metrics:
            result.update(midrib_width_metrics)
        if midrib_outputs:
            result.update(midrib_outputs)

        if midrib_meta['success']:
            result.update({
                'detection_stage': midrib_meta['stage'],
                'threshold_used': midrib_meta['threshold_used'],
                'midrib_score': midrib_meta['score'],
                'midrib_aspect_ratio': midrib_meta['aspect_ratio'],
                'midrib_vertical_span': midrib_meta['vertical_span'],
                'midrib_horizontal_span': midrib_meta['horizontal_span'],
                'midrib_vertical_offset': midrib_meta['vertical_offset'],
                'midrib_solidity': midrib_meta.get('solidity', 0),
                'midrib_color_H': midrib_meta['color_H'],
                'midrib_color_S': midrib_meta['color_S'],
                'midrib_color_V': midrib_meta['color_V'],
                'midrib_color_R': midrib_meta['color_R'],
                'midrib_color_G': midrib_meta['color_G'],
                'midrib_color_B': midrib_meta['color_B'],
            })

            score = midrib_meta['score']
            if score >= 180:
                success_categories['excellent_180+'] += 1
            elif score >= 150:
                success_categories['very_good_150-179'] += 1
            elif score >= 120:
                success_categories['good_120-149'] += 1
            elif score >= 100:
                success_categories['acceptable_100-119'] += 1
            else:
                success_categories['marginal_75-99'] += 1
        else:
            result['failure_reason'] = midrib_meta['reason']
            failure_modes[midrib_meta['reason']] += 1

        results.append(result)

    # Save results
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'phenotype_data_NEBRASKA_v3_1.csv')
    df.to_csv(csv_path, index=False)

    total = len(results)
    successes = sum(1 for r in results if r['midrib_detected'])
    failures = total - successes

    print(f"\n{'='*80}")
    print("V3.1 NEBRASKA - COMPLETE")
    print(f"{'='*80}\n")

    print(f"Duration: {duration/3600:.2f} hours")
    print(f"Rate: {total/duration:.2f} images/second\n")

    print(f"NEBRASKA RESULTS:")
    print(f"  Total: {total:,}")
    print(f"  Success: {successes:,} ({100*successes/total:.1f}%)")
    print(f"  Failed: {failures:,} ({100*failures/total:.1f}%)\n")

    print(f"BY DEVICE:")
    for loc in sorted(location_stats.keys()):
        stats = location_stats[loc]
        if stats['total'] > 0:
            success_rate = 100 * stats['success'] / stats['total']
            print(f"  {loc}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

    print(f"\n✅ NEBRASKA PROCESSING COMPLETE")
    print(f"CSV saved: {csv_path}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
