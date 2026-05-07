import os
import sys
import hashlib
import json
import pandas as pd
import numpy as np
from django.utils import timezone
from django.conf import settings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGO_ROOT = os.path.join(PROJECT_ROOT, 'BackdoorDetection-main', 'BackdoorDetection-main')
INFLUENCE_ROOT = os.path.join(PROJECT_ROOT, 'Influence-Based-Glitch-Detection-main')
DEFEND_ROOT = os.path.join(PROJECT_ROOT, 'defend_framework-main')

for p in [ALGO_ROOT, INFLUENCE_ROOT, DEFEND_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


def run_detection(task_id):
    from datasets.models import DetectionTask, PoisonedRecord, Dataset

    task = None
    try:
        task = DetectionTask.objects.get(id=task_id)
        task.status = 'running'
        task.started_at = timezone.now()
        task.save()

        dataset = task.dataset
        file_path = dataset.file.file_path if dataset.file else None

        if not file_path or not os.path.exists(file_path):
            raise ValueError("数据文件不存在或路径无效")

        if not file_path.endswith('.csv'):
            raise ValueError("仅支持CSV格式数据集")

        df = pd.read_csv(file_path)
        task.total_samples = len(df)

        metadata = dataset.metadata or {}
        label_col = _find_column(df, metadata.get('target_column', ''), ['label', 'class', 'target', 'Label', 'Class'])
        text_col = _find_column(df, metadata.get('text_column', ''), ['text', 'content', 'sentence', 'input', 'Text', 'Content', 'document'])

        method_code = task.method.code
        parameters = task.parameters or {}

        detector = DetectionEngine(df, text_col, label_col, parameters)

        if method_code == 'onion':
            poisoned_indices = detector.detect_onion()
        elif method_code == 'strip':
            poisoned_indices = detector.detect_strip()
        elif method_code == 'backdoor_detect':
            poisoned_indices = detector.detect_backdoor()
        elif method_code == 'influence_mislabelled':
            poisoned_indices = detector.detect_influence_mislabelled()
        elif method_code == 'influence_anomaly':
            poisoned_indices = detector.detect_influence_anomaly()
        elif method_code == 'defend_certified':
            poisoned_indices = detector.detect_defend_certified()
        elif method_code == 'statistical':
            poisoned_indices = detector.detect_statistical()
        elif method_code == 'comprehensive':
            poisoned_indices = detector.detect_comprehensive()
        else:
            poisoned_indices = detector.detect_statistical()

        task.detected_poisoned = len(poisoned_indices)
        task.detection_rate = len(poisoned_indices) / task.total_samples if task.total_samples > 0 else 0
        task.status = 'completed'
        task.completed_at = timezone.now()
        task.duration_seconds = int((task.completed_at - task.started_at).total_seconds())

        severity_map = {}
        for item in poisoned_indices:
            idx = item['index']
            severity_map[idx] = item.get('severity', 'medium')

        raw_indices = [item['index'] for item in poisoned_indices]

        task.result_summary = {
            'method': method_code,
            'total_samples': task.total_samples,
            'detected_poisoned': len(raw_indices),
            'detection_rate': task.detection_rate,
            'label_column': label_col,
            'text_column': text_col,
            'parameters_used': parameters,
        }
        task.detailed_results = poisoned_indices[:5000]
        task.save()

        for item in poisoned_indices:
            idx = item['index']
            row_data = df.iloc[idx]
            row_dict = {}
            for col in df.columns:
                val = row_data[col]
                row_dict[col] = None if pd.isna(val) else str(val)

            row_str = ','.join(str(v) for v in row_dict.values())
            row_hash = hashlib.md5(row_str.encode('utf-8')).hexdigest()

            original_label = str(row_data[label_col]) if label_col and label_col in df.columns else ''
            severity = item.get('severity', 'medium')
            confidence = item.get('confidence', 0.5)
            trigger_pattern = item.get('trigger_pattern', '')
            poisoning_type = item.get('poisoning_type', method_code)
            attack_vector = item.get('attack_vector', '')
            predicted_label = item.get('predicted_label', '')
            feature_importance = item.get('feature_importance', {})

            PoisonedRecord.objects.create(
                task=task,
                dataset=dataset,
                row_index=int(idx),
                row_hash=row_hash,
                original_label=original_label,
                predicted_label=predicted_label,
                confidence=confidence,
                severity=severity,
                trigger_pattern=trigger_pattern,
                attack_vector=attack_vector,
                poisoning_type=poisoning_type,
                data_snapshot=row_dict,
                feature_importance=feature_importance,
            )

        dataset.poisoned_rows = dataset.poisoned_records.filter(
            verification_status__in=['detected', 'verified', 'confirmed']
        ).count()
        dataset.clean_rows = dataset.total_rows - dataset.poisoned_rows
        dataset.contamination_rate = dataset.poisoned_rows / dataset.total_rows if dataset.total_rows > 0 else 0
        dataset.save()

    except Exception as e:
        if task:
            task.status = 'failed'
            task.error_message = str(e)
            import traceback
            task.traceback = traceback.format_exc()
            task.completed_at = timezone.now()
            if task.started_at:
                task.duration_seconds = int((task.completed_at - task.started_at).total_seconds())
            task.save()


def _find_column(df, preferred, candidates):
    if preferred and preferred in df.columns:
        return preferred
    for col in candidates:
        if col in df.columns:
            return col
    return None


class DetectionEngine:
    def __init__(self, df, text_col, label_col, parameters=None):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.parameters = parameters or {}

    def detect_onion(self):
        return ONIONDetector(self.df, self.text_col, self.label_col, self.parameters).detect()

    def detect_strip(self):
        return STRIPDetector(self.df, self.text_col, self.label_col, self.parameters).detect()

    def detect_backdoor(self):
        return BackdoorDetector(self.df, self.text_col, self.label_col, self.parameters).detect()

    def detect_influence_mislabelled(self):
        return InfluenceMislabelledDetector(self.df, self.text_col, self.label_col, self.parameters).detect()

    def detect_influence_anomaly(self):
        return InfluenceAnomalyDetector(self.df, self.text_col, self.label_col, self.parameters).detect()

    def detect_defend_certified(self):
        return DefendCertifiedDetector(self.df, self.text_col, self.label_col, self.parameters).detect()

    def detect_statistical(self):
        return StatisticalDetector(self.df, self.text_col, self.label_col, self.parameters).detect()

    def detect_comprehensive(self):
        results_map = {}
        all_indices = set()

        detectors = {
            'onion': self.detect_onion,
            'strip': self.detect_strip,
            'backdoor': self.detect_backdoor,
            'statistical': self.detect_statistical,
        }

        for name, fn in detectors.items():
            try:
                results = fn()
                for item in results:
                    idx = item['index']
                    if idx not in results_map:
                        results_map[idx] = {}
                    results_map[idx][name] = item.get('confidence', 0.5)
                    all_indices.add(idx)
            except Exception:
                continue

        poisoned = []
        for idx in all_indices:
            scores = results_map[idx]
            vote_count = len(scores)
            avg_confidence = np.mean(list(scores.values()))
            combined = vote_count / len(detectors) * 0.6 + avg_confidence * 0.4

            if combined >= 0.3:
                trigger_parts = [f"{k}:{v:.2f}" for k, v in scores.items()]
                poisoned.append({
                    'index': idx,
                    'confidence': min(combined, 1.0),
                    'severity': 'critical' if vote_count >= 3 else ('high' if vote_count >= 2 else 'medium'),
                    'trigger_pattern': '; '.join(trigger_parts),
                    'poisoning_type': 'comprehensive',
                    'attack_vector': f'multi_method_vote({vote_count}/{len(detectors)})',
                    'feature_importance': scores,
                })

        return poisoned


class ONIONDetector:
    def __init__(self, df, text_col, label_col, parameters):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.ppl_threshold = parameters.get('ppl_threshold', 2.0)
        self.rare_word_threshold = parameters.get('rare_word_threshold', 0.0005)
        self.rare_char_threshold = parameters.get('rare_char_threshold', 0.0001)
        self.sample_size = min(parameters.get('sample_size', 5000), len(df))

    def detect(self):
        if not self.text_col or self.text_col not in self.df.columns:
            return self._detect_numeric_anomaly()

        df = self.df.sample(n=self.sample_size, random_state=42) if self.sample_size < len(self.df) else self.df
        texts = df[self.text_col].astype(str).tolist()

        word_freq = {}
        for text in texts:
            for word in text.split():
                w = word.lower()
                word_freq[w] = word_freq.get(w, 0) + 1

        total_words = sum(word_freq.values())
        rare_words = {w for w, c in word_freq.items() if c / total_words < self.rare_word_threshold and len(w) > 2}

        char_freq = {}
        for text in texts:
            for ch in text:
                char_freq[ch] = char_freq.get(ch, 0) + 1

        total_chars = sum(char_freq.values())
        rare_chars = {ch for ch, c in char_freq.items() if c / total_chars < self.rare_char_threshold and not ch.isalnum() and ch != ' '}

        text_lengths = [len(t.split()) for t in texts]
        mean_len = np.mean(text_lengths)
        std_len = np.std(text_lengths)

        label_ppl = {}
        if self.label_col and self.label_col in df.columns:
            for label, group in df.groupby(self.label_col):
                label_texts = group[self.text_col].astype(str).tolist()
                label_word_freq = {}
                for t in label_texts:
                    for w in t.split():
                        label_word_freq[w.lower()] = label_word_freq.get(w.lower(), 0) + 1
                label_total = sum(label_word_freq.values())
                label_ppl[label] = {
                    'word_freq': label_word_freq,
                    'total': label_total,
                    'mean_length': np.mean([len(t.split()) for t in label_texts]),
                    'std_length': np.std([len(t.split()) for t in label_texts]),
                }

        poisoned = []
        for pos, (idx, row) in enumerate(df.iterrows()):
            text = str(row[self.text_col])
            words = text.split()
            score = 0.0
            triggers = []

            rare_word_count = sum(1 for w in words if w.lower() in rare_words)
            if rare_word_count > 0 and len(words) > 0:
                rw_ratio = rare_word_count / len(words)
                score += rw_ratio * 10
                triggers.append(f'rare_words:{rare_word_count}')

            rare_char_count = sum(1 for c in text if c in rare_chars)
            if rare_char_count > 0:
                rc_ratio = rare_char_count / max(len(text), 1)
                score += rc_ratio * 15
                triggers.append(f'rare_chars:{rare_char_count}')

            if std_len > 0:
                z_len = abs(len(words) - mean_len) / std_len
                score += z_len * 0.5
                if z_len > 2:
                    triggers.append(f'length_z:{z_len:.1f}')

            if self.label_col and self.label_col in df.columns and row[self.label_col] in label_ppl:
                lp = label_ppl[row[self.label_col]]
                if lp['std_length'] > 0:
                    z_in_label = abs(len(words) - lp['mean_length']) / lp['std_length']
                    score += z_in_label * 0.3

            if score >= self.ppl_threshold:
                confidence = min(score / (self.ppl_threshold * 2), 1.0)
                poisoned.append({
                    'index': int(idx),
                    'confidence': confidence,
                    'severity': 'high' if score > self.ppl_threshold * 2 else 'medium',
                    'trigger_pattern': '; '.join(triggers) if triggers else 'ppl_anomaly',
                    'poisoning_type': 'backdoor',
                    'attack_vector': 'onion_ppl_anomaly',
                })

        return poisoned

    def _detect_numeric_anomaly(self):
        threshold = self.parameters.get('threshold', 3.0)
        poisoned = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            col_data = self.df[col].dropna()
            mean_val, std_val = col_data.mean(), col_data.std()
            if std_val > 0:
                for idx in self.df.index:
                    val = self.df.loc[idx, col]
                    if pd.notna(val):
                        z = abs((val - mean_val) / std_val)
                        if z > threshold:
                            poisoned.append({
                                'index': int(idx),
                                'confidence': min(z / (threshold * 2), 1.0),
                                'severity': 'high' if z > threshold * 2 else 'medium',
                                'trigger_pattern': f'{col}_zscore:{z:.1f}',
                                'poisoning_type': 'anomaly',
                                'attack_vector': 'numeric_outlier',
                            })
                            break
        return poisoned


class STRIPDetector:
    def __init__(self, df, text_col, label_col, parameters):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.frr = parameters.get('frr', 0.01)
        self.swap_ratio = parameters.get('swap_ratio', 0.5)
        self.repeat = parameters.get('repeat', 3)
        self.sample_size = min(parameters.get('sample_size', 5000), len(df))

    def detect(self):
        if not self.text_col or self.text_col not in self.df.columns:
            return self._detect_numeric_entropy()

        df = self.df.sample(n=self.sample_size, random_state=42) if self.sample_size < len(self.df) else self.df
        texts = df[self.text_col].astype(str).tolist()

        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1

        total_words = sum(word_freq.values())
        word_probs = {w: c / total_words for w, c in word_freq.items()}

        text_entropies = []
        for text in texts:
            words = text.split()
            if not words:
                text_entropies.append(0)
                continue
            probs = np.array([word_probs.get(w.lower(), 1e-10) for w in words])
            probs = probs / probs.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            text_entropies.append(entropy)

        text_entropies = np.array(text_entropies)
        mean_entropy = np.mean(text_entropies)
        std_entropy = np.std(text_entropies)

        perturbed_entropies = self._compute_perturbed_entropy(texts, word_probs)

        threshold_idx = int(len(text_entropies) * self.frr)
        threshold = np.sort(text_entropies)[threshold_idx] if threshold_idx < len(text_entropies) else mean_entropy - std_entropy

        poisoned = []
        for pos, (idx, row) in enumerate(df.iterrows()):
            orig_entropy = text_entropies[pos]
            pert_entropy = perturbed_entropies[pos] if pos < len(perturbed_entropies) else orig_entropy

            entropy_diff = abs(orig_entropy - pert_entropy)

            is_poisoned = False
            score = 0.0
            triggers = []

            if orig_entropy < threshold:
                is_poisoned = True
                score += 0.5
                triggers.append(f'low_entropy:{orig_entropy:.2f}')

            if std_entropy > 0:
                z_entropy = abs(orig_entropy - mean_entropy) / std_entropy
                if z_entropy > 2.0:
                    is_poisoned = True
                    score += z_entropy * 0.3
                    triggers.append(f'entropy_z:{z_entropy:.1f}')

            if entropy_diff < 0.1 and orig_entropy > mean_entropy + std_entropy:
                is_poisoned = True
                score += 0.3
                triggers.append('perturbation_resistant')

            if is_poisoned:
                confidence = min(score, 1.0)
                poisoned.append({
                    'index': int(idx),
                    'confidence': confidence,
                    'severity': 'high' if score > 0.8 else 'medium',
                    'trigger_pattern': '; '.join(triggers) if triggers else 'entropy_anomaly',
                    'poisoning_type': 'backdoor',
                    'attack_vector': 'strip_entropy_anomaly',
                })

        return poisoned

    def _compute_perturbed_entropy(self, texts, word_probs):
        all_words = list(word_probs.keys())
        perturbed = []
        for text in texts:
            words = text.split()
            if not words:
                perturbed.append(0)
                continue
            entropies = []
            for _ in range(self.repeat):
                m = int(len(words) * self.swap_ratio)
                perturbed_words = words.copy()
                swap_positions = np.random.randint(0, len(words), min(m, len(words)))
                for pos in swap_positions:
                    if all_words:
                        perturbed_words[pos] = np.random.choice(all_words)
                p = np.array([word_probs.get(w.lower(), 1e-10) for w in perturbed_words])
                p = p / p.sum()
                entropies.append(-np.sum(p * np.log2(p + 1e-10)))
            perturbed.append(np.mean(entropies))
        return perturbed

    def _detect_numeric_entropy(self):
        threshold = self.parameters.get('threshold', 2.0)
        poisoned = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            col_data = self.df[col].dropna()
            if len(col_data) < 10:
                continue
            hist, bin_edges = np.histogram(col_data, bins=min(20, len(col_data.unique())))
            probs = hist / hist.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            mean_val, std_val = col_data.mean(), col_data.std()
            if std_val > 0:
                for idx in self.df.index:
                    val = self.df.loc[idx, col]
                    if pd.notna(val):
                        z = abs((val - mean_val) / std_val)
                        if z > threshold:
                            poisoned.append({
                                'index': int(idx),
                                'confidence': min(z / (threshold * 2), 1.0),
                                'severity': 'high' if z > threshold * 2 else 'medium',
                                'trigger_pattern': f'{col}_z:{z:.1f}',
                                'poisoning_type': 'anomaly',
                                'attack_vector': 'numeric_entropy',
                            })
                            break
        return poisoned


class BackdoorDetector:
    def __init__(self, df, text_col, label_col, parameters):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.parameters = parameters
        self.pct_words_masked = parameters.get('pct_words_masked', 0.7)
        self.span_length = parameters.get('span_length', 2)
        self.n_perturbation = parameters.get('n_perturbation', 5)
        self.sample_size = min(parameters.get('sample_size', 3000), len(df))

    def detect(self):
        if not self.text_col or self.text_col not in self.df.columns:
            return self._detect_label_flip()

        df = self.df.sample(n=self.sample_size, random_state=42) if self.sample_size < len(self.df) else self.df
        texts = df[self.text_col].astype(str).tolist()

        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1
        total_words = sum(word_freq.values())

        label_text_profiles = {}
        if self.label_col and self.label_col in df.columns:
            for label, group in df.groupby(self.label_col):
                label_words = {}
                for text in group[self.text_col].astype(str):
                    for w in text.split():
                        label_words[w.lower()] = label_words.get(w.lower(), 0) + 1
                label_total = sum(label_words.values())
                label_text_profiles[label] = {
                    'word_freq': label_words,
                    'total': label_total,
                    'word_probs': {w: c / label_total for w, c in label_words.items()},
                }

        poisoned = []
        for pos, (idx, row) in enumerate(df.iterrows()):
            text = str(row[self.text_col])
            words = text.split()
            if not words:
                continue

            score = 0.0
            triggers = []

            perturbed_lls = self._perturb_and_score(text, word_freq, total_words)
            original_ll = self._compute_text_ll(text, word_freq, total_words)

            if original_ll > 0:
                ll_diffs = [abs(original_ll - pll) for pll in perturbed_lls]
                avg_diff = np.mean(ll_diffs)
                if avg_diff < 0.05:
                    score += 0.4
                    triggers.append('perturbation_insensitive')

            if self.label_col and self.label_col in df.columns and row[self.label_col] in label_text_profiles:
                label = row[self.label_col]
                lp = label_text_profiles[label]
                label_word_set = set(lp['word_freq'].keys())

                oov_words = [w.lower() for w in words if w.lower() not in label_word_set]
                if len(oov_words) > 0 and len(words) > 0:
                    oov_ratio = len(oov_words) / len(words)
                    if oov_ratio > 0.5:
                        score += oov_ratio * 0.5
                        triggers.append(f'oov_ratio:{oov_ratio:.2f}')

                for other_label, other_lp in label_text_profiles.items():
                    if other_label == label:
                        continue
                    other_word_set = set(other_lp['word_freq'].keys())
                    overlap = label_word_set & other_word_set
                    text_word_set = set(w.lower() for w in words)
                    trigger_words = text_word_set - overlap - (text_word_set - label_word_set)
                    cross_triggers = text_word_set & other_word_set - label_word_set
                    if len(cross_triggers) > 0:
                        score += 0.3
                        triggers.append(f'cross_label_triggers:{len(cross_triggers)}')

            if score >= 0.3:
                confidence = min(score, 1.0)
                poisoned.append({
                    'index': int(idx),
                    'confidence': confidence,
                    'severity': 'high' if score > 0.7 else 'medium',
                    'trigger_pattern': '; '.join(triggers) if triggers else 'backdoor_pattern',
                    'poisoning_type': 'backdoor',
                    'attack_vector': 'backdoor_trigger',
                })

        return poisoned

    def _compute_text_ll(self, text, word_freq, total_words):
        words = text.split()
        if not words:
            return 0
        ll = 0
        for w in words:
            p = word_freq.get(w.lower(), 1) / total_words
            ll += -np.log2(p + 1e-10)
        return ll / len(words)

    def _perturb_and_score(self, text, word_freq, total_words):
        words = text.split()
        if not words:
            return [0] * self.n_perturbation
        results = []
        all_words = list(word_freq.keys())
        for _ in range(self.n_perturbation):
            n_mask = int(len(words) * self.pct_words_masked)
            perturbed = words.copy()
            mask_positions = np.random.choice(len(words), min(n_mask, len(words)), replace=False)
            for pos in mask_positions:
                if all_words:
                    perturbed[pos] = np.random.choice(all_words)
            ll = self._compute_text_ll(' '.join(perturbed), word_freq, total_words)
            results.append(ll)
        return results

    def _detect_label_flip(self):
        if not self.label_col or self.label_col not in self.df.columns:
            return []
        return StatisticalDetector(self.df, self.text_col, self.label_col, self.parameters).detect()


class InfluenceMislabelledDetector:
    def __init__(self, df, text_col, label_col, parameters):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.contamination = parameters.get('contamination', 0.1)
        self.sample_size = min(parameters.get('sample_size', 5000), len(df))

    def detect(self):
        if not self.label_col or self.label_col not in self.df.columns:
            return StatisticalDetector(self.df, self.text_col, self.label_col, self.parameters).detect()

        df = self.df.sample(n=self.sample_size, random_state=42) if self.sample_size < len(self.df) else self.df

        label_groups = df.groupby(self.label_col)
        label_profiles = {}
        for label, group in label_groups:
            if self.text_col and self.text_col in df.columns:
                texts = group[self.text_col].astype(str).tolist()
                word_freq = {}
                for t in texts:
                    for w in t.split():
                        word_freq[w.lower()] = word_freq.get(w.lower(), 0) + 1
                total = sum(word_freq.values())
                label_profiles[label] = {
                    'word_freq': word_freq,
                    'total': total,
                    'word_probs': {w: c / total for w, c in word_freq.items()} if total > 0 else {},
                    'mean_length': np.mean([len(t.split()) for t in texts]),
                    'std_length': np.std([len(t.split()) for t in texts]),
                    'count': len(group),
                }
            else:
                label_profiles[label] = {'count': len(group)}

        poisoned = []
        for idx, row in df.iterrows():
            label = row[self.label_col]
            if label not in label_profiles:
                continue

            lp = label_profiles[label]
            score = 0.0
            triggers = []

            if self.text_col and self.text_col in df.columns:
                text = str(row[self.text_col])
                words = text.split()

                if lp['total'] > 0:
                    text_ll = 0
                    for w in words:
                        p = lp['word_probs'].get(w.lower(), 1e-10)
                        text_ll += -np.log2(p + 1e-10)
                    avg_ll = text_ll / max(len(words), 1)

                    all_avg_lls = []
                    for other_label, other_lp in label_profiles.items():
                        if 'word_probs' not in other_lp or not other_lp['word_probs']:
                            continue
                        other_ll = 0
                        for w in words:
                            p = other_lp['word_probs'].get(w.lower(), 1e-10)
                            other_ll += -np.log2(p + 1e-10)
                        all_avg_lls.append((other_label, other_ll / max(len(words), 1)))

                    if all_avg_lls:
                        all_avg_lls.sort(key=lambda x: x[1])
                        best_label, best_ll = all_avg_lls[0]
                        if best_label != label and best_ll < avg_ll * 0.8:
                            mislabel_score = (avg_ll - best_ll) / avg_ll
                            score += mislabel_score * 0.6
                            triggers.append(f'better_fit:{best_label}(ll={best_ll:.1f} vs {avg_ll:.1f})')

                if lp['std_length'] > 0:
                    z_len = abs(len(words) - lp['mean_length']) / lp['std_length']
                    if z_len > 2.5:
                        score += min(z_len / 5, 0.3)
                        triggers.append(f'length_outlier:{z_len:.1f}')

            if score >= 0.3:
                confidence = min(score, 1.0)
                predicted = ''
                for t in triggers:
                    if 'better_fit:' in t:
                        parts = t.split('(')
                        predicted = parts[0].replace('better_fit:', '')
                poisoned.append({
                    'index': int(idx),
                    'confidence': confidence,
                    'severity': 'high' if score > 0.7 else 'medium',
                    'trigger_pattern': '; '.join(triggers) if triggers else 'influence_mislabelled',
                    'poisoning_type': 'mislabelled',
                    'attack_vector': 'influence_nil',
                    'predicted_label': predicted,
                })

        return poisoned


class InfluenceAnomalyDetector:
    def __init__(self, df, text_col, label_col, parameters):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.contamination = parameters.get('contamination', 0.1)
        self.sample_size = min(parameters.get('sample_size', 5000), len(df))

    def detect(self):
        df = self.df.sample(n=self.sample_size, random_state=42) if self.sample_size < len(self.df) else self.df

        feature_matrix = self._build_feature_matrix(df)

        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)

        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
        )
        predictions = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.score_samples(X_scaled)

        score_threshold = np.percentile(anomaly_scores, self.contamination * 100)

        poisoned = []
        for pos, (idx, row) in enumerate(df.iterrows()):
            if predictions[pos] == -1:
                score = -anomaly_scores[pos]
                score_normalized = min(score / max(-score_threshold, 1e-10), 1.0)

                triggers = []
                if self.text_col and self.text_col in df.columns:
                    text = str(row[self.text_col])
                    triggers.append(f'text_len:{len(text)}')
                if self.label_col and self.label_col in df.columns:
                    triggers.append(f'label:{row[self.label_col]}')

                poisoned.append({
                    'index': int(idx),
                    'confidence': score_normalized,
                    'severity': 'high' if score_normalized > 0.7 else 'medium',
                    'trigger_pattern': '; '.join(triggers) if triggers else 'anomaly_detected',
                    'poisoning_type': 'anomaly',
                    'attack_vector': 'influence_pcid',
                })

        return poisoned

    def _build_feature_matrix(self, df):
        features = []

        if self.text_col and self.text_col in df.columns:
            texts = df[self.text_col].astype(str)
            features.append(texts.apply(len).values.reshape(-1, 1))
            features.append(texts.apply(lambda x: len(x.split())).values.reshape(-1, 1))
            features.append(texts.apply(lambda x: sum(1 for c in x if not c.isalnum() and c != ' ') / max(len(x), 1)).values.reshape(-1, 1))
            features.append(texts.apply(lambda x: len(set(x.split())) / max(len(x.split()), 1)).values.reshape(-1, 1))
            features.append(texts.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)).values.reshape(-1, 1))
            features.append(texts.apply(lambda x: sum(1 for c in x if c.isdigit()) / max(len(x), 1)).values.reshape(-1, 1))

            word_freq = {}
            for text in texts:
                for w in text.split():
                    word_freq[w.lower()] = word_freq.get(w.lower(), 0) + 1
            total_words = sum(word_freq.values())
            word_probs = {w: c / total_words for w, c in word_freq.items()}

            def text_entropy(x):
                words = x.split()
                if not words:
                    return 0
                probs = np.array([word_probs.get(w.lower(), 1e-10) for w in words])
                probs = probs / probs.sum()
                return -np.sum(probs * np.log2(probs + 1e-10))

            features.append(texts.apply(text_entropy).values.reshape(-1, 1))

        for col in df.select_dtypes(include=[np.number]).columns:
            features.append(df[col].fillna(0).values.reshape(-1, 1))

        if not features:
            features.append(np.zeros((len(df), 1)))

        return np.hstack(features)


class DefendCertifiedDetector:
    def __init__(self, df, text_col, label_col, parameters):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.confidence = parameters.get('confidence', 0.999)
        self.n_bags = parameters.get('n_bags', 100)
        self.sample_size = min(parameters.get('sample_size', 5000), len(df))

    def detect(self):
        df = self.df.sample(n=self.sample_size, random_state=42) if self.sample_size < len(self.df) else self.df

        if self.label_col and self.label_col in df.columns:
            return self._detect_with_bagging(df)
        return self._detect_anomaly_only(df)

    def _detect_with_bagging(self, df):
        labels = df[self.label_col].values
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)

        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        label_indices = np.array([label_to_idx[l] for l in labels])

        feature_matrix = self._build_features(df)

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)

        n_samples = len(df)
        bag_size = int(n_samples * 0.8)
        vote_matrix = np.zeros((n_samples, n_classes))

        for bag_idx in range(self.n_bags):
            sample_indices = np.random.choice(n_samples, bag_size, replace=True)
            X_bag = X_scaled[sample_indices]
            y_bag = label_indices[sample_indices]

            clf = DecisionTreeClassifier(max_depth=5, random_state=bag_idx)
            clf.fit(X_bag, y_bag)

            preds = clf.predict(X_scaled)
            for i in range(n_samples):
                vote_matrix[i, preds[i]] += 1

        poisoned = []
        for i in range(n_samples):
            majority = np.argmax(vote_matrix[i])
            top_1 = vote_matrix[i, majority]
            top_2 = np.sort(vote_matrix[i])[-2]
            true_label = label_indices[i]

            margin = (top_1 - top_2) / self.n_bags

            if majority != true_label:
                confidence = 1.0 - margin
                poisoned.append({
                    'index': int(df.index[i]),
                    'confidence': min(confidence, 1.0),
                    'severity': 'high',
                    'trigger_pattern': f'vote_margin:{margin:.3f}',
                    'poisoning_type': 'mislabelled',
                    'attack_vector': 'certified_bagging',
                    'predicted_label': str(unique_labels[majority]),
                })
            elif margin < 0.2:
                confidence = 1.0 - margin / 0.2
                poisoned.append({
                    'index': int(df.index[i]),
                    'confidence': min(confidence * 0.7, 0.7),
                    'severity': 'medium',
                    'trigger_pattern': f'low_margin:{margin:.3f}',
                    'poisoning_type': 'suspicious',
                    'attack_vector': 'certified_bagging',
                    'predicted_label': str(unique_labels[majority]),
                })

        return poisoned

    def _detect_anomaly_only(self, df):
        return InfluenceAnomalyDetector(df, self.text_col, self.label_col, self.parameters).detect()

    def _build_features(self, df):
        features = []
        if self.text_col and self.text_col in df.columns:
            texts = df[self.text_col].astype(str)
            features.append(texts.apply(len).values.reshape(-1, 1))
            features.append(texts.apply(lambda x: len(x.split())).values.reshape(-1, 1))
            features.append(texts.apply(lambda x: sum(1 for c in x if not c.isalnum() and c != ' ') / max(len(x), 1)).values.reshape(-1, 1))
        for col in df.select_dtypes(include=[np.number]).columns:
            features.append(df[col].fillna(0).values.reshape(-1, 1))
        if not features:
            features.append(np.zeros((len(df), 1)))
        return np.hstack(features)


class StatisticalDetector:
    def __init__(self, df, text_col, label_col, parameters):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.threshold = parameters.get('threshold', 0.7)
        self.sample_size = min(parameters.get('sample_size', 5000), len(df))

    def detect(self):
        df = self.df.sample(n=self.sample_size, random_state=42) if self.sample_size < len(self.df) else self.df
        poisoned = []

        if self.label_col and self.label_col in df.columns:
            label_counts = df[self.label_col].value_counts()
            total = len(df)
            label_ratios = label_counts / total
            rare_labels = label_counts[label_ratios < 0.05].index.tolist()

            if self.text_col and self.text_col in df.columns:
                text_lengths = df[self.text_col].astype(str).apply(len)
                mean_length = text_lengths.mean()
                std_length = text_lengths.std()

                for idx, row in df.iterrows():
                    score = 0.0
                    triggers = []

                    label = row[self.label_col]
                    if label in rare_labels:
                        score += 0.3
                        triggers.append('rare_label')

                    if std_length > 0:
                        text_len = len(str(row[self.text_col]))
                        z_score = abs((text_len - mean_length) / std_length)
                        if z_score > 3:
                            score += 0.4
                            triggers.append(f'length_z:{z_score:.1f}')

                    text = str(row[self.text_col]).lower()
                    words = text.split()
                    dup_words = len(words) - len(set(words))
                    if dup_words > 5:
                        score += 0.3
                        triggers.append(f'dup_words:{dup_words}')

                    if score >= self.threshold:
                        poisoned.append({
                            'index': int(idx),
                            'confidence': min(score, 1.0),
                            'severity': 'high' if score > 0.85 else 'medium',
                            'trigger_pattern': '; '.join(triggers) if triggers else 'statistical',
                            'poisoning_type': 'mixed',
                            'attack_vector': 'statistical_heuristic',
                        })
            else:
                for idx, row in df.iterrows():
                    label = row[self.label_col]
                    if label in rare_labels:
                        poisoned.append({
                            'index': int(idx),
                            'confidence': 0.6,
                            'severity': 'low',
                            'trigger_pattern': 'rare_label',
                            'poisoning_type': 'label_anomaly',
                            'attack_vector': 'statistical',
                        })
        else:
            poisoned = self._detect_numeric_only(df)

        return poisoned

    def _detect_numeric_only(self, df):
        threshold = self.parameters.get('z_threshold', 3.0)
        poisoned = []
        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col].dropna()
            mean_val, std_val = col_data.mean(), col_data.std()
            if std_val > 0:
                for idx in df.index:
                    val = df.loc[idx, col]
                    if pd.notna(val):
                        z = abs((val - mean_val) / std_val)
                        if z > threshold:
                            poisoned.append({
                                'index': int(idx),
                                'confidence': min(z / (threshold * 2), 1.0),
                                'severity': 'high' if z > threshold * 2 else 'medium',
                                'trigger_pattern': f'{col}_z:{z:.1f}',
                                'poisoning_type': 'anomaly',
                                'attack_vector': 'statistical_zscore',
                            })
                            break
        return poisoned


def generate_detection_report(task_id):
    from datasets.models import DetectionTask, PoisonedRecord

    task = DetectionTask.objects.get(id=task_id)
    records = PoisonedRecord.objects.filter(task=task)

    severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
    type_counts = {}
    for r in records:
        severity_counts[r.severity] = severity_counts.get(r.severity, 0) + 1
        type_counts[r.poisoning_type] = type_counts.get(r.poisoning_type, 0) + 1

    top_records = records.order_by('-confidence')[:20]
    top_details = []
    for r in top_records:
        top_details.append({
            'row_index': r.row_index,
            'original_label': r.original_label,
            'predicted_label': r.predicted_label,
            'confidence': r.confidence,
            'severity': r.severity,
            'trigger_pattern': r.trigger_pattern,
            'poisoning_type': r.poisoning_type,
            'data_snapshot': r.data_snapshot,
        })

    report = {
        'task_id': task.id,
        'task_name': task.name,
        'method': task.method.code,
        'method_name': task.method.name,
        'dataset': task.dataset.name,
        'status': task.status,
        'total_samples': task.total_samples,
        'detected_poisoned': task.detected_poisoned,
        'detection_rate': task.detection_rate,
        'duration_seconds': task.duration_seconds,
        'started_at': str(task.started_at) if task.started_at else None,
        'completed_at': str(task.completed_at) if task.completed_at else None,
        'severity_distribution': severity_counts,
        'type_distribution': type_counts,
        'top_poisoned_records': top_details,
        'result_summary': task.result_summary,
        'error_message': task.error_message if task.status == 'failed' else None,
    }

    return report
