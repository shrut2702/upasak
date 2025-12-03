from gliner import GLiNER
import re
import json
from typing import List
import yaml
import random
import string
import rstr
import exrex
from faker import Faker
import copy
from typing import Literal
import importlib.resources as pkg_resources

CONTROL_WHITESPACE = re.compile(r"[\t\n\r\f\v]")

def clean_control_whitespace(value: str) -> str:
    if value is None:
        return value
    cleaned = CONTROL_WHITESPACE.sub(" ", value)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()

class SyntheticDataGenerator:
    """
    A class to generate synthetic data for various PII categories.
    """
    def __init__(self):
        # Initialize Faker for different regions to get appropriate data
        self.faker_us = Faker('en_US')
        self.faker_in = Faker('en_IN')
        self.faker_uk = Faker('en_GB')
        # A generic instance for other cases
        self.faker_generic = Faker()

    def generate_fake_name(self):
        """Generates a realistic full name, randomly choosing a locale."""
        return random.choice([self.faker_us.name(), self.faker_in.name(), self.faker_uk.name()])

    def generate_fake_address(self):
        """Generates a full address, randomly choosing a locale."""
        return random.choice([self.faker_us.address(), self.faker_in.address(), self.faker_uk.address()])

    def generate_fake_organization(self):
        """Generates a fake company name."""
        return self.faker_generic.company()

    def generate_fake_email(self):
        """Generates a fake email address."""
        return self.faker_generic.email()

    def generate_fake_phone(self):
        """Generates a fake phone number, randomly choosing a locale."""
        return random.choice([self.faker_us.phone_number(), self.faker_in.phone_number()])
    
    def generate_fake_ssn(self):
        """Generates a fake US Social Security Number in the correct format."""
        return self.faker_us.ssn() # Faker has a provider for this!

    def generate_fake_license_number(self):
        """Generates a generic 15-character alphanumeric license number."""
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choices(chars, k=15))

    def generate_fake_passport_number(self):
        """Generates a generic 9-character alphanumeric passport number."""
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choices(chars, k=9))

    def generate_fake_api_token(self):
        """Generates a fake 32-character hex API token."""
        return ''.join(random.choices(string.hexdigits.lower(), k=32))
    
    def generate_fake_uri(self):
        """Generates a fake URI."""
        return self.faker_generic.uri()
    
    def generate_keep_original(self, original_value: str):
        """Keeps the original value."""
        return original_value

    def generate_fake_based_on_regex(self, regex: str = None):
        """Generates a fake value based on a regex."""
        if regex is None:
            return "[PSEUDONYMIZED]"
        try:
            max_length = 200
            s = exrex.getone(regex, flags=0, maxrepeat=6)
            if max_length is not None and len(s) > max_length:
                attempts = 5
                ok = False
                for _ in range(attempts):
                    s = exrex.getone(regex, flags=0, maxrepeat=6)
                    if len(s) <= max_length:
                        ok = True
                        break
                if not ok:
                    raise ValueError(f"Generated string too long (len={len(s)}) for pattern {regex!r}")
            return s
        except Exception:
            try:
                compiled = re.compile(regex, flags=0)
                s = rstr.xeger(compiled)
                if max_length is not None and len(s) > max_length:
                    attempts = 5
                    ok = False
                    for _ in range(attempts):
                        s = rstr.xeger(compiled)
                        if len(s) <= max_length:
                            ok = True
                            break
                    if not ok:
                        raise ValueError(f"Generated string too long (len={len(s)}) for pattern {regex!r}")
                return s
            except Exception as e:
                return "[PSEUDONYMIZED]"

class PIISanitizer:
    def __init__(self, is_nlp_based: bool = True):
        self.is_nlp_based = is_nlp_based
        if self.is_nlp_based:
            self.model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
            self.labels = ["location", "full address", "driver licence", "person name", "person", "company", "email", "passport number", "Social Security Number", "phone number"]
        with pkg_resources.files(__package__).joinpath("pii-detection-patterns.yaml").open("r") as f:
            self.patterns = yaml.safe_load(f)
        with pkg_resources.files(__package__).joinpath("pii-generator-mapping.yaml").open("r") as f:
            self.generator_mapping = yaml.safe_load(f)
        self.reverse_mapping = self._create_reverse_map()
        self.count = 0
    
    def _create_reverse_map(self):
        reverse_map = {}
        for category, details in self.generator_mapping.items():
            for source in details['sources']:
                reverse_map[source] = category
                
        return reverse_map
    
    def _regex_pii_detection(self, text: str, sample_id: int, text_key: str) -> List[dict]:
        detections = []
        for pattern_name, pattern_info in self.patterns.items():
            regex = re.compile(pattern_info["regex"])
            for m in regex.finditer(text):
                detections.append({
                    "sample_id": sample_id,
                    "text_key": text_key,
                    "pattern": pattern_name,
                    "match": m.group(),
                    "start": m.start(),
                    "end": m.end(),
                    "score": 1.0  
                })
        return detections
    
    def _nlp_pii_detection(self, text: str, sample_id: int, text_key: str) -> List[dict]: # implement chunking of the text if the text is too large
        detections = []
        entities = self.model.predict_entities(text, self.labels)
        for entity in entities:
            detections.append({
                "sample_id": sample_id,
                "text_key": text_key,
                "pattern": entity["label"],
                "match": entity["text"],
                "start": entity["start"],
                "end": entity["end"],
                "score": entity["score"]
            })
        return detections
    
    def _merge_pii_detections(self, all_detections: List[dict]) -> List[dict]:
        merged = []

        all_detections.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

        for det in all_detections:
            overlap = False
            for existing in merged:
                if not (det["end"] <= existing["start"] or det["start"] >= existing["end"]):
                    overlap = True
                    if (det["end"] - det["start"]) > (existing["end"] - existing["start"]):
                        merged.remove(existing)
                        merged.append(det)
                    elif (det["end"] - det["start"]) == (existing["end"] - existing["start"]):
                        if det.get("score", 0) > existing.get("score", 0):
                            merged.remove(existing)
                            merged.append(det)
                    break
            if not overlap:
                merged.append(det)

        merged.sort(key=lambda x: x["start"])
        return merged
    
    def _filter_uncertain_detections(self, detections: List[dict]) -> List[dict]:
        """Return detections that need human review."""
        return [d for d in detections if 0.5 <= d.get("score", 1.0) < 0.9]
    
    def detect_pii(self, text: str, sample_id: int = None, text_key: str = None) -> List[dict]:
        all_detections = []
        regex_detections = self._regex_pii_detection(text, sample_id, text_key)
        all_detections.extend(regex_detections)
        if self.is_nlp_based:
            nlp_detections = self._nlp_pii_detection(text, sample_id, text_key)
            all_detections.extend(nlp_detections)
        deduplicated_detections = self._merge_pii_detections(all_detections)
        certain_detections = [d for d in deduplicated_detections if d.get("score", 1.0) >= 0.9]
        uncertain_detections = self._filter_uncertain_detections(deduplicated_detections)

        return certain_detections, uncertain_detections
    
    def pii_action(self, text:str, detections: List[dict], action_type: str = "redact") -> str: 
        detections.sort(key=lambda x: x["start"], reverse=True)
        
        if action_type == "redact":
            placeholder = "[REDACTED]"
            for det in detections:
                text = text[:det["start"]] + placeholder + text[det["end"]:]
        elif action_type == "mask":
            for det in detections:
                placeholder = f"[{det['pattern'].upper()}]"
                length = det["end"] - det["start"]
                text = text[:det["start"]] + placeholder + text[det["end"]:]
        elif action_type == "pseudonymize":
            generator = SyntheticDataGenerator()
            for det in detections:
                normalized_category = self.reverse_mapping.get(det["pattern"])
                if normalized_category:
                    generator_name = self.generator_mapping[normalized_category]['generator']
                    if hasattr(generator, generator_name):
                        generator_function  = getattr(generator, generator_name)
                        if generator_name == 'generate_fake_based_on_regex':
                            pattern_info = self.patterns.get(det["pattern"], {})
                            regex = pattern_info.get("regex", None)
                            synth_value = generator_function(regex)
                        elif generator_name == 'generate_keep_original':
                            synth_value = generator_function(det["match"])
                        else:
                            synth_value = generator_function()
                        clean_synth_value = clean_control_whitespace(synth_value)
                        text = text[:det["start"]] + clean_synth_value + text[det["end"]:]
                    else:
                        # If no specific generator method, use a generic placeholder
                        placeholder = f"[{det['pattern'].upper()}]"
                        text = text[:det["start"]] + placeholder + text[det["end"]:]
                else:
                    # If no mapping found, use a generic placeholder
                    placeholder = f"[{det['pattern'].upper()}]"
                    text = text[:det["start"]] + placeholder + text[det["end"]:]
        else:
            raise ValueError("Not a valid action_type. Choose from ['redact', 'mask', 'pseudonymize'].")
        
        return text
    
    def _select_hitl_subset(self, detections: List[dict], hitl_ratio: float, max_hitl: int) -> List[dict]:
        """
        Select subset of samples for HITL.
        detections: list of dicts, each dict has "sample_id" in it.
        """
        total_samples = len(detections)
        target = int(total_samples * hitl_ratio)

        if max_hitl is not None:
            target = min(target, max_hitl)

        sorted_samples = sorted(
            detections,
            key=lambda x: x.get("score", 0.5),
            reverse=True
        )
        return sorted_samples[:target]

    def _organize_pii_detections(self, detections: List[dict], max_sample_id: int) -> dict:
        """
        Organize detections by sample_id and field path.
        
        Returns:
            {
                sample_id: {
                    field_path: [detection1, detection2, ...]
                }
            }
        """
        organized = {i: {} for i in range(max_sample_id)}
        for detection in detections:
            sample_id = detection.get("sample_id")
            field_path = detection.get("text_key", "")

            if sample_id not in organized:
                organized[sample_id] = {}

            if field_path not in organized[sample_id]:
                organized[sample_id][field_path] = []

            organized[sample_id][field_path].append(detection)
        
        return organized

    def _extract_text_fields(self, obj: list | dict, parent_path: str = "", sample_id: int = None):
        """
        Recursively extract all text fields from nested structures.
        
        Returns:
            List of tuples: (path, text_value)
            where path is a string like "messages[0].content" or "conversations[1].value"
        """
        text_fields = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{parent_path}.{key}" if parent_path else key
                if isinstance(value, str):
                    text_fields.append((current_path, value))
                elif isinstance(value, (dict, list)):
                    text_fields.extend(self._extract_text_fields(value, current_path, sample_id))
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                current_path = f"{parent_path}[{idx}]"
                if isinstance(item, str):
                    text_fields.append((current_path, item))
                elif isinstance(item, (dict, list)):
                    text_fields.extend(self._extract_text_fields(item, current_path, sample_id))
        
        return text_fields


    def _set_nested_value(self, obj: dict | list, path: str, value):
        """
        Set a value in a nested structure using a path like "messages[0].content"
        
        Parameters:
            obj: The root object (dict or list)
            path: String path like "messages[0].content" or "key.subkey"
            value: The value to set
        """
        parts = re.findall(r'(\w+)|\[(\d+)\]', path)
        parts = [p[0] if p[0] else int(p[1]) for p in parts]
        
        current = obj
        for part in parts[:-1]:
            if isinstance(part, int):
                current = current[part]
            else:
                current = current[part]

        final_key = parts[-1]
        if isinstance(final_key, int):
            current[final_key] = value
        else:
            current[final_key] = value


    def pipeline(self, records: List[dict], output_file: str = None, action_type: Literal["redact", "mask", "pseudonymize"] = "redact", 
                hitl_ratio: float = 1, max_hitl: int = 50, enable_hitl: bool = True, 
                hitl_fn=None, **kwargs) -> List[dict]:
        """
        Parameters:
            records: List[dict] - List of samples (any schema)
            output_file: str - Path to save the sanitized dataset
            action_type: str - 'redact', 'mask', or 'pseudonymize'
            hitl_ratio: float - percentage of total uncertain detections for human verification
            max_hitl: int - max number of uncertain detections for human verification
            enable_hitl: bool - Whether to use human-in-the-loop for uncertain detections
            hitl_fn: function - implements hitl logic
        
        Returns:
            List of processed records (maintaining original schema)

        Orchestrates PII detection + optional HITL + action on a dataset with any schema.
        
        Handles flat schemas like:
            {"text": "...", "label": "..."}
        
        And nested schemas like:
            {"messages": [{"role": "user", "content": "..."}]}
            {"conversations": [{"from": "user", "value": "..."}]}
        
        """   
        if "session_state" not in kwargs or "pii_detections" not in kwargs.get("session_state") or kwargs.get("session_state").pii_detections is None:
            detections = {"certain_detections": [], "uncertain_detections": []}
            for i, rec in enumerate(records):
                # Extract all text fields from this record
                text_fields = self._extract_text_fields(rec, sample_id=i)
                for field_path, text in text_fields:
                    if text and isinstance(text, str):
                        key_certain_detections, key_uncertain_detections = self.detect_pii(
                            text, sample_id=i, text_key=field_path
                        )
                        detections["certain_detections"].extend(key_certain_detections)
                        detections["uncertain_detections"].extend(key_uncertain_detections)
            if "session_state" in kwargs:
                kwargs["session_state"].pii_detections = detections
        else:
            detections = kwargs.get("session_state").pii_detections
        
        final_detections = detections["certain_detections"]
        if enable_hitl and hitl_fn:
            if detections["uncertain_detections"]:
                top_uncertain_detections = self._select_hitl_subset(
                    detections["uncertain_detections"], hitl_ratio=hitl_ratio, max_hitl=max_hitl
                )
                verified_detections = hitl_fn(top_uncertain_detections)
                final_detections += verified_detections
        
        max_id = len(records)
        organized_detections = self._organize_pii_detections(final_detections, max_id)
        
        processed_data = []
        for i, rec in enumerate(records):
            # Deep copy to avoid modifying original
            sanitized_rec = copy.deepcopy(rec)
            text_fields = self._extract_text_fields(rec, sample_id=i)
            for field_path, original_text in text_fields:
                if original_text and isinstance(original_text, str):
                    field_detections = organized_detections[i].get(field_path, [])
                    sanitized_text = self.pii_action(original_text, field_detections, action_type=action_type)
                    self._set_nested_value(sanitized_rec, field_path, sanitized_text)
            processed_data.append(sanitized_rec)
        
        if output_file:
            out_file_ext = output_file.split(".")[-1].lower()
            if out_file_ext == "jsonl":
                with open(output_file, "w", encoding="utf-8") as f:
                    for rec in processed_data:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
        return processed_data
