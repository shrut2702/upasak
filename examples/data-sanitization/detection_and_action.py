from upasak.data_sanitization import PIISanitizer

sample_text = "You and I lives in Sydney and her email is pilot.xyz@yahoo.in and she went to New York"

sanitizer = PIISanitizer(is_nlp_based=False) # use True if you want to use hybrid approach, rule-based + NER model
certain_detections, uncertain_detections = sanitizer.detect_pii(text=sample_text)

detections = certain_detections # confidence score is greater than 0.9
# detections = certain_detections + uncertain_detections # you can include uncertain_detections also for sanitization, but uncertain_detections has confidence score of 0.5-0.9
sanitized_text = sanitizer.pii_action(text=sample_text, detections=detections)

print(f"Original text: {sample_text}")
print(f"Sanitized text: {sanitized_text}")