from paddleocr import PPStructureV3

pipeline = PPStructureV3()
# ocr = TableRecognitionPipelineV2(use_doc_orientation_classify=True) # Specify whether to use the document orientation classification model with use_doc_orientation_classify
# ocr = TableRecognitionPipelineV2(use_doc_unwarping=True) # Specify whether to use the text image unwarping module with use_doc_unwarping
# ocr = TableRecognitionPipelineV2(device="gpu") # Specify the device to use GPU for model inference
output = pipeline.predict("Screenshot 2025-08-05 125850.png")
for res in output:
    res.print() ## Print the predicted structured output
    res.save_to_img("./output/")
    res.save_to_xlsx("./output/")
    res.save_to_html("./output/")
    res.save_to_json("./output/")