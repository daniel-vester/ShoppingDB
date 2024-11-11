from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import analyze_receipt as anz_rec
from datetime import datetime
import write_to_db as wtdb
from os import listdir
from os.path import join, isfile
import torch
import gc




path = r"C:\ShoppingDB\Images"

all_images = [join(path,f) for f in listdir(path) if isfile(join(path, f))]


batch_size = 10

for i in range(0, len(all_images), batch_size):
    print("Batch " + str((i/10) + 1))

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    batch_images = all_images[i:i + batch_size]

    for image in batch_images:
        purchase_id = image[image.rfind("\\") + 1 : image.rfind(".")]
        date_str = purchase_id[:8]
        date_obj = datetime.strptime(date_str, "%Y%m%d")
        formatted_date = date_obj.strftime("%Y-%m-%d")
        result = anz_rec.analyze_receipt(image, formatted_date, purchase_id, model, processor)
        wtdb.insert_purchase_to_db(result)

    del model
    del processor

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



