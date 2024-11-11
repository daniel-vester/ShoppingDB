from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import ast



def receipt_to_text(message, model, processor, max_new_tokens=1280):
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
        )
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]



def analyze_receipt(image, date, purchase_id, model, processor):


    message_get_supermarket = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": """This image is a receipt from a German Supermarket. 
                I want you to extract the following Information: 
                Brand of Supermarket
                Name of Supermarket
                Adress of supermarket
                Do not extract any other information like price or bought items.
                If the Brand and the Name are the same, still add both to the result.
                If only one of Brand or Name is given, still return both and give both the same value.
                If you cant see the Brand or name directly, try to discern it from other info on the receipt, like an email adress or an web-url.
                Do note that the receipt can be skewed or wrinkled. 
                Put this information prices into a python list. Separate them by , and enclose them with square brackets.
                Do not add any aditional text. The result should only be the array. Do not hallucinate.
                """},
            ],
        }
    ]

    message_get_items = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": """This image is a receipt from a German Supermarket. 
                I want you to extract the following Information: 
                Extract all bought items. 
                For each item create a new string with the following information:
                Name of Product,  Amount bought
                The amount for the product is on the line below it.
                Do note that the receipt can be skewed or wrinkled. 
                Also add Pfand as their own item.
                Read every single entry in the list, including the very last one.
                If only one instance of a product was bought, there is no amount given, so write a 1 into the Amount bought attribute.
                Separate these attributes by , and put a square bracket around each product. The result should be a
                python array of arrays. Do not add any aditional text. The result should only be the array of arrays. Do not hallucinate.
                """},
            ],
        }
    ]

    message_get_prices = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": """This image is a receipt from a German Supermarket. 
                I want you to extract the following Information: 
                Extract all prices that are written on the right column. If a price repeats do extract the repeated price. Ignore the per amount prices written 
                in the left column. 
                Do note that the receipt can be skewed or wrinkled. 
                Put all prices into a python list. Separate them by , and enclose them with square brackets.
                Also add the Price of Pfand
                Do not add any aditional text. The result should only be the array. Do not hallucinate.
                """},
            ],
        }
    ]

    output_get_supermarket = receipt_to_text(message_get_supermarket, model, processor)

    output_get_items = receipt_to_text(message_get_items, model, processor)

    output_get_items= output_get_items.replace("\n", "")
    output_get_items = output_get_items.replace("python", "")
    output_get_items = output_get_items.replace("'", "")
    output_get_items = output_get_items.replace("`", "")

    list_items= ast.literal_eval(output_get_items)

    output_get_prices = receipt_to_text(message_get_prices, model, processor)

    output_get_prices= output_get_prices.replace("\n", "")
    output_get_prices = output_get_prices.replace("python", "")
    output_get_prices = output_get_prices.replace("`", "")

    output_prices = ast.literal_eval(output_get_prices)



    for i in range(len(list_items)):
        if i < len(output_prices):
            list_items[i].append(output_prices[i])
        else:
            list_items[i].append(-9999)
    return_dict = {}

    return_dict["date"] = date
    return_dict["purchase_id"] = purchase_id
    return_dict["shop"] = output_get_supermarket
    return_dict["items"] = list_items

    return return_dict

