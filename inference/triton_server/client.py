import tritonclient.http as http_client
from tritonclient.utils import *
import numpy as np
import time
ENABLE_SSL = False
ENDPOINT_URL = 'localhost:8000'
HTTP_HEADERS = {"Authorization": "Bearer __PASTE_KEY_HERE__"}

# Connect to the server
if ENABLE_SSL:
    import gevent.ssl
    triton_http_client = http_client.InferenceServerClient(
        url=ENDPOINT_URL, verbose=False,
        ssl=True, ssl_context_factory=gevent.ssl._create_default_https_context,
    )
else:
    triton_http_client = http_client.InferenceServerClient(
        url=ENDPOINT_URL, verbose=False,
    )

print("Is server ready - {}".format(triton_http_client.is_server_ready(headers=HTTP_HEADERS)))

def get_string_tensor(string_values, tensor_name):
    string_obj = np.array(string_values, dtype="object")
    input_obj = http_client.InferInput(tensor_name, string_obj.shape, np_to_triton_dtype(string_obj.dtype))
    input_obj.set_data_from_numpy(string_obj)
    return input_obj

def get_translation_input_for_triton(texts: list, src_lang: str, tgt_lang: str):
    return [
        get_string_tensor([[text] for text in texts], "INPUT_TEXT"),
        get_string_tensor([[src_lang]] * len(texts), "INPUT_LANGUAGE_ID"),
        get_string_tensor([[tgt_lang]] * len(texts), "OUTPUT_LANGUAGE_ID"),
    ]

# Prepare input and output tensors
input_sentences = ["I want to do homework today","I will apply on the celebaltech.com, please share the document.", "शाम को पार्क में चलने चलेंगे?","यह किताब बहुत दिलचस्प लग रही है।","main aaj ghar jaanaa chaahata hoon", "main ack gaadi khareedoonga."]

def test(input, target):
    if not input:  # Handle empty or None values
        return None, None
    inputs = get_translation_input_for_triton([input], "hin", target)
    output0 = http_client.InferRequestedOutput("OUTPUT_TEXT")
    start_time = time.time()
    # Send request
    response = triton_http_client.infer(
        "nmt",
        model_version='1',
        inputs=inputs,
        outputs=[output0],
        headers=HTTP_HEADERS,
    )#.get_response()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inference request completed in {elapsed_time:.4f} seconds.")
    # Decode the response
    output_batch = response.as_numpy('OUTPUT_TEXT').tolist()
    result = []
    for input_sentence, translation in zip(input_sentences, output_batch):
        print()
        print(input)
        print(translation[0].decode("utf-8"))
        result.append(translation[0].decode("utf-8"))

    return result, elapsed_time

# source =  [
#     'as', 'bn', 'brx', 'doi',  'gu', 'hi',"en", 'kn', 'ks', 'gom', 'mai', 'ne', 'ml',
#     'mni', 'mr', 'or', 'pa', 'sa','sat','sd', 'ta', 'te', 'ur'
# ]
# # l = ["hi", "mr", "te","en"]
# # print(test("aap kya kar rahe ho?", "mr"))
# import pandas as pd
# df = pd.read_excel("/home/translation-exp-vm/sab/transliteration_test/Hinglish_data_pipeline testing.xlsx")
# for i in source:
#     df[i] = df['Hinglish'].apply(lambda x: test(x, i))
# # print(test("hi","en"))
# # for i in l:
# #     df.loc[:67, i], df.loc[:67, f"time_{i}"] = zip(*df.loc[:67, 'Hinglish'].apply(lambda x: test(x, i)))


# df.to_excel("/home/translation-exp-vm/sab/transliteration_test/Hinglish_data_pipeline testing.xlsx")    
inputs = get_translation_input_for_triton(input_sentences, "en", 'hin')
output0 = http_client.InferRequestedOutput("OUTPUT_TEXT")
start_time = time.time()
# Send request
response = triton_http_client.infer(
    "nmt",
    model_version='1',
    inputs=inputs,
    outputs=[output0],
    headers=HTTP_HEADERS,
)#.get_response()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Inference request completed in {elapsed_time:.4f} seconds.")
# Decode the response
output_batch = response.as_numpy('OUTPUT_TEXT').tolist()
result = []
for input_sentence, translation in zip(input_sentences, output_batch):
    print()
    print(input_sentence)
    print(translation[0].decode("utf-8"))