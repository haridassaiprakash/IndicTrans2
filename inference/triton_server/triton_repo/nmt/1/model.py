import os
import sys
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from ai4bharat.transliteration import XlitEngine
PWD = os.path.dirname(__file__)

INFERENCE_MODULE_DIR = "/home/indicTrans2/"
sys.path.insert(0, INFERENCE_MODULE_DIR)
from inference.engine import Model, iso_to_flores
INDIC_LANGUAGES = set(iso_to_flores)

ALLOWED_DIRECTION_STRINGS = {"en-indic", "indic-en", "indic-indic"}
FORCE_PIVOTING = False
DEFAULT_PIVOT_LANG = "en"

class TritonPythonModel:


    def __init__(self):
        # Initialize transliteration engine once
        self.translit_engine = XlitEngine("hi", beam_width=10)
        self.translit_engine_en = XlitEngine("en", src_script_type="indic", beam_width=10)
          # Assuming other models are loaded here

    def transliterate(self, input_text):
        # Use the preloaded transliteration engine
        out = self.translit_engine.translit_sentence(input_text)
        return out['hi']
    
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.model_instance_device_id = json.loads(args['model_instance_device_id'])
        self.output_name = "OUTPUT_TEXT"
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(self.model_config, self.output_name)["data_type"])
        

        # checkpoints_root_dir = os.path.join(PWD, "checkpoints")
        checkpoints_root_dir = "/models/checkpoints"
        checkpoint_folders = [ f.path for f in os.scandir(checkpoints_root_dir) if f.is_dir() ]
        # The assumption is that, each folder name is `<src_direction>-to-<tgt_direction>`

        if not checkpoint_folders:
            raise RuntimeError(f"No checkpoint folders in: {checkpoints_root_dir}")

        self.models = {}
        for checkpoint_folder in checkpoint_folders:
            direction_string = os.path.basename(checkpoint_folder)
            assert direction_string in ALLOWED_DIRECTION_STRINGS, f"Checkpoint folder-name `{direction_string}` not allowed"
            self.models[direction_string] = Model(os.path.join(checkpoint_folder, "ct2_fp16_model"), input_lang_code_format="iso", model_type="ctranslate2")
            # self.models[direction_string] = Model(checkpoint_folder, input_lang_code_format="iso", model_type="fairseq")
        
        self.pivot_lang = None
        if "en-indic" in self.models and "indic-en" in self.models:
            if  "indic-indic" not in self.models:
                self.pivot_lang = DEFAULT_PIVOT_LANG
            elif FORCE_PIVOTING:
                del self.models["indic-indic"]
                self.pivot_lang = DEFAULT_PIVOT_LANG
    
    def get_direction_string(self, input_language_id, output_language_id):
        direction_string = None
        print(f"input_language_id : {input_language_id}, output_language_id; {output_language_id}")
        if input_language_id == DEFAULT_PIVOT_LANG and output_language_id in INDIC_LANGUAGES:
            direction_string = "en-indic"
            print("1 ______________")
        elif input_language_id in INDIC_LANGUAGES and output_language_id != "hin":
            print("2 ______________")
            if output_language_id == DEFAULT_PIVOT_LANG:
                direction_string = "indic-en"
                print("3 ______________")
            elif output_language_id in INDIC_LANGUAGES:
                direction_string = "indic-indic"
                print("4 ______________")
        elif input_language_id == "hin" and output_language_id == DEFAULT_PIVOT_LANG:
            direction_string = "hin-en"        
            print("5 ______________")
        elif input_language_id == "hin" and output_language_id in  INDIC_LANGUAGES:
            direction_string = "hin-indic"
            print("6 ______________")
        elif output_language_id == "hin" and input_language_id ==  DEFAULT_PIVOT_LANG:
            direction_string = "en-hin"
            print("7 ______________")
        elif output_language_id == "hin" and input_language_id in  INDIC_LANGUAGES:
            direction_string = "indic-hin"  

            print(f"8 ______________{direction_string}")  
        
        return direction_string

    def get_model(self, input_language_id, output_language_id):
        direction_string = self.get_direction_string(input_language_id, output_language_id)
        
        if direction_string in self.models:
            return self.models[direction_string]
        raise RuntimeError(f"Language-pair not supported: {input_language_id}-{output_language_id}")
    # def trasnliterate(self, input_text):
    #     e = XlitEngine("hi", beam_width=10)
    #     out = e.translit_sentence(input_text)
    #     return out['hi']


        
    def execute(self,requests):
        # print("REQ_COUNT", len(requests))
        modelwise_batches = {}
        responses = []
        for request_id, request in enumerate(requests):
            input_text_batch = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT").as_numpy()
            input_language_id_batch = pb_utils.get_input_tensor_by_name(request, "INPUT_LANGUAGE_ID").as_numpy()
            output_language_id_batch = pb_utils.get_input_tensor_by_name(request, "OUTPUT_LANGUAGE_ID").as_numpy()
            
            input_text_batch = [input_text[0].decode("utf-8", "ignore") for input_text in input_text_batch]
            input_language_id_batch = [input_language_id[0].decode("utf-8", "ignore") for input_language_id in input_language_id_batch]
            output_language_id_batch = [output_language_id[0].decode("utf-8", "ignore") for output_language_id in output_language_id_batch]

            responses.append([['']] * len(input_text_batch))
            hi_results = []
            for input_id, (input_text, input_language_id, output_language_id) in enumerate(zip(input_text_batch, input_language_id_batch, output_language_id_batch)):
                is_hin = "False"
                
                if input_language_id == "hi" and output_language_id == "hin" :
                    print(f"input_text: - {input_text}")
                    
                    hi_results.append(self.translit_engine_en.translit_sentence(input_text, lang_code="hi"))
                elif input_language_id == "hin" and output_language_id== "hi":
                    print(f"input_text: - {input_text}")
                    
                    hi_results.append(self.translit_engine.translit_sentence(input_text, lang_code="hi"))

                direction_string = self.get_direction_string(input_language_id, output_language_id)
                print(f"input_language_id1 : {input_language_id}, output_language_id; {output_language_id}")
                print(f"direction_string: - {direction_string}")
                if direction_string == "hin-indic":
                    input_text = self.transliterate(input_text)
                    direction_string = "indic-indic"
                    input_language_id = "hi"
                if direction_string == "hin-en":
                    input_text = self.transliterate(input_text)
                    direction_string = "indic-en"
                    input_language_id = "hi"
                if direction_string == "indic-hin":
                    direction_string = "indic-indic"
                    output_language_id = "hi"
                    is_hin = "True"
                if direction_string == "en-hin":
                    direction_string = "en-indic"
                    output_language_id = "hi"
                    is_hin = "True"    

                    

                if direction_string not in self.models:
                    if direction_string == "indic-indic" and self.pivot_lang:
                        pass
                    else:
                        raise RuntimeError(f"Language-pair not supported: {input_language_id}-{output_language_id}")
                print(f"direction string {direction_string}")
                if direction_string not in modelwise_batches:
                    modelwise_batches[direction_string] = {
                        "payloads": [],
                        "text_id_to_req_id_input_id": [],
                    }
                print(f"input_language_idf2 : {input_language_id}, output_language_id; {output_language_id}")
                modelwise_batches[direction_string]["payloads"].append([input_text, input_language_id, output_language_id])
                modelwise_batches[direction_string]["text_id_to_req_id_input_id"].append((request_id, input_id))

        for direction_string, batch in modelwise_batches.items():
            if input_language_id == "hi" and output_language_id== "hi":
                print(f"hi_results : - {hi_results}")
                for translation, (request_id, output_id) in zip(hi_results, batch["text_id_to_req_id_input_id"]):
                    responses[request_id][output_id] = [translation]
                    print("Hinglish")
                print( f"batch payload1:{batch['payloads']}")
            else:
                if direction_string == "indic-indic" and self.pivot_lang:
                    
                        model = self.get_model("hi", self.pivot_lang)
                        original_langs = []
                        for i in range(len(batch["payloads"])):
                            original_langs.append(batch["payloads"][i][2])
                            batch["payloads"][i][2] = self.pivot_lang
                        print( f"batch payload:{batch['payloads']}")
                        pivot_texts = model.paragraphs_batch_translate__multilingual(batch["payloads"])
                        print(f"pivot_text :- {pivot_texts}")
                        for i in range(len(batch["payloads"])):
                            batch["payloads"][i][0] = pivot_texts[i]
                            batch["payloads"][i][1] = self.pivot_lang
                            batch["payloads"][i][2] = original_langs[i]
                        
                        model = self.get_model(self.pivot_lang, "hi")
                        translations = model.paragraphs_batch_translate__multilingual(batch["payloads"])
                        print(f"translations:- {translations}")
                        if is_hin == "True":
                            for i in range(len(translations)):
                                translations[i] = self.translit_engine_en.translit_sentence(translations[i], lang_code="hi")
                        print(f"translations:- {translations}")        

                        
                else:
                    model = self.models[direction_string]
                    
                    translations = model.paragraphs_batch_translate__multilingual(batch["payloads"])
                    if is_hin == "True":
                            for i in range(len(translations)):
                                translations[i] = self.translit_engine_en.translit_sentence(translations[i], lang_code="hi")
                    print(f"translations:- {translations}")        
                    # translations = ["bro"] * len(batch["payloads"])
                print(f"input_language_idf2 : {input_language_id}, output_language_id; {output_language_id}")
                
                
                for translation, (request_id, output_id) in zip(translations, batch["text_id_to_req_id_input_id"]):
                        responses[request_id][output_id] = [translation]
        
        for i in range(len(responses)):
            responses[i] = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    self.output_name,
                    np.array(responses[i], dtype=self.output_dtype),
                )
            ])
        return responses
    
    def execute_sequential(self,requests):
        # print("REQ_COUNT", len(requests))
        responses = []
        for request in requests:
            input_text_batch = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT").as_numpy()
            input_language_id_batch = pb_utils.get_input_tensor_by_name(request, "INPUT_LANGUAGE_ID").as_numpy()
            output_language_id_batch = pb_utils.get_input_tensor_by_name(request, "OUTPUT_LANGUAGE_ID").as_numpy()
            
            input_text_batch = [input_text[0].decode("utf-8", "ignore") for input_text in input_text_batch]
            input_language_id_batch = [input_language_id[0].decode("utf-8", "ignore") for input_language_id in input_language_id_batch]
            output_language_id_batch = [output_language_id[0].decode("utf-8", "ignore") for output_language_id in output_language_id_batch]

            generated_outputs = []

            for input_text, input_language_id, output_language_id in zip(input_text_batch, input_language_id_batch, output_language_id_batch):
                if self.pivot_lang and (input_language_id != self.pivot_lang and output_language_id != self.pivot_lang):
                    model = self.get_model(input_language_id, self.pivot_lang)
                    pivot_text = model.translate_paragraph(input_text, input_language_id, self.pivot_lang)
                    
                    model = self.get_model(self.pivot_lang, output_language_id)
                    translation = model.translate_paragraph(pivot_text, self.pivot_lang, output_language_id)
                else:
                    model = self.get_model(input_language_id, output_language_id)
                    translation = model.translate_paragraph(input_text, input_language_id, output_language_id)
                generated_outputs.append([translation])

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    self.output_name,
                    np.array(generated_outputs, dtype=self.output_dtype),
                )
            ])
            responses.append(inference_response)
        return responses
