import hashlib
import os
import uuid
from typing import List, Tuple, Union, Dict

import regex as re
import sentencepiece as spm
from indicnlp.normalize import indic_normalize
from indicnlp.tokenize import indic_detokenize, indic_tokenize
from indicnlp.tokenize.sentence_tokenize import DELIM_PAT_NO_DANDA, sentence_split
from indicnlp.transliterate import unicode_transliterate
from mosestokenizer import MosesSentenceSplitter
from nltk.tokenize import sent_tokenize
from sacremoses import MosesDetokenizer, MosesPunctNormalizer, MosesTokenizer
from tqdm import tqdm

from .flores_codes_map_indic import flores_codes, iso_to_flores
from .normalize_punctuation import punc_norm
from .normalize_regex_inference import EMAIL_PATTERN, normalize


def split_sentences(paragraph: str, lang: str) -> List[str]:
    """
    Splits the input text paragraph into sentences. It uses `moses` for English and
    `indic-nlp` for Indic languages.

    Args:
        paragraph (str): input text paragraph.
        lang (str): flores language code.

    Returns:
        List[str] -> list of sentences.
    """
    
    print("split_sentence")
    print("3rd test")
    print("4th test")
    if lang == "eng_Latn":
        with MosesSentenceSplitter(flores_codes[lang]) as splitter:
            sents_moses = splitter([paragraph])
        sents_nltk = sent_tokenize(paragraph)
        if len(sents_nltk) < len(sents_moses):
            sents = sents_nltk
        else:
            sents = sents_moses
        return [sent.replace("\xad", "") for sent in sents]
    else:
        return sentence_split(paragraph, lang=flores_codes[lang], delim_pat=DELIM_PAT_NO_DANDA)


def add_token(sent: str, src_lang: str, tgt_lang: str, delimiter: str = " ") -> str:
    """
    Add special tokens indicating source and target language to the start of the input sentence.
    The resulting string will have the format: "`{src_lang} {tgt_lang} {input_sentence}`".

    Args:
        sent (str): input sentence to be translated.
        src_lang (str): flores lang code of the input sentence.
        tgt_lang (str): flores lang code in which the input sentence will be translated.
        delimiter (str): separator to add between language tags and input sentence (default: " ").

    Returns:
        str: input sentence with the special tokens added to the start.
    """
    return src_lang + delimiter + tgt_lang + delimiter + sent


def apply_lang_tags(sents: List[str], src_lang: str, tgt_lang: str) -> List[str]:
    """
    Add special tokens indicating source and target language to the start of the each input sentence.
    Each resulting input sentence will have the format: "`{src_lang} {tgt_lang} {input_sentence}`".

    Args:
        sent (str): input sentence to be translated.
        src_lang (str): flores lang code of the input sentence.
        tgt_lang (str): flores lang code in which the input sentence will be translated.

    Returns:
        List[str]: list of input sentences with the special tokens added to the start.
    """
    tagged_sents = []
    for sent in sents:
        tagged_sent = add_token(sent.strip(), src_lang, tgt_lang)
        tagged_sents.append(tagged_sent)
    return tagged_sents


def truncate_long_sentences(
    sents: List[str], placeholder_entity_map_sents: List[Dict]
) -> Tuple[List[str], List[Dict]]:
    """
    Truncates the sentences that exceed the maximum sequence length.
    The maximum sequence for the IndicTrans2 model is limited to 256 tokens.

    Args:
        sents (List[str]): list of input sentences to truncate.

    Returns:
        Tuple[List[str], List[Dict]]: tuple containing the list of sentences with truncation applied and the updated placeholder entity maps.
    """
    MAX_SEQ_LEN = 256
    new_sents = []
    placeholders = []

    for j, sent in enumerate(sents):
        words = sent.split()
        num_words = len(words)
        if num_words > MAX_SEQ_LEN:
            sents = []
            i = 0
            while i <= len(words):
                sents.append(" ".join(words[i : i + MAX_SEQ_LEN]))
                i += MAX_SEQ_LEN
            placeholders.extend([placeholder_entity_map_sents[j]] * (len(sents)))
            new_sents.extend(sents)
        else:
            placeholders.append(placeholder_entity_map_sents[j])
            new_sents.append(sent)
    return new_sents, placeholders


class Model:
    """
    Model class to run the IndicTransv2 models using python interface.
    """

    def __init__(
        self,
        ckpt_dir: str,
        device: str = "cuda",
        input_lang_code_format: str = "flores",
        model_type: str = "ctranslate2",
    ):
        """
        Initialize the model class.

        Args:
            ckpt_dir (str): path of the model checkpoint directory.
            device (str, optional): where to load the model (defaults: cuda).
        """
        self.ckpt_dir = ckpt_dir
        self.en_tok = MosesTokenizer(lang="en")
        self.en_normalizer = MosesPunctNormalizer()
        self.en_detok = MosesDetokenizer(lang="en")
        self.xliterator = unicode_transliterate.UnicodeIndicTransliterator()

        print("Initializing sentencepiece model for SRC and TGT")
        self.sp_src = spm.SentencePieceProcessor(
            model_file=os.path.join(ckpt_dir, "vocab", "model.SRC")
        )
        self.sp_tgt = spm.SentencePieceProcessor(
            model_file=os.path.join(ckpt_dir, "vocab", "model.TGT")
        )

        self.input_lang_code_format = input_lang_code_format

        print("Initializing model for translation")
        # initialize the model
        if model_type == "ctranslate2":
            import ctranslate2

            self.translator = ctranslate2.Translator(
                self.ckpt_dir, device=device
            )  # , compute_type="auto")
            self.translate_lines = self.ctranslate2_translate_lines
        elif model_type == "fairseq":
            from .custom_interactive import Translator

            self.translator = Translator(
                data_dir=os.path.join(self.ckpt_dir, "final_bin"),
                checkpoint_path=os.path.join(self.ckpt_dir, "model", "checkpoint_best.pt"),
                batch_size=100,
            )
            self.translate_lines = self.fairseq_translate_lines
        else:
            raise NotImplementedError(f"Unknown model_type: {model_type}")

    def is_english(self,char: list, ignore_list: list) -> bool:
        allowed_ranges = [
            ('\u0041', '\u005A'),  # A-Z (uppercase English letters)
            ('\u0061', '\u007A')   # a-z (lowercase English letters)
        ]
        print(f"char : - {char}")
        # Iterate through each word in the list
        for word in char:
            print(f"word: {word}")
            # Now iterate through each character in the word
            for character in word:
                print(f"character: {character}")
                # Check if the character is not in the ignore list and is within the allowed ranges
                if character not in ignore_list and any(start <= character <= end for start, end in allowed_ranges):
                    
                    return True  # Found a valid English letter
        
        return False
    
    
    def ctranslate2_translate_lines(self,  lines: List[str], len_id: list) -> List[str]:
        tokenized_sents = [x.strip().split(" ") for x in lines]
        if tokenized_sents[0][0] == "eng_Latn":
            print(f"ctranslate2_translate_lines : tokenized sents{lines}")
            print(f"ctranslate2_translate_lines : tokenized sents{tokenized_sents}")
            translations = self.translator.translate_batch(
                tokenized_sents,
                max_batch_size=9216,
                batch_type="tokens",
                max_input_length=160,
                max_decoding_length=256,
                return_scores=True,
                beam_size=5,
                num_hypotheses = 5
                # target_prefix = [["नरेंद्र मोदी स्वतंत्रता दिवस पर लोगों को संबोधित कर रहे हैं।"]]
            )
            final_response = []
            for i, len_ids in zip(translations,len_id):
                
                found_correct = False
                for j in i.hypotheses:
                    
                    if len_ids == 0:
                        ignore_list = []
                        
                    else:
                        ignore_list = ['I',"D"] 
                        
                    if not self.is_english(j,ignore_list):
                        
                        final_response.append(j)
                        found_correct = True
                        break
                if not found_correct:
                    final_response.append(i.hypotheses[0])                  
            translations = [" ".join(x) for x in final_response]   
            
            return translations         
        else:
            
            tokenized_sents = [x.strip().split(" ") for x in lines]
            
            translations = self.translator.translate_batch(
            tokenized_sents,
            max_batch_size=9216,
            batch_type="tokens",
            max_input_length=160,
            max_decoding_length=256,
            beam_size=5,
            )
            translations = [" ".join(x.hypotheses[0]) for x in translations]
            return translations

    def fairseq_translate_lines(self, lines: List[str]) -> List[str]:
        return self.translator.translate(lines)


    def char_percent_check(input):
        """
        
        Calculate the percentage of Roman characters (English letters and digits) 
        in the given input string after removing special characters, spaces, 
        newlines, emails, and URLs.

        Args:
            input (str): The input string to analyze.

        Returns:
            float: The percentage of Roman characters in the total valid characters 
                of the string. Returns 0 if the total valid characters are zero.
        
        """
       
        input_len = len(list(input))
        print(input_len)
        spaces = len(re.findall('\s', input))
        newlines = len(re.findall('\n', input))
        email_pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
        url_pattern = re.compile(r'(https?://\S+|www\.\S+)')
        emails = email_pattern.findall(input)
        urls = url_pattern.findall(input)
        
        email_len = sum(len(email) for email in emails)
       
        urls = url_pattern.findall(input)
        url_len = sum(len(url) for url in urls)
     
        input_str_no_emails_urls = email_pattern.sub('', input)
        input_str_no_emails_urls = url_pattern.sub('', input_str_no_emails_urls)

        special_char_pattern = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        special_char_matches = special_char_pattern.findall(input_str_no_emails_urls)
        special_chars = len(special_char_matches)
        total_chars = input_len - ( special_chars + spaces + newlines + email_len + url_len)
        print(f"input_len :{input_len}")
        print(f"special_chars :{special_chars}")
        print(f"spaces :{spaces}")
        print(f"newlines :{newlines}")
        print(f"email_len :{email_len}")
        print(f"url_len :{url_len}")
        
        en_pattern = re.compile('[a-zA-Z0-9]')
        en_matches = en_pattern.findall(input_str_no_emails_urls)
        en_chars = len(en_matches)
        print(f"en_chars :{en_chars}")
        print(f"total_chars :{total_chars}")
        
        if total_chars == 0:
            return 0
        return (en_chars/total_chars)
    
    
    def paragraphs_batch_translate__multilingual(self, batch_payloads: List[tuple]) -> List[str]:
        """
        Translates a batch of input paragraphs (including pre/post processing) 
        from any language to any language.
        
        Args:
            batch_payloads (List[tuple]): batch of long input-texts to be translated, each in format: (paragraph, src_lang, tgt_lang)
        
        Returns:
            List[str]: batch of paragraph-translations in the respective languages.
        """
        paragraph_id_to_sentence_range = []
        global__sents = []
        global__preprocessed_sents = []
        global__preprocessed_sents_placeholder_entity_map = []
        
        len_id = []
        dict_of_non_english = {}
        for i in range(len(batch_payloads)):
            paragraph, src_lang, tgt_lang = batch_payloads[i]
            
                
            if self.input_lang_code_format == "iso":
                src_lang, tgt_lang = iso_to_flores[src_lang], iso_to_flores[tgt_lang]
            
            if src_lang == "eng_Latn":
                print(f"char_percent_check: - {self.char_percent_check(paragraph)}")
                if  self.char_percent_check(paragraph) < 0.5:
                    dict_of_non_english[i] = paragraph
            
            
            
            
            batch = split_sentences(paragraph, src_lang)
            global__sents.extend(batch)

            preprocessed_sents, placeholder_entity_map_sents = self.preprocess_batch(batch, src_lang, tgt_lang)

            # Sab: *************************
            for i in range(len(placeholder_entity_map_sents)):
                
                len_id.append(len(placeholder_entity_map_sents[i]))
            print(f"Len ID : -{len_id}")
            print(f"placeholder_entity_map_sents {placeholder_entity_map_sents}")
            # ***************************************
            global_sentence_start_index = len(global__preprocessed_sents)
            global__preprocessed_sents.extend(preprocessed_sents)
            global__preprocessed_sents_placeholder_entity_map.extend(placeholder_entity_map_sents)
            paragraph_id_to_sentence_range.append((global_sentence_start_index, len(global__preprocessed_sents)))
        
        translations = self.translate_lines(global__preprocessed_sents,len_id)

        translated_paragraphs = []
        for paragraph_id, sentence_range in enumerate(paragraph_id_to_sentence_range):
            tgt_lang = batch_payloads[paragraph_id][2]
            if self.input_lang_code_format == "iso":
                tgt_lang = iso_to_flores[tgt_lang]
            
            postprocessed_sents = self.postprocess(
                translations[sentence_range[0]:sentence_range[1]],
                global__preprocessed_sents_placeholder_entity_map[sentence_range[0]:sentence_range[1]],
                tgt_lang,
            )
            translated_paragraph = " ".join(postprocessed_sents)
            translated_paragraphs.append(translated_paragraph)
        
        print(f"translated_paragraphs: - {translated_paragraphs}")
        for index, new_sentence in dict_of_non_english.items():
            translated_paragraphs[index] = new_sentence
        print(f"translated_paragraphs: - {translated_paragraphs}")
        
        return translated_paragraphs

    # translate a batch of sentences from src_lang to tgt_lang
    def batch_translate(self, batch: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        Translates a batch of input sentences (including pre/post processing)
        from source language to target language.

        Args:
            batch (List[str]): batch of input sentences to be translated.
            src_lang (str): flores source language code.
            tgt_lang (str): flores target language code.

        Returns:
            List[str]: batch of translated-sentences generated by the model.
        """

        assert isinstance(batch, list)

        if self.input_lang_code_format == "iso":
            src_lang, tgt_lang = iso_to_flores[src_lang], iso_to_flores[tgt_lang]

        preprocessed_sents, placeholder_entity_map_sents = self.preprocess_batch(
            batch, src_lang, tgt_lang
        )
        translations = self.translate_lines(preprocessed_sents)
        return self.postprocess(translations, placeholder_entity_map_sents, tgt_lang)

    # translate a paragraph from src_lang to tgt_lang
    def translate_paragraph(self, paragraph: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translates an input text paragraph (including pre/post processing)
        from source language to target language.

        Args:
            paragraph (str): input text paragraph to be translated.
            src_lang (str): flores source language code.
            tgt_lang (str): flores target language code.

        Returns:
            str: paragraph translation generated by the model.
        """

        assert isinstance(paragraph, str)

        if self.input_lang_code_format == "iso":
            flores_src_lang = iso_to_flores[src_lang]
        else:
            flores_src_lang = src_lang

        sents = split_sentences(paragraph, flores_src_lang)
        postprocessed_sents = self.batch_translate(sents, src_lang, tgt_lang)
        translated_paragraph = " ".join(postprocessed_sents)

        return translated_paragraph

    def preprocess_batch(self, batch: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        """
        Preprocess an array of sentences by normalizing, tokenization, and possibly transliterating it. It also tokenizes the
        normalized text sequences using sentence piece tokenizer and also adds language tags.

        Args:
            batch (List[str]): input list of sentences to preprocess.
            src_lang (str): flores language code of the input text sentences.
            tgt_lang (str): flores language code of the output text sentences.

        Returns:
            Tuple[List[str], List[Dict]]: a tuple of list of preprocessed input text sentences and also a corresponding list of dictionary
                mapping placeholders to their original values.
        """
        preprocessed_sents, placeholder_entity_map_sents = self.preprocess(batch, lang=src_lang)
        tokenized_sents = self.apply_spm(preprocessed_sents)
        tokenized_sents, placeholder_entity_map_sents = truncate_long_sentences(
            tokenized_sents, placeholder_entity_map_sents
        )
        tagged_sents = apply_lang_tags(tokenized_sents, src_lang, tgt_lang)
        return tagged_sents, placeholder_entity_map_sents

    def apply_spm(self, sents: List[str]) -> List[str]:
        """
        Applies sentence piece encoding to the batch of input sentences.

        Args:
            sents (List[str]): batch of the input sentences.

        Returns:
            List[str]: batch of encoded sentences with sentence piece model
        """
        return [" ".join(self.sp_src.encode(sent, out_type=str)) for sent in sents]

    def preprocess_sent(
        self,
        sent: str,
        normalizer: Union[MosesPunctNormalizer, indic_normalize.IndicNormalizerFactory],
        lang: str,
    ) -> Tuple[str, Dict]:
        """
        Preprocess an input text sentence by normalizing, tokenization, and possibly transliterating it.

        Args:
            sent (str): input text sentence to preprocess.
            normalizer (Union[MosesPunctNormalizer, indic_normalize.IndicNormalizerFactory]): an object that performs normalization on the text.
            lang (str): flores language code of the input text sentence.

        Returns:
            Tuple[str, Dict]: A tuple containing the preprocessed input text sentence and a corresponding dictionary
            mapping placeholders to their original values.
        """
        iso_lang = flores_codes[lang]
        sent = punc_norm(sent, iso_lang)
        sent, placeholder_entity_map = normalize(sent)

        transliterate = True
        if lang.split("_")[1] in ["Arab", "Aran", "Olck", "Mtei", "Latn"]:
            transliterate = False

        if iso_lang == "en":
            processed_sent = " ".join(
                self.en_tok.tokenize(self.en_normalizer.normalize(sent.strip()), escape=False)
            )
        elif transliterate:
            # transliterates from the any specific language to devanagari
            # which is why we specify lang2_code as "hi".
            processed_sent = self.xliterator.transliterate(
                " ".join(
                    indic_tokenize.trivial_tokenize(normalizer.normalize(sent.strip()), iso_lang)
                ),
                iso_lang,
                "hi",
            ).replace(" ् ", "्")
        else:
            # we only need to transliterate for joint training
            processed_sent = " ".join(
                indic_tokenize.trivial_tokenize(normalizer.normalize(sent.strip()), iso_lang)
            )

        return processed_sent, placeholder_entity_map

    def preprocess(self, sents: List[str], lang: str):
        """
        Preprocess an array of sentences by normalizing, tokenization, and possibly transliterating it.

        Args:
            batch (List[str]): input list of sentences to preprocess.
            lang (str): flores language code of the input text sentences.

        Returns:
            Tuple[List[str], List[Dict]]: a tuple of list of preprocessed input text sentences and also a corresponding list of dictionary
                mapping placeholders to their original values.
        """
        processed_sents, placeholder_entity_map_sents = [], []

        if lang == "eng_Latn":
            normalizer = None
        else:
            normfactory = indic_normalize.IndicNormalizerFactory()
            normalizer = normfactory.get_normalizer(flores_codes[lang])

        for sent in sents:
            sent, placeholder_entity_map = self.preprocess_sent(sent, normalizer, lang)
            processed_sents.append(sent)
            placeholder_entity_map_sents.append(placeholder_entity_map)

        return processed_sents, placeholder_entity_map_sents

    def postprocess(
        self,
        sents: List[str],
        placeholder_entity_map: List[Dict],
        lang: str,
        common_lang: str = "hin_Deva",
    ) -> List[str]:
        """
        Postprocesses a batch of input sentences after the translation generations.

        Args:
            sents (List[str]): batch of translated sentences to postprocess.
            placeholder_entity_map (List[Dict]): dictionary mapping placeholders to the original entity values.
            lang (str): flores language code of the input sentences.
            common_lang (str, optional): flores language code of the transliterated language (defaults: hin_Deva).

        Returns:
            List[str]: postprocessed batch of input sentences.
        """

        lang_code, script_code = lang.split("_")
        # SPM decode
        for i in range(len(sents)):
            # sent_tokens = sents[i].split(" ")
            # sents[i] = self.sp_tgt.decode(sent_tokens)

            sents[i] = sents[i].replace(" ", "").replace("▁", " ").strip()

            # Fixes for Perso-Arabic scripts
            # TODO: Move these normalizations inside indic-nlp-library
            if script_code in {"Arab", "Aran"}:
                # UrduHack adds space before punctuations. Since the model was trained without fixing this issue, let's fix it now
                sents[i] = sents[i].replace(" ؟", "؟").replace(" ۔", "۔").replace(" ،", "،")
                # Kashmiri bugfix for palatalization: https://github.com/AI4Bharat/IndicTrans2/issues/11
                sents[i] = sents[i].replace("ٮ۪", "ؠ")

        assert len(sents) == len(placeholder_entity_map)

        for i in range(0, len(sents)):
            for key in placeholder_entity_map[i].keys():
                sents[i] = sents[i].replace(key, placeholder_entity_map[i][key])

        # Detokenize and transliterate to native scripts if applicable
        postprocessed_sents = []

        if lang == "eng_Latn":
            for sent in sents:
                postprocessed_sents.append(self.en_detok.detokenize(sent.split(" ")))
        else:
            for sent in sents:
                outstr = indic_detokenize.trivial_detokenize(
                    self.xliterator.transliterate(
                        sent, flores_codes[common_lang], flores_codes[lang]
                    ),
                    flores_codes[lang],
                )
                
                # Oriya bug: indic-nlp-library produces ଯ଼ instead of ୟ when converting from Devanagari to Odia
                # TODO: Find out what's the issue with unicode transliterator for Oriya and fix it
                if lang_code == "ory":
                    outstr = outstr.replace("ଯ଼", 'ୟ')

                postprocessed_sents.append(outstr)

        return postprocessed_sents
