import numpy as np
import tokenization
import pdb
import six
import json
import collections
from tqdm import tqdm
import tensorflow as tf
rand_seed = '12345'
do_lower_case = False
vocab_file = 'D:/model/multi_cased_L-12_H-768_A-12/vocab.txt'
# set random seed (i don't know whether it works or not)
np.random.seed(int(rand_seed))


#
class SquadExample(object):


    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 # doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 # input_ids,
                 # input_mask,
                 # segment_ids,
                 # input_span_mask,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        # self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        # self.input_ids = input_ids
        # self.input_mask = input_mask
        # self.segment_ids = segment_ids
        # self.input_span_mask = input_span_mask
        self.start_position = start_position
        self.end_position = end_position


#
def customize_tokenizer(text, do_lower_case=False):
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)
    temp_x = ""
    text = tokenization.convert_to_unicode(text)
    for c in text:
        if tokenizer._is_chinese_char(ord(c)) or tokenization._is_punctuation(c) or tokenization._is_whitespace(
                c) or tokenization._is_control(c):
            temp_x += " " + c + " "
        else:
            temp_x += c
    if do_lower_case:
        temp_x = temp_x.lower()
    return temp_x.split()


#
class ChineseFullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=False):
        self.vocab = tokenization.load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=self.vocab)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        split_tokens = []
        for token in customize_tokenizer(text, do_lower_case=self.do_lower_case):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return tokenization.convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return tokenization.convert_by_vocab(self.inv_vocab, ids)


#
def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""
    with tf.gfile.Open(input_file, "r") as reader:
        input_data = json.load(reader)["data"]

    #
    examples = []
    for entry in tqdm(input_data):
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            raw_doc_tokens = customize_tokenizer(paragraph_text)
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            k = 0
            temp_word = ""
            for c in paragraph_text:
                if tokenization._is_whitespace(c):
                    char_to_word_offset.append(k - 1)
                    continue
                else:
                    temp_word += c
                    char_to_word_offset.append(k)
                if do_lower_case:
                    temp_word = temp_word.lower()
                if temp_word == raw_doc_tokens[k]:
                    doc_tokens.append(temp_word)
                    temp_word = ""
                    k += 1

            assert k == len(raw_doc_tokens)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None

                if is_training:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]

                    if orig_answer_text not in paragraph_text:
                        tf.logging.warning("Could not find answer")
                    else:
                        answer_offset = paragraph_text.index(orig_answer_text)
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]

                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = "".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = "".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            pdb.set_trace()
                            tf.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                            continue

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                examples.append(example)
    tf.logging.info("**********read_squad_examples complete!**********")

    return examples


def convert_examples_to_features(examples, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 data_collector, bert_client):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    tokenizer = ChineseFullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        # doc_tokens 整个段落
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                # sub_tokens 在处理例如 playing 这种词的时候会有不止一个元素 [play, ###ing]
                # tok_to_orig_index 相当于 all_doc_tokens 对 doc_tokens 的映射，
                # orig_to_tok_index 相当于 doc_tokens 对 all_doc_tokens 的映射
                tok_to_orig_index.append(i)
                # all_doc_tokens 中保存了例如 play ###ing 这种形式的词
                all_doc_tokens.append(sub_token)
        tok_start_position = None  # 答案开始的位置
        tok_end_position = None  # 答案结束的位置
        if is_training:
            # 应对 ###ing 这种情况，更新答案的 start position 和 end position
            tok_start_position = orig_to_tok_index[example.start_position]
            # 截断答案的结束位置 len(example.dic_tokens) <= len(all_doc_tokens)
            if example.end_position < len(example.doc_tokens) - 1:
                # 答案的结束位置必须用其在 all_doc_tokens 中的新位置进行更新
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            # all_doc_tokens 长度没有超出 max_tokens_for_doc
            if start_offset + length == len(all_doc_tokens):
                break
            # 如果 all_doc_tokens 长度超出了 max_tokens_for_doc 则将 start_offset 增加 （max_tokens_for_doc， doc_stride）中的最小值
            # 如果 doc_stride 较小，则会出现重复采样，即完整的答案可能会出现在两个 doc_span 里。
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            # segment_ids = []
            # input_span_mask = []
            tokens.append("[CLS]")
            # segment_ids.append(0)
            # input_span_mask.append(1)
            for token in query_tokens:
                tokens.append(token)
                # segment_ids.append(0)
                # input_span_mask.append(0)
            tokens.append("[SEP]")
            # segment_ids.append(0)
            # input_span_mask.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                # segment_ids.append(1)
                # input_span_mask.append(1)
            tokens.append("[SEP]")
            # segment_ids.append(1)
            # input_span_mask.append(0)

            # input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            # input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            # while len(input_ids) < max_seq_length:
            #     input_ids.append(0)
            #     input_mask.append(0)
            #     segment_ids.append(0)
            #     input_span_mask.append(0)

            # assert len(input_ids) == max_seq_length
            # assert len(input_mask) == max_seq_length
            # assert len(segment_ids) == max_seq_length
            # assert len(input_span_mask) == max_seq_length

            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                # 不满足 （答案开始位置 > 文本开始位置 且 答案结束位置 < 文本结束位置），就将该问答对设置为 out of span
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if example_index < 50:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                # return_tokens = get_bert_as_service_token_result(bert_client, convert_to_bert_as_service_input(tokens))
                # return_tokens = get_bert_as_service_token_result(bert_client, [tokenization.printable_text(x) for x in tokens[1:-1]])
                # tf.logging.info("server_input_tokens: %s" % convert_to_bert_as_service_input(tokens))
                # tf.logging.info("retrun_tokens: %s" % return_tokens)
                # assert  return_tokens == \
                       # " ".join([tokenization.printable_text(x) for x in tokens])
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                # tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                # tf.logging.info(
                #     "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                # tf.logging.info(
                #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                # tf.logging.info(
                #     "input_span_mask: %s" % " ".join([str(x) for x in input_span_mask]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info("orig_to_tok_index: %s" % " ".join([str(x) for x in orig_to_tok_index]))
                    tf.logging.info("tok_to_orig_index: %s" % " ".join([str(x) for x in tok_to_orig_index]))
                    tf.logging.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                # doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                # input_ids=input_ids,
                # input_mask=input_mask,
                # segment_ids=segment_ids,
                # input_span_mask=input_span_mask,
                start_position=start_position,
                end_position=end_position)

            data_collector.append(feature)
            unique_id += 1
    tf.logging.info("**********convert complete!**********")

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)

# 当答案出现在doc_spans 中的多个 doc_span 中时，选择最重要的那个 doc_span
def _check_is_max_context(doc_spans, cur_span_index, position):
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index

def get_bert_as_service_token_result(bert_client, input_doc):
    result = bert_client.encode([input_doc], show_tokens=True, is_tokenized=True)
    print(np.shape(result[0]))
    return ' '.join(result[1][0])
