"""Exact locations in normalized sentences; never guessed raw-file offsets."""


def locate(document, sentence_index, start, end):
    if type(sentence_index) is not int or not 0 <= sentence_index < len(document['sentences']):
        raise ValueError('Unknown sentence index')
    sentence = document['sentences'][sentence_index]
    text = sentence['normalized_text']
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        raise ValueError('Invalid sentence character range')
    doc_start = sentence['document_char_start'] + start
    doc_end = sentence['document_char_start'] + end
    return {'sentence_index': sentence_index, 'sentence_id': sentence['sentence_id'],
            'start': start, 'end': end, 'quote': text[start:end],
            'document_start': doc_start, 'document_end': doc_end,
            'token_ids': [token_id for token_id in sentence['token_ids']
                          if document['tokens'][token_id]['document_char_start'] < doc_end
                          and document['tokens'][token_id]['document_char_end'] > doc_start]}


def whole_sentences(document, indices):
    return [locate(document, i, 0, len(document['sentences'][i]['normalized_text'])) for i in indices]
