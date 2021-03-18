from flask import Flask, render_template, request, jsonify
import os
import spacy

from utils import QueryProcessor, DocumentRetrieval, PassageRetrieval, AnswerExtractor

SPACY_MODEL = os.environ.get('SPACY_MODEL', 'en_core_web_sm')
QA_MODEL = os.environ.get('QA_MODEL', 'distilbert-base-cased-distilled-squad')
nlp = spacy.load(SPACY_MODEL, disable=['ner', 'parser', 'textcat'])
query_processor = QueryProcessor(nlp)
document_retriever = DocumentRetrieval()
passage_retriever = PassageRetrieval(nlp)
answer_extractor = AnswerExtractor(QA_MODEL, QA_MODEL)

app = Flask(__name__)

port = 3000


@app.route('/', methods = ['POST', 'GET'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        query = query_processor.generate_query(question)
        docs = document_retriever.search(query)
        passage_retriever.fit(docs)
        passages = passage_retriever.most_similar(question)
        answers = answer_extractor.extract(question, passages)
        return render_template('index.html', query=query, question=question, flag=True, answers=answers)
    else:
        return render_template('index.html', flag=False)

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=port, debug=True)
