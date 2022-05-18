# -*- coding: utf-8 -*-
from odoo import api, fields, models, _, tools
from odoo.osv import expression
from odoo.exceptions import UserError, ValidationError
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from datetime import datetime
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,CategoricalNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from pickle import dump
import wget
from os.path import exists
import os
class LangModel(models.Model):
    _name='language.model'
    _description="Language Model"

    def _get_valid_algorithms(self):
        domain = [('target_analysis', '=', 'lang_detection')]
        return domain

    model_name = fields.Char("Model Name",translate=True)
    model_algorithm = fields.Many2one ('algorithm','Algorithm', domain=_get_valid_algorithms)
    file_path_model = fields.Char('File Path')
    file_path_label_encoder = fields.Char('Label Encoder')
    file_path_count_vectorizer = fields.Char('Count Vectorizer')
    corpus_id=fields.Many2one('language.corpus','Corpus')
    model_accuracy = fields.Float ('Accuracy')

    def name_get(self):

        return [(record.id, record.model_name) for record in self]

    def prepare_corpus (self):
        sentences=[]
        languages=[]

        for i in self.corpus_id.sentences_ids:
             sentences.append(i.name)
             languages.append(i.language_id.code)
        d = {'sentence': sentences, 'language': languages}
        return d
    def train_model(self):
        a=self.prepare_corpus()
        data = pd.DataFrame(data=a)
        X = data["sentence"]
        y = data["language"]
        le = LabelEncoder()
        y = le.fit_transform(y)
        dump(le, open('/odoo/odoo-server/addons/linguistica_nlp/models/language_model/'+self.model_algorithm.code+'_label_model.pkl', 'wb'))
        self.file_path_label_encoder = '/odoo/odoo-server/addons/linguistica_nlp/models/language_model/'+self.model_algorithm.code+'_label_model.pkl'

        data_list = []
        for text in X:
            text = re.sub(r'[!@#$(),"%^*?:;~`0-9]', ' ', text)
            text = re.sub(r'[[]]', ' ', text)
            # converting the text to lower case
            text = text.lower()
            # appending to data_list
            data_list.append(text)
        cv = CountVectorizer()
        X = cv.fit_transform(data_list).toarray()
        dump(cv, open('/odoo/odoo-server/addons/linguistica_nlp/models/language_model/'+self.model_algorithm.code+'_model_cv.pkl', 'wb'))
        self.file_path_count_vectorizer = '/odoo/odoo-server/addons/linguistica_nlp/models/language_model/'+self.model_algorithm.code+'_model_cv.pkl'

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
        if self.model_algorithm.code == 'gaussian':
           model = GaussianNB()
        if self.model_algorithm.code == 'complement':
           model = ComplementNB ()
        if self.model_algorithm.code == 'bernoulli':
           model = BernoulliNB ()
        if self.model_algorithm.code == 'multinomial':
           model = MultinomialNB()
        if self.model_algorithm.code == 'categorical':
           model = CategoricalNB()

        model.fit(x_train, y_train)
        dump(model, open('/odoo/odoo-server/addons/linguistica_nlp/models/language_model/'+self.model_algorithm.code+'_language_model.pkl', 'wb'))
        y_pred = model.predict(x_test)
        ac = accuracy_score(y_test, y_pred)
        self.model_accuracy=ac
        self.file_path_model='/odoo/odoo-server/addons/linguistica_nlp/models/language_model/'+self.model_algorithm.code+'_language_model.pkl'
        #raise ValidationError(_('Corpus %s ', a))
class CorpusLanguageDetection (models.Model):
    _name = 'language.corpus'
    _description = 'Language Corpus'

    def _get_valid_sentences(self):
     sentences = self.env['sentence'].search([('state','in',['correct','valid'])])
     if sentences:
        domain = [('id', 'in', sentences.ids)]
     else:
        domain = [('id', '=', -1)]
     return domain


    name= fields.Char('Corpus Name',translate=True)
    #model_ids= fields.One2many ('language.model','corpus_id','Models')
    sentences_ids= fields.Many2many  ('sentence', 'language_corpus_sentence_rel', 'corpus_id', 'sentence_id', 'Sentences')
    corpora_languages_ids = fields.Many2many  ('language', 'language_corpus_rel', 'corpus_id', 'language_id', 'Languages')
    number_of_sentences = fields.Integer("Number of sentences")
    sentence_max_length = fields.Integer("Sentence Max Length")
    sentence_min_length = fields.Integer("Sentence Min Length")
    def download_corpora_from_tatoeba (self):

        languages=[]
        for language in self.corpora_languages_ids:
            languages.append(language.code)
        #raise ValidationError(_('Unit Error please correct the syntaxe %s ', languages))
        for i in languages:
            if not exists('/odoo/odoo-server/addons/linguistica_nlp/models/tatoeba/'+i+'_sentences.tsv.bz2'):
               wget.download('https://downloads.tatoeba.org/exports/per_language/'+i+'/'+i+'_sentences.tsv.bz2','/odoo/odoo-server/addons/linguistica_nlp/models/tatoeba/'+str(i)+'_sentences.tsv.bz2')

        for i in languages:
            if not exists('/odoo/odoo-server/addons/linguistica_nlp/models/tatoeba/'+i+'_sentences.tsv'):
               os.system('bunzip2 '+'/odoo/odoo-server/addons/linguistica_nlp/models/tatoeba/'+str(i)+'_sentences.tsv.bz2')
        for i in languages:
            if exists('/odoo/odoo-server/addons/linguistica_nlp/models/tatoeba/'+i+'_sentences.tsv'):
               origine_file= open('/odoo/odoo-server/addons/linguistica_nlp/models/tatoeba/'+i+'_sentences.tsv',encoding='utf-8')
               if not exists('/odoo/odoo-server/addons/linguistica_nlp/models/tatoeba/'+i+'_sentences.txt'):
                   target_file=open('/odoo/odoo-server/addons/linguistica_nlp/models/tatoeba/'+i+'_sentences.txt',"w+",encoding='utf-8')
                   for sentence in origine_file:
                     sentence=sentence.replace("\ufeff","")
                     vals = sentence.split("\t")
                     target_file.write(vals[2])
                   target_file.close()
               origine_file.close()
        limit=self.number_of_sentences
        i=0
        learning_file= open("/odoo/odoo-server/addons/linguistica_nlp/models/tatoeba/LanguageDetection.csv","w+",encoding='utf-8')
        learning_file.write("sentence"+'\t'+"language"+'\r')
        query1 = " DELETE from language_corpus_sentence_rel where corpus_id = " + str(self.id)
        self.env.cr.execute(query1)

        for language in languages:
            i=0
            language_file=open('/odoo/odoo-server/addons/linguistica_nlp/models/tatoeba/'+language+"_sentences.txt",encoding='utf-8')

            for sentence in language_file:
                if len(sentence)>=self.sentence_min_length and len(sentence)<=self.sentence_max_length:
                    learning_file.write(sentence.replace("\n","").replace("\t","")+'\t'+language+'\r')
                    if not self.env['sentence'].search([('name','=',sentence.replace("\n","").replace("\t",""))]):
                        sentence_id = self.env['sentence'].create({'name':sentence.replace("\n","").replace("\t",""), 'language_id':self.env['language'].search([('code','=',language)]).id})
                        query = " INSERT INTO language_corpus_sentence_rel (corpus_id, sentence_id) VALUES ("+str(self.id)+","+str(sentence_id.id)+")"
                        self.env.cr.execute(query)



                        i=i+1
                        if limit!=0 and i>=limit:
                            language_file.close()
                            break
                    else:
                        query = " INSERT INTO language_corpus_sentence_rel (corpus_id, sentence_id) VALUES ("+str(self.id)+","+str(self.env['sentence'].search([('name','=',sentence.replace("\n","").replace("\t",""))]).id)+")"
                        self.env.cr.execute(query)
                        i=i+1
                        if limit!=0 and i>=limit:
                            language_file.close()
                            break



            language_file.close()
        learning_file.close()
