# -*- coding: utf-8 -*-
from odoo import api, fields, models, _, tools
from odoo.osv import expression
from odoo.exceptions import UserError, ValidationError
from joblib import dump, load
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB,CategoricalNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

def features(sentence, index):
    return {
        'word': sentence[index],    # Awal s timmad-is
        'is_first': index == 0,     # Ma yezga-d deg tazwar n tefyirt
        'is_last': index == len(sentence) - 1, # Ma yezgma-d deg taggar n tefyirt
        'is_capitalized': sentence[index][0].upper() == sentence[index][0], # MA ibeddu s usekkil meqqren
        'is_all_caps': sentence[index].upper() == sentence[index], # Ma yura meṛṛa s usekkil meqqren
        'is_all_lower': sentence[index].lower() == sentence[index], # ma yura meṛṛa s usekkil meẓẓiyen
        'prefix-1': sentence[index][0], #1 usekkil uzwir
        'prefix-2': sentence[index][:2], #2 isekkilen uzwiren
        'prefix-3': sentence[index][:3], #3 isekkilen uzwiren
        'prefix-4': sentence[index][:4], # 4 isekkilen uzwiren
        'prefix-5': sentence[index][:5], # 4 isekkilen uzwiren tettecmumuḥenḍ (aoriste intensif)
        'suffix-1': sentence[index][-1], #1 usekkil uḍfir
        'suffix-2': sentence[index][-2:], #2 isekkilen uḍfiren
        'suffix-3': sentence[index][-3:], #3 isekkilen uḍfiren
        'suffix-4': sentence[index][-4:], #2 isekkilen uḍfiren
        'prev_word': '' if index == 0 else sentence[index - 1], #awal uzwir
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1], #awal uḍfir

        'is_numeric': sentence[index].isdigit(),  #ma yegber kan izwilen
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:] #ma yegber asekkil meqqren daxel-is
    }


class sentence (models.Model):

    _name = 'sentence'
    _description = 'Sentence'


    name= fields.Char ('Sentence')
    tagged_sentence= fields.Char ('Tagged Sentence')
    number= fields.Integer ('Number')
    language_id= fields.Many2one ('language', 'Language')

    unit_ids= fields.One2many ('sentence.unit', 'sentence_id',"Tags")
    state=fields.Selection([('draft','Draft'),('corrected','Corrected'),('tagged','Tagged'),('valid','Valid')],'Status', default='draft')
    _sql_constraints = [
    ('sentence_text', 'unique(name)', 'This sentence exists.')
    ]
    #@api.multi
    def detect_language(self,model_id):


        le = LabelEncoder()
        le = load(open(model_id.file_path_label_encoder, 'rb'))
        cv = CountVectorizer()
        cv = load(open(model_id.file_path_count_vectorizer, 'rb'))
        if model_id.model_algorithm.code=='multinomial':
           model = MultinomialNB()
        else:
             return True
        model = load(open(model_id.file_path_model, 'rb'))

        lang = model.predict(cv.transform([self.name]).toarray())
        lang = le.inverse_transform(lang)
        self.language_id=self.env['language'].search([('code','=',lang[0])]).id
        return True

    def validate(self):
        self.state='valid'
        return  True
    def get_affixes(self):
        prefixes=[]
        suffixes=[]
        afx=self.env['affix'].search([('type_aff', 'in', ['prefix','suffix'])])
        #raise osv.except_osv(_('Invalid Action!'), _(afx))
        for i in afx:
            if i.type_aff=='prefix' and i not in prefixes :
             prefixes.append(i.name)
            if i.type_aff=='suffix' and i not in suffixes:
             suffixes.append(i.name)


        return prefixes,suffixes
    #@api.multi
    def tokenize_word(self,word,suffixes,prefixes):
        #a=''
        morpheme=word[0:word.find('-')+1]
        segmented_word=''
        if (morpheme in prefixes):
            word=word[word.find('-')+1:len(word)]
            segmented_word=segmented_word+' '+morpheme
            while word.find('-')>=0:

                morpheme=word[0:word.find('-')+1].strip()
                word=word[word.find('-')+1:len(word)].strip()
                segmented_word=segmented_word.strip()+' '+morpheme.strip()
            segmented_word=segmented_word.strip()+' '+word
        else:
            morpheme=word[0:word.find('-')].strip()
            segmented_word=segmented_word.strip()+' '+morpheme.strip()
            word=word[word.find('-')+1:len(word)]
            while word.find('-')>=0:

               morpheme=word[0:word.find('-')]
               segmented_word=segmented_word.strip()+' '+'-'+morpheme.strip()
               word=word[word.find('-')+1:len(word)].strip()
        if ('-'+word in suffixes):
             segmented_word=segmented_word.strip()+' '+'-'+word.strip()
        else:
             segmented_word=segmented_word.strip()
        return segmented_word
    #@api.multi
    def tokenize(self,sentence,suffixes,prefixes):
       a=sentence.split()
       sentence1=""
       for i in a: #mots
        if(i.find('-')<0):
            sentence1=sentence1+' '+i
        else:
            words=self.tokenize_word(i,suffixes,prefixes)
            sentence1=sentence1+' '+words
       sentence1=sentence1.strip()
       return sentence1
    #@api.multi
    def get_id_tag(self,i):
        ids=self.env['tag'].search([('code', 'in', [i])])
        for i in ids:
            return i.id
    #@api.multi
    def create_units(self):
        #raise osv.except_osv(_('Invalid Action!'), _(self))
        self.env['sentence.unit'].search([('sentence_id', 'in', [self.id])]).unlink()

        punctuation=['...',',',';','?','!',':','"','(',')','*','_','.','[',']','{','}','«','»','+','=','“','”']
        prefixes,suffixes=self.get_affixes()
        sentence=self.name
        for i in punctuation:
            sentence=sentence.replace(i,' '+i+' ')
        sentence=sentence.replace('  ',' ')
        sentence=sentence.replace("\ufeff","").strip()

        sentence=self.tokenize(sentence,suffixes,prefixes)
        sentence=sentence.split(" ")
        order=0
        for i in sentence:
            self.env['sentence.unit'].create({
                        'sentence_id': self.id,
                        'unit': i,
                        'tag_id':self.get_id_tag(i) if i in punctuation else '',
                        'order':order,
                     })
            order=order+1
        #raise osv.except_osv(_('Invalid Action!'), _(sentence))


        return True
    #@api.multi
    def get_tag_id(self,tag):
        tag_id=self.env['tag'].search([('code', 'in', [tag])])
        if tag_id:
         return tag_id.id
        else:
            return False


    #@api.multi
    def load_tagged_sentence(self):
        #raise osv.except_osv(_('Invalid Action!'), _(self))
        self.env['sentence.unit'].search([('sentence_id', 'in', [self.id])]).unlink()

        punctuation=['...',',',';','?','!',':','"','(',')','*','_','.','[',']','{','}','«','»','+','=','“','”']
        prefixes,suffixes=self.get_affixes()
        sentence=self.tagged_sentence
        sentence=sentence.replace('  ',' ')
        sentence=sentence.replace("\ufeff","").strip()

        #sentence=self.tokenize(sentence,suffixes,prefixes)
        sentence=sentence.split(" ")
        order=0
        for i in sentence:

            unit=i.split('/')
            try:
             unit_id=self.get_tag_id(unit[1])
            except:
                raise ValidationError(_('Unit Error please correct the syntaxe %s ', i))

            if not unit_id:
                raise ValidationError(_('Tag Error: please correct the tag %s ', unit[1]))

            #raise osv.except_osv(_('Invalid Action!'), _(unit_id))
            self.env['sentence.unit'].create({
                        'sentence_id': self.id,
                        'unit': unit[0],
                        'tag_id':unit_id,
                        'order':order,
                     })
            order=order+1
        #raise osv.except_osv(_('Invalid Action!'), _(sentence))


        return True

    def pos_tag(self,sentence,model):
        try:
         sentence_features = [features(sentence, index) for index in range(len(sentence))]
        except:
            return (sentence)
        return list(zip(sentence, model.predict([sentence_features])[0]))

    def tag2 (self,model):

        #raise ValidationError(_('Error: %s', model))
        clf = load(model)

        prefixes,suffixes=self.get_affixes()
        sentence=self.name
        punctuation=['...',',',';','?','!',':','"','(',')','*','_','.','[',']','{','}','«','»','+','=','“','”']
        for i in punctuation:
            sentence=sentence.replace(i,' '+i+' ')

        sentence=self.tokenize(sentence,suffixes,prefixes)
        sentence=sentence.replace("  "," ")
        izirig=""
        xx=self.pos_tag(sentence.split(" "),clf)
        for u in xx:

          try:
            yy=u[0]+'/'+u[1]

            izirig=izirig+yy+" "
          except:
            self.tagged_sentence=xx
            return True
        self.tagged_sentence=izirig
        return True



class sentence_unit (models.Model):

    _name = 'sentence.unit'
    _description = 'Sentence Unit'

    unit=fields.Char('Unit')
    order=fields.Integer('Ordre')
    tag_id=fields.Many2one ('tag', 'Tag', required=False)
    root_id=fields.Many2one ('root', 'Root', required=False)
    sentence_id= fields.Many2one ('sentence', 'Sentence', required=True)
    annotation_ids= fields.Many2many('annotation',
                                          'annotation_units_rel',
                                          'unit_id', 'annotation_id', 'Annotations'
                                          )

class anotation (models.Model):
    _name = 'annotation'
    _description = 'Annotation'


    name= fields.Char('Annotation name',translate=True)
    code= fields.Char('Annotation code')
    description= fields.Html('Annotation description',translate=True)



class language (models.Model):
    _name = 'language'
    _description = 'Language'


    name= fields.Char ('Language Name',translate=True)
    code= fields.Char ('Language Code')
    feature_ids= fields.Many2many  ('feature', 'lang_feature_rel', 'lang_id', 'feature_id', 'Mophological Features')
    language_family_id= fields.Many2one ('language.family', 'Language Family')
    language_script_id= fields.Many2one ('language.script', 'Language Script')


class LanguageFamily (models.Model):
    _name = 'language.family'
    _description = 'Language Family'


    name= fields.Char ('Family Name',translate=True)
    code= fields.Char ('Family Code')
    parent_family_id = fields.Many2one ('language.family', 'Parent Family')

class LanguageScript (models.Model):
    _name = 'language.script'
    _description = 'Language Script'


    name= fields.Char ('Script Name',translate=True)
    code= fields.Char ('Script Code')

class tag (models.Model):
    _name = 'tag'
    _description = 'Tag'


    name= fields.Char ('Tag Name',translate=True)
    code= fields.Char ('Tag Code')
    description = fields.Html ('Description')

class root (models.Model):
    _name = 'root'
    _description = 'Root'


    name= fields.Char ('Root')
    description = fields.Html ('Description')

class affix (models.Model):
    _name = 'affix'
    _description = 'Affix'
    language_id= fields.Many2one ('language', 'Language', required=False)
    name= fields.Char ('Affix Name')
    type_aff= fields.Selection ([('prefix', "Prefix"),('suffix', "Suffix")], 'Affix Type',default='prefix')

