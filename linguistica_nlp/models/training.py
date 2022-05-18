# -*- coding: utf-8 -*-
from odoo import api, fields, models, _, tools
from odoo.osv import expression
from odoo.exceptions import UserError, ValidationError
from joblib import dump, load
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from datetime import datetime
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


def untag(tagged_sentence):
    """
    Given a tagged sentence, return an untagged version of that
    sentence.  I.e., return a list containing the first element
    of each tuple in *tagged_sentence*.

        >>> from nltk.tag.util import untag
        >>> untag([('John', 'NNP'), ('saw', 'VBD'), ('Mary', 'NNP')])
        ['John', 'saw', 'Mary']

    """
    return [w for (w, t) in tagged_sentence]

def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for tagged in tagged_sentences:
        X.append([features(untag(tagged), index) for index in range(len(tagged))])
        y.append([tag for _, tag in tagged])

    return X, y


class feature (models.Model):
    _name = 'feature'
    _description = 'Feature'


    name= fields.Selection ([
 ('word','Word itself.'),
 ('is_first','If the word begins the sentence.'),
 ('is_last', 'If the word begins the sentence.'),
 ('is_capitalized','If the word begins with capital letter'),
 ('is_all_caps','If the word is all capitalized'),
 ('is_all_lower','If the word is lowerized'),
 ('prefix_1','1 st letter'),
 ('prefix_2','2 fisrt letters'),
 ('prefix_3','3 fisrt letters'),
 ('prefix_4','4 fisrt letters'),
 ('prefix_5','5 fisrt letters'),
 ('suffix_1','1 last letter'),
 ('suffix_2','2 last letters'),
 ('suffix_3','3 last letters'),
 ('suffix_4','4 last letters'),
 ('suffix_5','5 last letters'),
 ('prev_word','Previous word'),
 ('prev1_word','The second previous word'),
 ('next_word', 'Next word'),
 ('is_numeric','Second next word'),
 ('capitals_inside','If has capital lettre inside'),
], 'Feature name')
    description= fields.Html('Feature description')
    language_ids= fields.Many2many  ('language', 'lang_feature_rel', 'feature_id', 'lang_id', 'Languages')




class Algorithm (models.Model):
    _name = 'algorithm'
    _description = 'Algorithm Type'


    name= fields.Selection ([
 ('lbfgs', 'Gradient descent using the L-BFGS method'),
 ('l2sgd' , 'Stochastic Gradient Descent with L2 regularization term'),
 ('ap', 'Averaged Perceptron'),
 ('pa' ,'Passive Aggressive (PA)'),
 ('arow', 'Adaptive Regularization Of Weight Vector (AROW)'),
 ('tensorflow','Keras/TensorFlow'),
 ('pytorch','Pytorch'),
 ('gaussian','Gaussian Naive Bayes'),
 ('multinomial','Multinomial Naive Bayes'),
 ('complement','Complement Naive Bayes'),
 ('bernoulli','Bernoulli Naive Bayes'),
 ('categorical','Categorical Naive Bayes'),
  ],'Algorithm Name')
    code= fields.Char('Code')
    description = fields.Html('Description',translate=True)
    parameter_ids = fields.One2many ('algorithm.parameter','algorithm_id','Parameters')
    target_analysis= fields.Selection ([
 ('postag', 'Part Of Speech Tagging'),
 ('lang_detection' , 'Language Detection'),
   ],'Target Analysis')



class AlgorithmParameter (models.Model):
    _name = 'algorithm.parameter'
    _description = 'Algorithm Parameter'


    parameter_id= fields.Many2one('parameter')
    algorithm_id= fields.Many2one('algorithm')
    value= fields.Char('Value')

1
class Parameter (models.Model):
    _name = 'parameter'
    _description = 'Parameter'


    name= fields.Char('Name')
    description= fields.Text('Description',translate=True)


class Text (models.Model):
    _name = 'text'
    _description = 'Text'


    title= fields.Char('Title')
    content= fields.Text('Content')
    author= fields.Char('Author')
    release_date= fields.Date('Release date')
    type_id= fields.Many2one ('text.type', 'Type', required=True)
    style_id= fields.Many2one ('text.style', 'Style', required=True)

class Texttype (models.Model):
    _name = 'text.type'
    _description = 'Text Type'
    name= fields.Char('Name',translate=True)
    code= fields.Text('Code')

class Textstyle (models.Model):
    _name = 'text.style'
    _description = 'Text Style'
    name= fields.Char('Name',translate=True)
    code= fields.Text('Code')


class Corpus (models.Model):
    _name = 'corpus'
    _description = 'Corpus'
    def _get_valid_sentences(self):
     sentences = self.env['sentence'].search([('state','=','valid')])
     if sentences:
        domain = [('id', 'in', sentences.ids)]
     else:
        domain = [('id', '=', -1)]
     return domain

    name= fields.Char('Corpus Name',translate=True)
    model_ids= fields.One2many ('model','corpus_id','Models')
    sentences_ids= fields.Many2many  ('sentence', 'corpus_sentence_rel', 'corpus_id', 'sentence_id', 'Training Params',domain=_get_valid_sentences)
    language_id=fields.Many2one('language','Corpus Language')


class Model(models.Model):
    _name='model'
    _description="Pos Tag Model"
    def _get_valid_algorithms(self):
        domain = [('target_analysis', '=', 'postag')]
        return domain

    model_name = fields.Char("Model Name",translate=True)
    model_algorithm = fields.Many2one ('algorithm','Algorithm',domain=_get_valid_algorithms)
    file_model = fields.Binary('Model File')
    file_path = fields.Char('File Path')
    param_ids= fields.One2many ('model.parameter','model_id','Parameters')
    feature_ids= fields.One2many ('model.feature','model_id','Parameters')
    corpus_id=fields.Many2one('corpus','Corpus')
    model_accuracy = fields.Float ('model_accuracy')
    flat_f1_score= fields.Text ('Model Score')
    flat_accuracy_score = fields.Text ('Flat accuracy score')
    flat_precision_score = fields.Text ('flat precision score')
    flat_classification_report = fields.Text ('flat classification report')
    sequence_accuracy_score = fields.Text ('sequence accuracy score')
    #construct the corpus [[(),()],..]
    def name_get(self):

        return [(record.id, record.model_name) for record in self]


    def load_algo_default_parameters (self):
        for param in self.model_algorithm.parameter_ids:
             self.env['model.parameter'].create({'model_id':self.id,'param_id':param.parameter_id.id,'value':param.value})

        return True

    def load_lang_default_features (self):
        for param in self.model_algorithm.parameter_ids:
             self.env['model.parameter'].create({'model_id':self.id,'param_id':param.parameter_id.id,'value':param.value})

        return True


    def train_pos_tag_model(self):
        algo=self.model_algorithm.name
        if algo=='pa':
           return self.pa_model()
        if algo=='ap':
           return self.ap_model()
        if algo=='l2sgd':
           return self.l2sgd_model()
        if algo=='lbfgs':
           return self.lbfgs_model()
        if algo=='arow':
           return self.arow_model()

    def ap_model (self):
        tagged_sentences=[]
        for line in self.corpus_id.sentences_ids:
                sentence=[]
                line=line.tagged_sentence.split()

                for element in line:
                    word_tag=element.split('/')
                    try:
                      unit_tupple=(word_tag[0],word_tag[1])
                    except:
                      raise ValidationError(_('Error: %s', line))

                    sentence.append(unit_tupple)

                tagged_sentences.append(sentence)
        total=int(len(tagged_sentences)*0.80)
        X1_train, y1_train = transform_to_dataset(tagged_sentences[:total])
        X_test, y_test = transform_to_dataset(tagged_sentences[total:])
        model = CRF(
            algorithm='ap',
            min_freq=2, #min des occurences des features au dessous du min on ignore
            all_possible_states=True,
            all_possible_transitions=True,
            max_iterations=100,# lbfgs - unlimited; l2sgd - 1000; ap - 100; pa - 100; arow - 100.
            epsilon=1e-5,# ap, arow, lbfgs, pa
            )
        #entrainement
        model.fit(X1_train, y1_train)
        date=self.model_algorithm.name+'_'+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        file='/odoo/odoo-server/addons/linguistica_nlp/models/trained_models/model_'+date
        dump(model, file)
        self.file_path=file
        return True
    def pa_model (self):
        tagged_sentences=[]
        for line in self.corpus_id.sentences_ids:
                sentence=[]
                line=line.tagged_sentence.split()

                for element in line:
                    word_tag=element.split('/')
                    try:
                      unit_tupple=(word_tag[0],word_tag[1])
                    except:
                      raise ValidationError(_('Error: %s)', line))

                    sentence.append(unit_tupple)

                tagged_sentences.append(sentence)
        total=int(len(tagged_sentences)*0.80)
        X1_train, y1_train = transform_to_dataset(tagged_sentences[:total])
        X_test, y_test = transform_to_dataset(tagged_sentences[total:])
        model = CRF(
            algorithm='pa',
            min_freq=2, #min des occurences des features au dessous du min on ignore
            all_possible_states=True,
            all_possible_transitions=True,
            max_iterations=100,# lbfgs - unlimited; l2sgd - 1000; ap - 100; pa - 100; arow - 100.
            epsilon=1e-5,# ap, arow, lbfgs, pa
            )
        #entrainement
        model.fit(X1_train, y1_train)
        date=self.model_algorithm.name+'_'+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        file='/odoo/odoo-server/addons/linguistica_nlp/models/trained_models/model_'+date
        labels = list(model.classes_)
#labels.remove('O')
        Ponctuation= [ ';', '(', ')', '-','.','!','?','"',':','$',',']
        for i in Ponctuation:
            labels.remove(i)

        dump(model, file)
        self.file_path=file
        return True
    def lbfgs_model (self):
        tagged_sentences=[]
        for line in self.corpus_id.sentences_ids:
                sentence=[]
                line=line.tagged_sentence.split()

                for element in line:
                    word_tag=element.split('/')
                    try:
                      unit_tupple=(word_tag[0],word_tag[1])
                    except:
                      raise ValidationError(_('Error: %s', line))

                    sentence.append(unit_tupple)

                tagged_sentences.append(sentence)
        total=int(len(tagged_sentences)*0.80)
        X1_train, y1_train = transform_to_dataset(tagged_sentences[:total])
        X_test, y_test = transform_to_dataset(tagged_sentences[total:])
        model = CRF(
            algorithm='lbfgs',
            min_freq=2, #min des occurences des features au dessous du min on ignore
            all_possible_states=True,
            all_possible_transitions=True,
            max_iterations=100,# lbfgs - unlimited; l2sgd - 1000; ap - 100; pa - 100; arow - 100.
            epsilon=1e-5,# ap, arow, lbfgs, pa
            )
        #entrainement
        model.fit(X1_train, y1_train)
        date=self.model_algorithm.name+'_'+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        file='/odoo/odoo-server/addons/linguistica_nlp/models/trained_models/model_'+date
        dump(model, file)
        self.file_path=file
        return True
    def l2sgd_model (self):
        tagged_sentences=[]
        for line in self.corpus_id.sentences_ids:
                sentence=[]
                line=line.tagged_sentence.split()

                for element in line:
                    word_tag=element.split('/')
                    try:
                      unit_tupple=(word_tag[0],word_tag[1])
                    except:
                      raise ValidationError(_('Error: %s', line))

                    sentence.append(unit_tupple)

                tagged_sentences.append(sentence)
        total=int(len(tagged_sentences)*0.80)
        X1_train, y1_train = transform_to_dataset(tagged_sentences[:total])
        X_test, y_test = transform_to_dataset(tagged_sentences[total:])
        model = CRF(
            algorithm='l2sgd',
            min_freq=2, #min des occurences des features au dessous du min on ignore
            all_possible_states=True,
            all_possible_transitions=True,

            #c1=0.1,
            c2=0.01,
            max_iterations=100,# lbfgs - unlimited; l2sgd - 1000; ap - 100; pa - 100; arow - 100.
            #num_memories=6,
            #epsilon=1e-5,# ap, arow, lbfgs, pa
            period=10, # The duration of iterations to test the stopping criterion. l2sgd, lbfgs
            delta=1e-5, # l2sgd, lbfgs
            #linesearch= 'MoreThuente',#L-BFGS values MoreThuente Backtracking StrongBacktracking
            #max_linesearch=20,# The maximum number of trials for the line search algorithm. lbfgs
            #calibration_eta=0.1, # l2sgd
                    )
        #entrainement
        model.fit(X1_train, y1_train)
        date=self.model_algorithm.name+'_'+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        file='/odoo/odoo-server/addons/linguistica_nlp/models/trained_models/model_'+date
        dump(model, file)
        self.file_path=file
        labels = list(model.classes_)
#labels.remove('O')
        Ponctuation= [ ';', '(', ')', '-','.','!','?','"',':','$',',']
        for i in Ponctuation:
            labels.remove(i)

        y_pred = model.predict(X_test)
        self.flat_f1_score = metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)
        self.flat_accuracy_score=metrics.flat_accuracy_score(y_test, y_pred)
        self.sequence_accuracy_score=metrics.sequence_accuracy_score(y_test, y_pred)
        #self.flat_classification_report=metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3)

        return True

    def arow_model (self):
        tagged_sentences=[]
        for line in self.corpus_id.sentences_ids:
                sentence=[]
                line=line.tagged_sentence.split()

                for element in line:
                    word_tag=element.split('/')
                    try:
                      unit_tupple=(word_tag[0],word_tag[1])
                    except:
                      raise ValidationError(_('Error: %s', line))

                    sentence.append(unit_tupple)

                tagged_sentences.append(sentence)
        total=int(len(tagged_sentences)*0.80)
        X1_train, y1_train = transform_to_dataset(tagged_sentences[:total])
        X_test, y_test = transform_to_dataset(tagged_sentences[total:])
        model = CRF(
            algorithm='arow',
            min_freq=2, #min des occurences des features au dessous du min on ignore
            all_possible_states=True,
            all_possible_transitions=True,
            max_iterations=100,# lbfgs - unlimited; l2sgd - 1000; ap - 100; pa - 100; arow - 100.
            epsilon=1e-5,# ap, arow, lbfgs, pa
            )
        #entrainement
        model.fit(X1_train, y1_train)
        date=self.model_algorithm.name+'_'+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        file='/odoo/odoo-server/addons/linguistica_nlp/models/trained_models/model_'+date
        dump(model, file)
        self.file_path=file
        return True


class ModelParameter (models.Model):
    _name = 'model.parameter'
    _description = 'Model parameters'

    param_id= fields.Many2one ('parameter','Parameter')
    value= fields.Char  ('Value')
    model_id=fields.Many2one('model','Model')

class ModelFeature (models.Model):
    _name = 'model.feature'
    _description = 'Model Features'

    feature_id= fields.Many2one ('feature','Feature')

    model_id=fields.Many2one('model','Model')


##class langmodel (models.Model):
##    _name = 'langmodel'
##    _description = 'Language Model'
##
##
##    algorithm= fields.Char('Isem')
##    description= fields.Html('Aseglem')
##

#    unit_id= fields.Many2one ('sentence.unit', 'Unit')


