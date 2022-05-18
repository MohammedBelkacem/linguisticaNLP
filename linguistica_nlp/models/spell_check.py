# -*- coding: utf-8 -*-
from odoo import api, fields, models, _, tools
from odoo.osv import expression
from odoo.exceptions import UserError, ValidationError
from joblib import dump, load
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from datetime import datetime

class Word (models.Model):

    _name = 'dictionary.word'
    _description = 'Word'

    def name_get(self):

        return [(record.id, record.word) for record in self]

    word= fields.Char ('Word',required=True)
    word_rules = fields.Char ('Word Rules')
    language_id= fields.Many2one ('language', 'Language', required=True)
    rule_ids= fields.Many2many  ('dictionary.rule', 'word_dictionary_rule_rel', 'word_id', 'rule_id', 'Rules')
    dictionary_id=fields.Many2one ('dictionary', 'Dictionary')



class Rule (models.Model):

    _name = 'dictionary.rule'
    _description = 'Word Rules'

    name=fields.Char('Rule Name')
    type_id= fields.Selection ([('PFX', "Prefix"),('SFX', "Suffix")], 'Rule Type',default='prefix')
    rule_line_ids=fields.One2many ('dictionary.rule.line', 'rule_id', 'Rules Lines')

class RuleLine (models.Model):

    _name = 'dictionary.rule.line'
    _description = 'Word Rules Line'

    pfx_rule_id = fields.Many2one ('dictionary.rule', 'Combined prefix rule')
    regular_expression_prefix = fields.Char('Prefix Regular Expression')
    regular_expression_suffix = fields.Char('Suffix Regular Expression')
    substitution_chars = fields.Char('Substitution chars')
    rule_id=fields.Many2one('dictionary.rule','Dictionary Rules')


class Dictionary (models.Model):
    _name = 'dictionary'
    _description = 'Dictionary'


    name= fields.Char('Dictionary name',translate=True)
    code= fields.Char('Dictionary code')
    description= fields.Html('Dictionary description',translate=True)
    language_id= fields.Many2one ('language', 'Language', required=True)
    word_ids = fields.Many2many  ('word', 'word_dictionary_rel', 'dictionay_id', 'word_id', 'Words')
