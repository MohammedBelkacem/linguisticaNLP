# -*- coding: utf-8 -*-
from odoo import api, fields, models, _, tools
from odoo.osv import expression
from odoo.exceptions import UserError, ValidationError
from joblib import dump, load
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from datetime import datetime
import nltk

class rulegrammar (models.Model):
      _name='grammar.rule'
      _description='Grammar Rule'

      name=fields.Char("Rule Name")
      code=fields.Char("Rule Code")
      grammar_structure = fields.Char('Grammar Structure')
      language_id=fields.Many2one('language','Language')
      grammar_child_rule_ids = fields.One2many('grammar.rule.child', 'parent_rule',"Tags")
      trivial=fields.Boolean('Trivial?')

      def name_get(self):

        return [(record.id, record.code) for record in self]

      def create_child_rules(self):
          rule_structure = self.grammar_structure.split("|")
          obj_rules=[]
          for rule_lines in rule_structure:
              rules=rule_lines.replace("  "," ").strip().split(" ")
              for rule in rules:
                  id_rule=self.search([('code','=',rule)])
                  if id_rule.id and id_rule not in obj_rules:
                     obj_rules.append(id_rule)
                     self.env['grammar.rule.child'].create({'parent_rule':self.id,'child_rule':id_rule.id})
          #raise ValidationError(_('Rules %s ', obj_rules))




class ChildRule(models.Model):
      _name='grammar.rule.child'
      _description='Child Grammar Rule'
      parent_rule=fields.Many2one('grammar.rule','Parent rule')
      child_rule=fields.Many2one('grammar.rule','Child Rule')
      name=fields.Char("Rule Name")
      code=fields.Char("Rule Code")
      grammar_structure = fields.Char('Grammar Structure')
      language_id=fields.Many2one('language','Language')

      @api.onchange('child_rule')
      def onchange_parent_rule(self):
          self.name=self.child_rule.name
          self.code=self.child_rule.code
          self.grammar_structure=self.child_rule.grammar_structure
          self.language_id=self.parent_rule.language_id.id


class grammar (models.Model):
      _name='grammar'
      _description='Grammar'

      name=fields.Char('Grammar Name')
      language_id=fields.Many2one('language','Language')
      grammar_rules = fields.Many2many('grammar.rule', 'grammar_rule_grammar_rel', 'grammar_id', 'rule_id', 'Grammar rules')
      cfg=fields.Text("Context Free Grammar")
      parsed_sentence = fields.Text ("Parsed Sentence")

      def generate_cfg_grammar(self):
          cfg=""
          for rec in self.grammar_rules:
              cfg=cfg+rec.code+' -> ' + rec.grammar_structure +'\n'
          self.cfg=cfg
          return True
      def parse_sentence(self):

            sent = ['walaɣ','argaz','-a','s','nwaḍer']

            cfg='"""\n'+self.cfg+'\n"""'
            parser = nltk.ChartParser(cfg)
            raise ValidationError(_('Unit Error please correct the syntaxe %s ', parser))
            #print (parser)

            a=''
##            for tree in parser.parse(sent):
##                a=a+str(tree)

            self.parsed_sentence=cfg
