# -*- encoding: utf-8 -*-
##############################################################################
#
#    OpenERP, Open Source Management Solution
#    Copyright (C) 2004-2009 Tiny SPRL (<http://tiny.be>). All Rights Reserved
#    $Id$
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public Licensentencese
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################
from odoo import _, api, fields, models
from odoo.exceptions import UserError, ValidationError



class ValidateSentence(models.TransientModel):
    _name = 'sentence.validate'
    _description = 'Validate Sentence'

    def validate_sentence(self):
        #raise ValidationError(_('You cannot have a receivable/payable account that is not reconcilable. (account code: %s)', self.env.context.get('active_ids', [])))

        for rec in self.env.context.get('active_ids', []) :

            s=self.env['sentence'].browse(rec)
            #raise ValidationError(_('You cannot have a receivable/payable account that is not reconcilable. (account code: %s)', rec))
            s.validate()

        return True
