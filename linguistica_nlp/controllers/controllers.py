# -*- coding: utf-8 -*-
# from odoo import http


# class LinguisticaNlp(http.Controller):
#     @http.route('/linguistica_nlp/linguistica_nlp', auth='public')
#     def index(self, **kw):
#         return "Hello, world"

#     @http.route('/linguistica_nlp/linguistica_nlp/objects', auth='public')
#     def list(self, **kw):
#         return http.request.render('linguistica_nlp.listing', {
#             'root': '/linguistica_nlp/linguistica_nlp',
#             'objects': http.request.env['linguistica_nlp.linguistica_nlp'].search([]),
#         })

#     @http.route('/linguistica_nlp/linguistica_nlp/objects/<model("linguistica_nlp.linguistica_nlp"):obj>', auth='public')
#     def object(self, obj, **kw):
#         return http.request.render('linguistica_nlp.object', {
#             'object': obj
#         })
