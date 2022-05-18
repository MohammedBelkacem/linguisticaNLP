# -*- coding: utf-8 -*-
{
    'name': "linguistica_nlp",

    'summary': """
        Linguistica, a set of natural
        language processing dealing with mosrphology,
        syntax, grammar and some statistical analysis""",

    'description': """
        Free Natural language processing
    """,

    'author': "Mohammed Belkacem",
    'website': "https://www.linkedin.com/in/belkacem-mohammed/ , https://www.youtube.com/c/MohammedBelkacem/videos, https://github.com/MohammedBelkacem",

    'category': 'Linguistics',
    'version': '0.1',
    'depends': ['base'],

    # always loaded
    'data': [
        'security/ir.model.access.csv',


        'views/sentence_view.xml',
        'views/corpus_view.xml',
        'views/configuration_postag_view.xml',
        'views/configuration_training_view.xml',
        'views/spell_check.xml',
        'views/lang_detection.xml',
        'views/grammar.xml',
        'data/annotation_data.xml',
        'data/feature_data.xml',
        'data/language_data.xml',
        'data/parameter_data.xml',
        'data/algorithm_data.xml',
        'data/tagged_sentences_data.xml',
        'data/affix_data.xml',
        'data/language_family_data.xml',
        'data/language_script_data.xml',
        'data/tag_data.xml',
        'wizard/trained_model_selection_view.xml',
         'wizard/validate_sentence.xml',
         'wizard/lang_model_selection_view.xml',

         'reports/corpus.xml'
    ],
    # only loaded in demonstration mode
    'demo': [

    ],
    'installable': True,
    'application': True,
    'auto_install': False,
    'license': 'LGPL-3',


}
