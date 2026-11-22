import json
from pathlib import Path
import unittest

from modules.class_names_ru import to_ru, to_ru_list


CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def config(relative_path):
    return json.loads((CONFIGS / relative_path).read_text(encoding='utf-8'))


class ClassNamesRuTests(unittest.TestCase):
    def test_all_published_class_names_have_distinct_russian_labels(self):
        for names in (
            config('evaluation/hymars_hc_mica_sources.json')['products'],
            config('evaluation/hymars_plebani_taxonomy.json')['class_names'],
            config('datasets/plebani_taxonomy_reference.json')['raw_names'].values(),
            config('models/lwnet_salinas/bundle.json')['taxonomy']['class_names'],
        ):
            labels = to_ru_list(names)
            self.assertEqual(len(set(labels)), len(names))
            for name, label in zip(names, labels):
                with self.subTest(name=name):
                    self.assertRegex(label, r'[А-Яа-яЁё]')

    def test_aliases_order_and_lossless_fallback(self):
        names = ('High-Ca Pyroxene', 'crism_typespec_HCP.tab', 'HCP', 'New Class <B>',
                 'MgCO₃', 'Feldspar Plagioclase', 'Vineyard_untrained', '  Unknown-A  ')
        self.assertEqual(to_ru_list(names), [
            'Высококальциевый пироксен', 'Высококальциевый пироксен',
            'Высококальциевый пироксен', 'New Class <B>', 'Mg-карбонат',
            'Плагиоклаз', 'Виноградник без формировки', '  Unknown-A  ',
        ])
        self.assertEqual(to_ru('Saponite'), 'Сапонит')
        self.assertNotEqual(to_ru('Saponite'), to_ru('Mg-Smectite'))
