"""Русские подписи для визуализации классов MICA, Plebani/HyMars и Salinas."""

import re
import unicodedata

# Маппинг: английское имя → русское название.
RU_NAMES = {
    'al_smectite':      'Al-смектит',
    'alunite':          'Алунит',
    'analcime':         'Анальцим',
    'bassanite':        'Бассанит',
    'chloride':         'Хлорид',
    'chlorite':         'Хлорит',
    'co2_ice':          'Углекислотный лёд',
    'epidote':          'Эпидот',
    'fe_ca_carbonate':  'Fe/Ca-карбонат',
    'fe_olivine':       'Fe-оливин',
    'fe_smectite':      'Fe-смектит',
    'gypsum':           'Гипс',
    'h2o_ice':          'Водяной лёд',
    'hcp':              'Высококальциевый пироксен',
    'hematite':         'Гематит',
    'hydrated_silica':  'Гидратированный кремнезём',
    'hydrox_fe_sulf':   'Гидроксилированный Fe-сульфат',
    'illite_muscovite': 'Иллит / мусковит',
    'jarosite':         'Ярозит',
    'kaolinite':        'Каолинит',
    'lcp':              'Низкокальциевый пироксен',
    'margarite':        'Маргарит',
    'mg_carbonate':     'Mg-карбонат',
    'mg_olivine':       'Mg-оливин',
    'mg_smectite':      'Mg-смектит',
    'mono_hyd_sulf':    'Моногидрат сульфата',
    'plagioclase':      'Плагиоклаз',
    'poly_hyd_sulf':    'Полигидрат сульфата',
    'prehnite':         'Пренит',
    'serpentine':       'Серпентин',
    'talc':             'Тальк',
    'ferric_hydroxysulfate': 'Гидроксосульфат железа(III)',
    'nontronite': 'Нонтронит',
    'saponite': 'Сапонит',
    'akaganeite': 'Акаганеит',
    'beidellite': 'Бейделлит',
    'montmorillonite': 'Монтмориллонит',
    'rosenite': 'Розенит',
    'mg_cl_salt': 'Хлорсодержащая соль магния',
    'halloysite': 'Галлуазит',
    'epsomite': 'Эпсомит',
    'opal_1': 'Опал 1',
    'opal_2': 'Опал 2',
    'iron_oxide_silicate_sulfate': 'Оксид железа, силикат и сульфат',
    'clinochlore': 'Клинохлор',
    'olivine_forsterite': 'Оливин (форстерит)',
    'olivine_fayalite': 'Оливин (фаялит)',
    'artifact': 'Артефакт',
    'neutral': 'Нейтральный спектр',
    'brocoli_green_weeds_1': 'Брокколи с зелёными сорняками, участок 1',
    'brocoli_green_weeds_2': 'Брокколи с зелёными сорняками, участок 2',
    'fallow': 'Поле под паром',
    'fallow_rough_plow': 'Грубовспаханный пар',
    'fallow_smooth': 'Выровненный пар',
    'stubble': 'Стерня',
    'celery': 'Сельдерей',
    'grapes_untrained': 'Виноград без формировки',
    'soil_vinyard_develop': 'Почва молодого виноградника',
    'corn_senesced_green_weeds': 'Увядающая кукуруза с зелёными сорняками',
    'lettuce_romaine_4wk': 'Салат ромэн, 4 недели',
    'lettuce_romaine_5wk': 'Салат ромэн, 5 недель',
    'lettuce_romaine_6wk': 'Салат ромэн, 6 недель',
    'lettuce_romaine_7wk': 'Салат ромэн, 7 недель',
    'vinyard_untrained': 'Виноградник без формировки',
    'vinyard_vertical_trellis': 'Виноградник на вертикальной шпалере',
}

_ALIASES = {
    'hydroxylated_fe_sulfate': 'hydrox_fe_sulf',
    'ca_fe_co3': 'fe_ca_carbonate',
    'mgco3': 'mg_carbonate',
    'feldspar_plagioclase': 'plagioclase',
    'low_ca_pyroxene': 'lcp',
    'high_ca_pyroxene': 'hcp',
    'monohydrated_sulfate': 'mono_hyd_sulf',
    'polyhydrated_sulfate': 'poly_hyd_sulf',
    'neutral_bland': 'neutral',
    'broccoli_green_weeds_1': 'brocoli_green_weeds_1',
    'broccoli_green_weeds_2': 'brocoli_green_weeds_2',
    'soil_vineyard_develop': 'soil_vinyard_develop',
    'vineyard_untrained': 'vinyard_untrained',
    'vineyard_vertical_trellis': 'vinyard_vertical_trellis',
}

def to_ru(name: str) -> str:
    """
    Возвращает русское название по английскому ключу.
    Если ключа нет — возвращает исходное имя (без падения).
    """
    key = unicodedata.normalize('NFKC', name).strip().casefold()
    key = re.sub(r'[\s/‐‑–-]+', '_', key)
    if key.startswith('crism_typespec_'):
        key = key.removeprefix('crism_typespec_').removesuffix('.tab').removesuffix('.lbl')
    return RU_NAMES.get(_ALIASES.get(key, key), name)


def to_ru_list(names: list) -> list:
    """Переводит список английских имён в список русских."""
    return [to_ru(n) for n in names]
