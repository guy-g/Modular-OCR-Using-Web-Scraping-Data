import os


settings = {
    'Paths': {
        'Databases': {
            'Datasets': {},
            'DatasetsVisualizations': {},
            'ExternalDatasets': {
                'Image Translation Data': {},
                'OCR': {},
                'OCR_TESTS': {}
            },
            'Dictionaries': {},
            'Fonts': {},
            'CycleGAN': {},
            'MUNIT': {},
            'Languages': {},
            'Backgrounds': {
                'Natural': {},
                'Texture': {}
            },
            'WebLinks': {},
            'OfflineWeb': {},
            'Corpuses': {}
        },
        'Database Views': {
            'Datasets': {},
            'Dictionaries': {},
            'Fonts': {},
        },
        'Models': {
            'Detections': {},
            'Recognitions': {},
            'Layout Analysis': {},
            'Ocr Pipelines': {
                'Training': {},
                'Inference': {
                    'Pipelines': {},
                    'Parameters Search': {}
                }
            },
            'CycleGAN': {},
            'MUNIT': {},
            'Script Identification': {}
        },
        'tmp': {}
    },
}


def __create_paths(storage_path, paths=settings['Paths']):
    for folder, sub_folders in paths.items():
        os.makedirs(os.path.join(storage_path, folder), exist_ok=True)
        for sub_folder in sub_folders.keys():
            os.makedirs(os.path.join(storage_path, folder, sub_folder), exist_ok=True)
            __create_paths(os.path.join(storage_path, folder, sub_folder), sub_folders[sub_folder])


def __create_settings_dict(storage_path, chromedriver_path):
    settings_dict = {
        'STORAGE_PATH': storage_path,
        'WEB_LINKS_PATH': os.path.join(storage_path, 'Databases', 'WebLinks'),
        'DATASETS_PATH': os.path.join(storage_path, 'Databases', 'Datasets'),
        'DATASETS_VISUALIZATIONS_PATH': os.path.join(storage_path, 'Databases', 'DatasetsVisualizations'),
        'OFFLINE_WEB_PATH': os.path.join(storage_path, 'Databases', 'OfflineWeb'),
        'CORPUSES_PATH': os.path.join(storage_path, 'Databases', 'Corpuses'),
        'SCRIPT_IDENTIFICATION_MODELS_PATH': os.path.join(storage_path, 'Models', 'Script Identification'),
        'DATASETS_VIEWS_PATH': os.path.join(storage_path, 'Database Views', 'Datasets'),
        'DICTIONARIES_PATH': os.path.join(storage_path, 'Databases', 'Dictionaries'),
        'DICTIONARIES_VIEWS_PATH': os.path.join(storage_path, 'Database Views', 'Dictionaries'),
        'FONTS_PATH': os.path.join(storage_path, 'Databases', 'Fonts'),
        'FONTS_VIEWS_PATH': os.path.join(storage_path, 'Database Views', 'Fonts'),
        'LANGUAGES_PATH': os.path.join(storage_path, 'Databases', 'Languages'),
        'EXTERNAL_DATASETS_PATH': os.path.join(storage_path, 'Databases', 'ExternalDatasets'),
        'EXTERNAL_DATASETS_OCR_PATH': os.path.join(storage_path, 'Databases', 'ExternalDatasets', 'OCR'),
        'EXTERNAL_DATASETS_OCR_TESTS_PATH': os.path.join(storage_path, 'Databases', 'ExternalDatasets', 'OCR_TESTS'),
        'EXTERNAL_DATASETS_IMAGE_TRANSLATION_DATA_PATH': os.path.join(storage_path, 'Databases', 'ExternalDatasets', 'Image Translation Data'),
        'CYCLEGAN_DATASET_PATH': os.path.join(storage_path, 'Databases', 'CycleGAN'),
        'CYCLEGAN_MODELS_PATH': os.path.join(storage_path, 'Models', 'CycleGAN'),
        'MUNIT_DATASET_PATH': os.path.join(storage_path, 'Databases', 'MUNIT'),
        'MUNIT_MODELS_PATH': os.path.join(storage_path, 'Models', 'MUNIT'),
        'DETECTION_MODELS_PATH': os.path.join(storage_path, 'Models', 'Detections'),
        'RECOGNITION_MODELS_PATH': os.path.join(storage_path, 'Models', 'Recognitions'),
        'LAYOUT_ANALYSIS_MODELS_PATH': os.path.join(storage_path, 'Models', 'Layout Analysis'),
        'OCR_PIPELINES_TRAINING_PATH': os.path.join(storage_path, 'Models', 'Ocr Pipelines', 'Training'),
        'OCR_PIPELINES_INFERENCE_PATH': os.path.join(storage_path, 'Models', 'Ocr Pipelines', 'Inference', 'Pipelines'),
        'OCR_PIPELINES_INFERENCE_PARAMETERS_SEARCH_PATH': os.path.join(storage_path, 'Models', 'Ocr Pipelines', 'Inference', 'Parameters Search'),
        'NATURAL_BACKGROUNDS_PATH': os.path.join(storage_path, 'Databases', 'Backgrounds', 'Natural'),
        'TEXTURE_BACKGROUNDS_PATH': os.path.join(storage_path, 'Databases', 'Backgrounds', 'Texture'),
        'TMP_PATH': os.path.join(storage_path, 'tmp'),
        'CHROMEDRIVER_PATH': chromedriver_path
    }
    return settings_dict


def create_settings(storage_path, chromedriver_path):
    __create_paths(storage_path, paths=settings['Paths'])
    settings_dict = __create_settings_dict(storage_path, chromedriver_path)
    with open(os.path.join('Config', 'setting.py'), mode='w', encoding='utf-8') as f:
        for k, v in settings_dict.items():
            f.write('{} = r\"{}\"\n'.format(k, v))


if __name__ == '__main__':
    import sys
    create_settings(sys.argv[1], sys.argv[2])

