from deep_translator import GoogleTranslator


def translate_text(text):

    return GoogleTranslator(

        source='en',

        target='ur'

    ).translate(text)