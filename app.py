import gradio as gr
from synthesizer import Synthesizer

TTS_LANGUAGES = {
    "af": "Afrikaans",
    "sq": "Albanian",
    "am": "Amharic",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "ny": "Chichewa",
    "zh": "Chinese",
    "co": "Corsican",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "eo": "Esperanto",
    "et": "Estonian",
    "tl": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Frisian",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "ht": "Haitian Creole",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "iw": "Hebrew",
    "hi": "Hindi",
    "hmn": "Hmong",
    "hu": "Hungarian",
    "is": "Icelandic",
    "ig": "Igbo",
    "id": "Indonesian",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "km": "Khmer",
    "rw": "Kinyarwanda",
    "ko": "Korean",
    "ku": "Kurdish (Kurmanji)",
    "ky": "Kyrgyz",
    "lo": "Lao",
    "la": "Latin",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "lb": "Luxembourgish",
    "mk": "Macedonian",
    "mg": "Malagasy",
    "ms": "Malay",
    "ml": "Malayalam",
    "mt": "Maltese",
    "mi": "Maori",
    "mr": "Marathi",
    "mn": "Mongolian",
    "my": "Myanmar (Burmese)",
    "ne": "Nepali",
    "no": "Norwegian",
    "or": "Odia (Oriya)",
    "ps": "Pashto",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pa": "Punjabi",
    "ro": "Romanian",
    "ru": "Russian",
    "sm": "Samoan",
    "gd": "Scots Gaelic",
    "sr": "Serbian",
    "st": "Sesotho",
    "sn": "Shona",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "es": "Spanish",
    "su": "Sundanese",
    "sw": "Swahili",
    "sv": "Swedish",
    "tg": "Tajik",
    "ta": "Tamil",
    "tt": "Tatar",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "tk": "Turkmen",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "ug": "Uyghur",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "cy": "Welsh",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zu": "Zulu"
}

TTS_EXAMPLES = [
    ["Hello! This is Interlink AI TTS Agent. Our advanced voice synthesis technology ensures natural and clear speech, making communication more efficient and accessible.", "eng English", "target_audio/male_01.wav"],
    ["Hola! Este es el Agente TTS de Interlink AI. Nuestra avanzada tecnología de síntesis de voz garantiza un habla natural y clara, haciendo la comunicación más eficiente y accesible.", "spa Spanish", "target_audio/male_02.wav"],
    ["Bonjour! Voici l'Agent TTS d'Interlink AI. Notre technologie avancée de synthèse vocale assure une parole naturelle et claire, rendant la communication plus efficace et accessible.", "fra French", "target_audio/female_02.wav"],
    ["你好！这是 Interlink AI TTS 代理。我们的先进语音合成技术确保自然清晰的语音，使沟通更加高效和便捷。", "zho Chinese", "target_audio/male_03.wav"],
    ["안녕하세요! Interlink AI TTS 에이전트입니다. 저희의 첨단 음성 합성 기술은 자연스럽고 명확한 발음을 보장하여 보다 효율적이고 접근 가능한 의사소통을 가능하게 합니다.", "kor Korean", "target_audio/female_03.wav"],
    ["مرحبًا! هذا هو وكيل Interlink AI TTS. تضمن تقنيتنا المتقدمة لتركيب الصوت نطقًا طبيعيًا وواضحًا، مما يجعل التواصل أكثر كفاءة وسهولة.", "ara Arabic", "target_audio/male_04.wav"],
    ["Interlink AI TTS Agent မှ မင်္ဂလာပါ။ ကျွန်ုပ်တို့၏ အဆင့်မြင့် အသံဖြန့်စည်းမှုနည်းပညာသည် သဘာဝကျသော၊ ရှင်းလင်းသော အသံကို သေချာစေပြီး ဆက်သွယ်မှုကို ပိုမို ထိရောက်စေသည်။", "mya Burmese", "target_audio/female_04.wav"]
    ["Xin chào! Đây là Interlink AI TTS Agent. Công nghệ tổng hợp giọng nói tiên tiến của chúng tôi đảm bảo giọng nói tự nhiên và rõ ràng, giúp giao tiếp hiệu quả và dễ dàng hơn.", "vie Vietnamese", "target_audio/female_01.wav"],
]
synthesizer = Synthesizer()

app = gr.Interface(
    fn=synthesizer.synthesis,
    inputs=[
        gr.Text(label="Input text"),
        gr.Dropdown(
            [f"{k} ({v})" for k, v in TTS_LANGUAGES.items()],
            label="Language",
            value="eng English",
        ),
        # gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Speed", verbose=False),
        gr.Audio(source="upload", type="filepath", label="Target speaker (voice conversion)"),
    ],
    outputs=[
        gr.Audio(label="Synthesized audio", type="filepath"),
        gr.Audio(label="Voice conversion audio", type="filepath")],
    examples=TTS_EXAMPLES,
    title="Interlink | Text to speech & Voice conversion",
    description="",
    allow_flagging="never"
)

app.launch()
