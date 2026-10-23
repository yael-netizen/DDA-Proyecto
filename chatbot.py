# Chatbot de soporte técnico (mesa de ayuda de TI) basado en reglas
# Python 3.10+ y NLTK

import random
import unicodedata

from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

stemmer = SnowballStemmer("spanish")
tokenizador = RegexpTokenizer(r"[a-zñ]+")


def normalizar(texto: str) -> str:
    """Minúsculas y sin acentos."""
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


STOP = {
    "a", "al", "algo", "con", "como", "cual", "cuando", "de", "del", "el", "ella",
    "en", "es", "esta", "estan", "este", "esto", "la", "las", "le", "lo", "los",
    "me", "mi", "mis", "muy", "no", "o", "para", "pero", "por", "porque", "que",
    "se", "si", "sin", "su", "sus", "te", "tengo", "tiene", "un", "una", "uno",
    "y", "ya", "yo", "quiero", "necesito", "puedo", "esta", "estoy", "son", "hay",
}


def procesar(texto: str) -> set[str]:
    """Tokeniza, quita stopwords y devuelve las raíces (stems)."""
    tokens = tokenizador.tokenize(normalizar(texto))
    return {stemmer.stem(t) for t in tokens if t not in STOP}


# Base de conocimiento: intención -> palabras clave y respuestas
INTENCIONES = {
    "saludo": {
        "palabras": ["hola", "buenas", "buenos", "saludos", "hey"],
        "respuestas": [
            "¡Hola! Soy el asistente de soporte de TI. ¿En qué te puedo ayudar?",
            "¡Buen día! Cuéntame tu problema técnico y te ayudo.",
        ],
    },
    "ayuda": {
        "palabras": ["ayuda", "opciones", "menu", "puedes", "hacer"],
        "respuestas": [
            "Puedo ayudarte con: contraseñas, VPN, internet/Wi-Fi, impresoras, "
            "instalación de software, correo, equipo lento, horario de soporte "
            "o crear un ticket."
        ],
    },
    "contrasena": {
        "palabras": ["contrasena", "password", "clave", "olvide", "bloqueada",
                     "bloqueado", "restablecer", "resetear", "desbloquear"],
        "respuestas": [
            "Para restablecer tu contraseña entra al portal de autoservicio, "
            "elige 'Olvidé mi contraseña' y sigue los pasos de verificación. "
            "Si tu cuenta sigue bloqueada, escribe 'ticket'."
        ],
    },
    "vpn": {
        "palabras": ["vpn", "remoto", "remota", "teletrabajo", "tunel"],
        "respuestas": [
            "Para la VPN: 1) revisa que tengas internet, 2) cierra y abre el "
            "cliente VPN, 3) verifica tu usuario y contraseña. Si el error "
            "continúa, escribe 'ticket'."
        ],
    },
    "wifi": {
        "palabras": ["wifi", "internet", "red", "inalambrico", "conexion",
                     "senal", "conectar"],
        "respuestas": [
            "Para problemas de red: apaga y enciende el Wi-Fi de tu equipo, "
            "olvida la red y vuelve a conectarte. Si otros equipos tampoco "
            "tienen internet, avísame con un 'ticket'."
        ],
    },
    "impresora": {
        "palabras": ["impresora", "imprimir", "impresion", "toner", "papel",
                     "escaner"],
        "respuestas": [
            "Revisa que la impresora esté encendida y con papel, y que sea la "
            "predeterminada en tu equipo. Prueba cancelar la cola de "
            "impresión y volver a imprimir."
        ],
    },
    "software": {
        "palabras": ["instalar", "instalacion", "programa", "software",
                     "licencia", "aplicacion"],
        "respuestas": [
            "La instalación de software se solicita por ticket e indicando el "
            "nombre del programa y el motivo. Solo se instalan programas "
            "autorizados por TI."
        ],
    },
    "correo": {
        "palabras": ["correo", "outlook", "email", "mail", "buzon", "adjunto"],
        "respuestas": [
            "Si tienes problemas con el correo, cierra y vuelve a abrir "
            "Outlook y revisa que tu buzón no esté lleno. Si no se "
            "soluciona, escribe 'ticket'."
        ],
    },
    "equipo_lento": {
        "palabras": ["lento", "lenta", "tarda", "congela", "trabado",
                     "trabada", "computadora", "laptop", "equipo"],
        "respuestas": [
            "Si tu equipo va lento: reinícialo, cierra programas que no uses "
            "y libera espacio en disco. Si sigue igual, genera un ticket."
        ],
    },
    "horario": {
        "palabras": ["horario", "hora", "atienden", "abierto", "disponible"],
        "respuestas": [
            "El soporte de TI atiende de lunes a viernes de 8:00 a 18:00 h."
        ],
    },
    "ticket": {
        "palabras": ["ticket", "reporte", "reportar", "incidente", "persona",
                     "agente"],
        "respuestas": ["Claro, describe brevemente el problema para generar tu ticket."],
    },
    "agradecimiento": {
        "palabras": ["gracias", "agradezco", "excelente", "perfecto"],
        "respuestas": ["¡Con gusto! ¿Necesitas algo más?"],
    },
    "despedida": {
        "palabras": ["adios", "bye", "chao", "hasta", "salir", "nos", "vemos"],
        "respuestas": ["¡Hasta luego! Que tengas un excelente día."],
    },
}

SIN_ENTENDER = (
    "No logré entender tu consulta. Puedes escribir 'ayuda' para ver "
    "los temas que atiendo."
)

# Se guardan las raíces de las palabras clave una sola vez
RAICES = {
    nombre: {stemmer.stem(normalizar(p)) for p in datos["palabras"]}
    for nombre, datos in INTENCIONES.items()
}


def identificar_intencion(texto: str) -> str | None:
    """Regresa la intención con más coincidencias, o None si no hay."""
    raices_usuario = procesar(texto)
    mejor, mejor_puntaje = None, 0
    for nombre, raices in RAICES.items():
        puntaje = len(raices_usuario & raices)
        if puntaje > mejor_puntaje:
            mejor, mejor_puntaje = nombre, puntaje
    return mejor


def responder(texto: str, estado: dict) -> str:
    """Genera la respuesta; 'estado' guarda si se espera la descripción de un ticket."""
    if estado.get("esperando_ticket"):
        estado["esperando_ticket"] = False
        folio = f"TKT-{random.randint(1000, 9999)}"
        return (f"Listo, generé tu ticket {folio} con la descripción: "
                f"'{texto.strip()}'. TI te contactará pronto.")

    intencion = identificar_intencion(texto)
    if intencion is None:
        return SIN_ENTENDER
    if intencion == "ticket":
        estado["esperando_ticket"] = True
    estado["ultima_intencion"] = intencion
    return random.choice(INTENCIONES[intencion]["respuestas"])


def iniciar_chat():
    estado = {}
    print("Asistente de Soporte TI (escribe 'adiós' para salir)\n")
    while True:
        texto = input("Usuario: ")
        if not texto.strip():
            continue
        respuesta = responder(texto, estado)
        print(f"Chatbot: {respuesta}\n")
        if estado.get("ultima_intencion") == "despedida":
            break


if __name__ == "__main__":
    iniciar_chat()
