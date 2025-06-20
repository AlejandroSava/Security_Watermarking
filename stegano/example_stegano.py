from stegano import lsb

# Ocultar mensaje
mensaje = "Este es un mensaje secreto"
print("Este es el mensaje a incrustrar:", mensaje)
lsb.hide("kemonito.jpg", message=mensaje).save("kemonito_mensaje_oculto.png")

# Recuperar mensaje
mensaje = lsb.reveal("kemonito_mensaje_oculto.png")
print("Es es el mensaje que se recupero de la imagen oculta:", mensaje)

