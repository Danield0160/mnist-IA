
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io, re, base64
import tensorflow as tf


app = Flask(__name__)


# Carga el modelo
modelo = tf.keras.models.load_model('numeros.h5')


@app.route('/')
def index():
    return render_template('mnist_py.html')  #este fichero debe estar en la carpeta /templates


@app.route('/predecir', methods=['POST'])
def predecir():
    # Obtener la imagen en formato DataURL desde la petición
    data_url = request.form['imagen_data']

    # Decodificar el DataURL a bytes. La imagen se envia cómo: data:image/png;base64,iVBORw0K…(resto imagen)
    image_data = base64.b64decode(data_url.split(",")[1])

    # Convertir los bytes a una imagen y luego convertirla a escala de grises
    imagen = Image.open(io.BytesIO(image_data)).convert("LA") # Convertir a "Luminance Alpha"


    # Cambiar tamaño a 28x28 (enecesario si el canvas que mandamos es diferente a 28x28 pixels)
    #imagen = imagen.resize((28, 28))


    # Convertir la imagen a un array
    imagen_array = np.array(imagen)


    # Usa solo el canal alfa y normaliza
    alpha_channel = imagen_array[:, :, 1] / 255.0


    # Cambiar la forma para que coincida con el input del modelo y expandir dimensiones
    alpha_channel = alpha_channel.reshape((-1, 784))


    # Hacer una predicción
    predicciones = modelo.predict(alpha_channel)
    numero_predicho = np.argmax(predicciones[0])

    # Devolver el número predicho como respuesta
    return jsonify(prediccion=int(numero_predicho))


if __name__ == '__main__':
    app.run(debug=True)

