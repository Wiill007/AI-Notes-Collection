# Conceptos y notas capítulo 12
- Input embedding: Al introducir nuestros inputs a la red neuronal necesitamos convertir cada extracto de texto (token) en un input embedding (token id)

## RNN
### Explicación
> Una red neuronal recursiva simple y básica se reduce a algo tan simple como un bloque que utiliza la primera palabra como entrada que pasa por una primera capa de neuronas cuya salida sumada al input embedding de la siguiente palabra pasa de nuevo por otra capa de este mismo tipo. Haríamos este proceso por el número de veces que queramos

Este proceso se repite hasta la última capa en la cual tendríamos las activaciones.

### Mantener el estado anterior
> Se puede mantener la salida como *hidden state*. El problema es que en backprop tendríamos un consumo muy elevado de memoria. Dado el caso de los 10.000 números tendríamos que hacer derivada 10.000 veces para obtener nuestros gradientes. Para evitar mantener el historial de gradientes en Pytorch utilizamos el método *detach*

### Backpropagation Through Time (BTT)
> En lugar de hacer backprop por todo el historial se hace únicamente por las muestras vistas como es el caso de la RNN que se describe anteriormente (con hidden state)

## Exploding/disappearing activations
Es el mismo concepto que Andrej Karpathy explica en su vídeo [Building makemore Part 3](!https://youtu.be/P6sfmUTpUmc?feature=shared).

Dada la repetida multiplicación de matrices, en un entrenamiento con una red neuronal multicapa, los gradientes pueden tender a los extremos +inf, -inf o 0. Esto es un problema porque hace que el entrenamiento de estas neuronas en particular falle dado que los pesos de las mismas saltarían a 0 o a un valor infinito.

## LSTM
Arquitectura de redes neuronales que mantiene un *cell state* y un *hidden state*. La idea de lo primero es tener una memoria a corto largo plazo y el hidden state busca predecir el siguiente token.
Esta arquitectura consta de 4 puertas (de izqda a derecha según esquema del libro pág 391, Figura 12-9):
1. Forget gate: Modifica el estado de la celda para que olvide información irrelevante.
2. Input gate: Actualiza el estado de la celda guardando información nueva
3. "Cell gate": Setea cómo se guarda la información en el estado de celda.
4. Output gate: Determina qué información del cell state se usa para determinar la salida.

## Dropout
Aleatoriamente setear las activaciones de algunas neuronas a cero. Es una técnica de regularización que hace que todas las neuronas tengan que trabajar conjuntamente para la salida. Al haber algunas neuronas con activación 0 a veces, mejor cooperación entre distintas neuronas es necesaria.
> Nota: Es importante tener en cuenta que para ajustar esas activaciones 0 por ejemplo, cuando se usa una función de activación ReLU, es necesario aplicar un reescalado adecuado. Por ejemplo, dada una probabilidad de activación 0 *p*, el reescalado sería dividir entre *1-p*

## Activation Regularization AND Temporal Activation Regularization
Métodos de regularización similares a *weight decay*

### Activation Regularization
Busca hacer **un cambio en las activaciones finales de la red neuronal haciendo que sean lo más pequeñas posible**

```python
loss += alpha * activations.pow(2).mean()
```

### Temporal Activation Regularization
Hace una penalización al loss para hacer que la diferencia entre dos activaciones consecutivas sean lo más pequeñas posible.

```python
loss += beta * (activations[:,1:] - activations[:,:-1]).pow(2).mean()
```