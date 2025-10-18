import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

def cargar_imagenes(ids):
    imagenes = []
    for id in ids:
        path = f"formulario_{id}.png"
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        imagenes.append(img)
    
    return imagenes


def agrupar_lineas(posiciones, distancia_minima=3):

    agrupadas = []
    grupo_actual = [posiciones[0]]

    for i in range(1, len(posiciones)):
        #Si la posición actual esta cerca de la anterior, se agrega al grupo 
        if posiciones[i] - posiciones[i - 1] <= distancia_minima:
            grupo_actual.append(posiciones[i])
        #Sino se cierra el grupo y se calcula el promedio, y se pasa al siguiente grupo
        else:
            agrupadas.append(int(np.mean(grupo_actual)))
            grupo_actual = [posiciones[i]]

    agrupadas.append(int(np.mean(grupo_actual)))
    return agrupadas



def valida_nombre(cant_caracteres, cant_palabras):      #Nombre y apellido: Debe contener un mínimo de dos palabras y no más de 25 caracteres en total.
    return (cant_caracteres>0 and cant_caracteres<=25 and cant_palabras>1)

def valida_edad(cant_caracteres, cant_palabras):        #Edad: Debe contener 2 o 3 caracteres consecutivos (no deben existir espacios entre ellos).
    return (cant_caracteres in [2,3] and cant_palabras == 1)

def valida_mail(cant_caracteres, cant_palabras):        #Mail: Debe contener una palabra y no más de 25 caracteres.
    return (cant_caracteres>=1 and cant_caracteres<=25  and cant_palabras == 1)

def valida_legajo(cant_caracteres, cant_palabras):      #Legajo: Debe contener sólo 8 caracteres en total, formando una única palabra.
    return (cant_caracteres==8 and cant_palabras == 1)

def valida_pregunta(cant_caracteres, cant_palabras):   
    #Preguntas 1, 2 y 3: En cada pregunta debe haber una única celda marcada (referenciadas a las columnas de Si y No), con un único caracter. Es decir, en
    #cada pregunta, ambas celdas no pueden estar marcadas ni ambas celdas pueden
    #quedar vacías.
    return (cant_caracteres==1 and cant_palabras == 1)

def valida_comentario(cant_caracteres, cant_palabras):  #Comentarios: Debe contener al menos una palabra y no más de 25 caracteres.
    return (cant_caracteres>=1 and cant_caracteres<=25 and cant_palabras >= 1)



def cuenta_elementos(campo):
    
    # Umbralamos para que nos quede imagen binaria y bien marcadas las letras
    imagen_umbralada = campo < 190 
    # Casteo a uint8 para poder usar connectedComponentsWithStats()
    imagen_umbralada = imagen_umbralada.astype(np.uint8)
    # Detectar componentes conectadas
    componentes = cv2.connectedComponentsWithStats(imagen_umbralada, 8, cv2.CV_32S) 
    estadisticas = componentes[2]

    #Cestadisticas es una matriz con una fila por cada componente conectada detectada en la imagen.
    #Formato de fila: x,y,width,height,area
    #Los elementos se detectan en cualquier orden, por ende deberias ordenarlos bajo algun criterios para validarlos en su correcto orden
    #Ordenamos las filas en orden ascendente de acuerdo al elemento 0 (coordenadas eje x) de cada subarray
    indices_ordenados = np.argsort(estadisticas[:, 0])
    estadisticas = estadisticas[indices_ordenados]

    #El primer elemento siempre es el fondo, lo descartamos
    estadisticas = estadisticas[1:]

    #La cantidad de elementos detectados representan la cantidad de caracteres
    cant_caracteres = len(estadisticas)

    if cant_caracteres != 0:

        #Si hay mas de un caracter, podemos interpretar que tenemos como minimo una palabra
        cant_palabras = 1

        for i in range(len(estadisticas)-1):
            
            #Calculamos distancia entre componentes
            #Componente i: a su coordenada x le sumamos el ancho. Obtenemos lo que ocupa
            #Componente i+1: a su coordenada x le restamos lo que ocupa la componente i
            #Si la distancia supera un umbral, lo consideramos separacion de palabra y no de letras. El umbral se obtuvo de manera experimental, probando valores.
            if ( estadisticas[i+1][0] - (estadisticas[i][0] + estadisticas[i][2]) ) > 8:
                cant_palabras+=1
    
    # Si no tengo ningun caracter, no tengo ninguna palabra
    else: cant_palabras = 0

    return cant_caracteres, cant_palabras    




def validar_imagen(imagen):
    
    #imagen = imagenes[2]
    # Detectar lo que no es blanco
    no_blanco = imagen < 190
    no_blanco_uint8 = no_blanco.astype(np.uint8)
    #plt.imshow(no_blanco_uint8, cmap='gray')
    #plt.show()

    # Suma por filas y columnas
    enc_cols = np.sum(no_blanco_uint8, axis=0)
    enc_rows = np.sum(no_blanco_uint8, axis=1)

    # Umbrales. Lo obtuvimos de manera experimental, mirando valores de la matriz.
    umbral_vertical = 180
    umbral_horizontal = 913

    # Detección de líneas (sin agrupar)
    lineas_verticales_raw = np.where(enc_cols > umbral_vertical)[0]
    lineas_horizontales_raw = np.where(enc_rows > umbral_horizontal)[0]

    # Agrupar líneas cercanas
    lineas_verticales = agrupar_lineas(lineas_verticales_raw, distancia_minima = 2)
    lineas_horizontales = agrupar_lineas(lineas_horizontales_raw, distancia_minima = 1)

    campos = [] 

    validadores = {
        1: ("nombre_y_apellido", valida_nombre),
        2: ("edad", valida_edad),
        3: ("mail", valida_mail),
        4: ("legajo", valida_legajo),
        6: ("pregunta_1", valida_pregunta),
        7: ("pregunta_2", valida_pregunta),
        8: ("pregunta_3", valida_pregunta),
        9: ("comentarios", valida_comentario),
    }

    
    validaciones = {}

    for h in range(len(lineas_horizontales) - 1):
        cant_caracteres = 0
        cant_palabras = 0
        
        if h in [0, 5]: #Renglones que no tengo que validar
            continue
        
        #Campos que su validacion dependen de una sola imagen
        if (h>=1 and h<=4) or (h==9):     
            #h=1      
            #Uso limites verticales desde el segundo hasta el final
            x_ini = lineas_verticales[1]
            x_fin = lineas_verticales[-1]
            #Uso limites verticales dependiendo el campo que voy a validar
            y_ini = lineas_horizontales[h]
            y_fin = lineas_horizontales[h+1]
            #Recorto el campo en cuestion
            campo = imagen[y_ini:y_fin, x_ini:x_fin]
            campo = campo[5:-5,5:-5] #Elimino margenes para evitar que los contornos confundan mi validacion
            #plt.imshow(campo, cmap='gray')
            #plt.show()
            cant_caracteres, cant_palabras = cuenta_elementos(campo)

        #Campos de preguntas: para validar necesito chequear dos imagenes
        elif (h>=6 and h<=8):
            cant_caracteres_total_pregunta = 0
            cant_palabras_total_pregunta = 0
            
            for v in range(1,len(lineas_verticales) - 1):
                #h=6
                #v=1
                x_ini = lineas_verticales[v]
                x_fin = lineas_verticales[v+1]
                y_ini = lineas_horizontales[h]
                y_fin = lineas_horizontales[h+1]
                campo = imagen[y_ini:y_fin, x_ini:x_fin]
                campo = campo[5:-5,5:-5] #Elimino margenes para evitar que los contornos confundan mi validacion
                #plt.imshow(campo, cmap='gray')
                #plt.show()
                #print("v: ",v)
                cant_caracteres, cant_palabras = cuenta_elementos(campo)
                #print("Caracteres del campo: ",cant_caracteres," Palabras del campo: ", cant_palabras)
                cant_caracteres_total_pregunta += cant_caracteres
                cant_palabras_total_pregunta += cant_palabras
                #print("Caracteres TOTALES: ",cant_caracteres_total_pregunta," Palabras TOTALES: ", cant_palabras_total_pregunta)
            
            #La validacion depende de los totales, por ende asigno los totales como los valores a validar
            cant_caracteres = cant_caracteres_total_pregunta
            cant_palabras = cant_palabras_total_pregunta
            
            #Los elementos se contaron en cada respuesta, pero voy a guardar el campo completo (compuesto por las dos respuestas)
            x_ini = lineas_verticales[1]
            x_fin = lineas_verticales[-1]
            y_ini = lineas_horizontales[h]
            y_fin = lineas_horizontales[h+1]
            campo = imagen[y_ini:y_fin, x_ini:x_fin]

        # Guardo validacion
        if h in validadores:
            clave, funcion = validadores[h]
            validaciones[clave] = ("OK" if funcion(cant_caracteres, cant_palabras) else "MAL", campo)

    return validaciones



def estado_validacion(validacion):
    #validacion = validar_imagen(img)
    for campo in validacion:
        estado, crop = validacion[campo]
        if estado == "MAL":
            return "MAL"
    return "OK"




def generar_csv(ids, validaciones, ruta_salida="detalle_validacion.csv"):
    
    # Encabezados en el orden requerido
    campos = ["ID", "nombre_y_apellido", "edad", "mail", "legajo", "pregunta_1", "pregunta_2", "pregunta_3", "comentarios"]

    with open(ruta_salida, mode="w", newline="", encoding="utf-8") as archivo_csv:
        writer = csv.writer(archivo_csv)
        writer.writerow(campos)  # Agregamos encabezado al CSV?

        for id, validacion in zip(ids,validaciones):

            fila = [id]                                             #Creamos un array que representa nuestra fila del archivo. Le agregamos el primer elemento que es el id
            for campo in campos[1:]:                                #Para cada campo (omitimos el ID) de mi validacion
                estado = validacion.get(campo, ("MAL", None))[0]    #Buscamos el estado en el diccionario de validaciones. La clave es el campo.
                fila.append(estado)                                 #Lo agregamos a nuestra fila

            writer.writerow(fila)                                   #Escribimos la fila en el archivo

    print("Archivo CSV generado")



def generar_validaciones(ids, imagenes):
    validaciones=[]
    for id,img in zip(ids,imagenes):
        print("Imagen ID:", id)
        validacion = validar_imagen(img)
        validaciones.append(validacion)
        estado_formulario = estado_validacion(validacion)
        print("Estado:", estado_formulario)
        print("\nDetalle:")
        for campo in validacion:
            estado, crop = validacion[campo]
            print(campo, estado)
        print("\n")

    return validaciones


def generar_imagen_validaciones(validaciones):

    margen = 20
    tamaño_fuente = 0.6     #Para los estados 
    grosor_texto = 2        #Para los estados 

    resultados = []
    for validacion in validaciones:
        # Estado general del formulario
        estado = estado_validacion(validacion)
        
        # Recorte del nombre
        crop_nombre = validacion["nombre_y_apellido"][1]

        # Asegurarse de que sea RGB
        if len(crop_nombre.shape) == 2:
            crop_nombre = cv2.cvtColor(crop_nombre, cv2.COLOR_GRAY2BGR)

        resultados.append((estado, crop_nombre))

    # Calcular tamaño final de la imagen
    ancho_max = max(crop.shape[1] for _, crop in resultados)
    alto_total = sum(crop.shape[0] + margen for _, crop in resultados)

    # Crear imagen final con fondo blanco
    resumen = np.full((alto_total + margen, ancho_max + 2*margen, 3), 255, dtype=np.uint8)

    y = margen
    for estado, crop in resultados:
        alto, ancho, _ = crop.shape

        # Pegar recorte directamente en la imagen final
        resumen[y:y+alto, margen:margen+ancho] = crop

        # Dibujar borde para que quede mas lindo
        cv2.rectangle(resumen, (margen, y), (margen+ancho-1, y+alto-1), (0,0,0), 1)

        #Color del texto dependiendo del estado
        color_texto = (0, 255, 0) if estado == "OK" else (0, 0, 255)
        cv2.putText(resumen, estado, (margen + ancho - 50, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX, tamaño_fuente, color_texto, grosor_texto)

        y += alto + margen

    cv2.imwrite("resumen_validacion.png", resumen)
    print("Imagen generada")
    #cv2.imshow(resumen)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



#Genero array de ids que deseo cargar
ids = ["01","02","03","04","05"]

#Genero array de imagenes
imagenes = cargar_imagenes(ids)

#Visualizo validaciones en consola
validaciones = generar_validaciones(ids, imagenes)

#Genero imagen
generar_imagen_validaciones(validaciones)

#Generar CSV
generar_csv(ids,validaciones)



