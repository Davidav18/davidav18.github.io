import pandas as pd
import os
import time
import vertexai
from vertexai.generative_models import GenerativeModel


# --- CONFIGURACIÓN ---
ID_PROYECTO = "++++++++++++"
UBICACION = "us-central1"
archivos_a_procesar = ['split_file_1.csv']
TAMANO_LOTE = 2000   # Ajusta según lo que soporte Gemini y tu caso


# --- PROMPT MEJORADO ---
prompt_instruccion = """
# Role:
Actúa como un experto analista de software. A continuación te proporcionaré una lista de nombres de softwares, fuentes, paquetes, etc.
Tu tarea es eliminar duplicados y consolidar esta lista de manera muy específica: si varias entradas se refieren al mismo software base, debes unificarlas en un único nombre representativo o genérico. Elimina números de versión, detalles de arquitectura (x64, x86), fechas y descripciones específicas como 'Runtime', 'Update', 'Agent' o 'Edition'.


# Reglas del Output
Devuelve la respuesta **únicamente** como un archivo CSV **de dos columnas** llamado "Original" y "Consolidated", donde:
- La columna "Original" tiene el valor original de entrada.
- La columna "Consolidated" tiene el nombre reducido o agrupado.
- NO incluyas ningún texto explicativo antes o después, sólo las líneas del archivo CSV.
- Si el nombre comienza con `lib` y es un paquete de Linux (por ejemplo, libnspr4, libnss3, libxcomposite1, libxinerama1, etc.), su valor consolidado o genérico será "System Library".
- Todo lo relacionado a Python (incluyendo librerías y paqueterías) debe consolidarse como "Python Library".
- Todo lo relacionado a fuentes tipográficas o “font” (incluyendo librerías, paquetes y archivos) debe consolidarse como "Text Font".
- Si es un paquete o servicio DNS como bind9, bind9-host, isc-dhcp-server, su valor consolidado debe ser "Network Service".
- Si es un controlador, driver o paquete de impresora, consolida como "Printer Driver".
- Si el software es un browser web, consolida como "Web Browser".
- Si es una utilidad o herramienta de compresión (zip, 7zip, gzip, bzip2, etc.), consolida como "Compression Tool".
- Si es un paquete o utilidad de editor de texto (vim, nano, gedit, notepad++), consolida como "Text Editor".
- Si es un antivirus o software de seguridad, consolida como "Security Software".
- Si es un sistema gestor de bases de datos (MySQL, PostgreSQL, Oracle, etc.), consolida como "Database System".
- Si es una herramienta de administración remota o acceso remoto, consolida como "Remote Access Tool".
- Si es una herramienta de virtualización (VMware, VirtualBox, etc.), consolida como "Virtualization Tool".
- Si es un sistema operativo o componente principal del sistema operativo, consolida como "Operating System Component".
- Si no corresponde a ninguna categoría anterior, sigue el criterio de consolidación por familia de software, tal como en los ejemplos.


# Formato del Output
Ejemplo de formato a devolver:
Original,Consolidated
Microsoft Visual C++ 2015-2022 Redistributable,Microsoft Visual C++
Microsoft Visual C++ 2022 x86 Minimum Runtime,Microsoft Visual C++
Microsoft Visual C++ 2022 x64 Minimum Runtime,Microsoft Visual C++
Windows 7 WDK Header and Libs,WDK Header and Libs
Windows 8 WDK Header and Libs,WDK Header and Libs
Windows 8.1 WDK Header and Libs,WDK Header and Libs
Windows Driver Kit ARM Additions,Windows Driver Kit
Windows Driver Kit ARM Binaries,Windows Driver Kit
Windows Driver Kit ARM Headers and Libs,Windows Driver Kit
Windows Driver Kit Binaries,Windows Driver Kit
libnspr4,System Library
libnss3,System Library
libxcomposite1,System Library
numpy,Python Library
Open Sans,Text Font
bind9,Network Service


Aquí está la lista de datos para procesar:
"""


# --- INICIO DEL SCRIPT ---
try:
    vertexai.init(project=ID_PROYECTO, location=UBICACION)
    model = GenerativeModel("gemini-2.5-pro")
    print("✅ Conectado a la API de Google Gemini (Vertex AI) exitosamente.")

except Exception as e:
    print(f"❌ ERROR: No se pudo inicializar Vertex AI. Detalle: {e}")
    exit()


for archivo in archivos_a_procesar:
    print("-" * 40)
    print(f"Procesando archivo: {archivo}")
    try:
        df = pd.read_csv(archivo)
        primera_columna_series = df.iloc[:, 0].dropna().astype(str).reset_index(drop=True)
        total = len(primera_columna_series)
        resultados_lotes = []
        lotes_fallidos = []


        for i in range(0, total, TAMANO_LOTE):
            lote = primera_columna_series[i:i+TAMANO_LOTE]
            contenido_texto_a_enviar = '\n'.join(lote)
            prompt_completo = prompt_instruccion + "\n" + contenido_texto_a_enviar
            num_lote = i // TAMANO_LOTE + 1
            print(f"   -> Enviando a Gemini... Lote {num_lote} ({i+1} a {min(i+TAMANO_LOTE, total)})")
            start_time = time.time()
            try:
                response = model.generate_content(prompt_completo)
                if hasattr(response, "text") and response.text:
                    respuesta_ia = response.text.strip()
                else:
                    print("   ❌ La respuesta de Gemini no tiene campo `.text` o está vacía.")
                    if hasattr(response, "candidates"):
                        print("   -> Respuesta en 'candidates':", response.candidates)
                    raise Exception("Respuesta de Gemini inesperada, revisa los logs anteriores.")


                from io import StringIO
                delimiter = ',' if respuesta_ia.splitlines()[0].count(',') >= 1 else ';'
                csv_buffer = StringIO(respuesta_ia)
                try:
                    df_respuesta = pd.read_csv(csv_buffer, delimiter=delimiter)
                    df_respuesta = df_respuesta.drop_duplicates().dropna()
                    df_respuesta.columns = ['Original', 'Consolidated']
                    resultados_lotes.append(df_respuesta)
                    # Guarda cada lote individual
                    nombre_csv_lote = f'lote_{num_lote:03d}_{archivo}'
                    df_respuesta.to_csv(nombre_csv_lote, index=False, encoding='utf-8')
                    print(f"   -> Lote {num_lote} procesado y guardado como '{nombre_csv_lote}'. [{time.time() - start_time:.1f} seg]")
                except Exception as e_csv:
                    print(f"   ❌ Error procesando el CSV del lote {num_lote}: {e_csv}")
                    print("   -> Respuesta de Gemini fue:")
                    print(respuesta_ia)
                    lotes_fallidos.append(num_lote)
                    continue  # Sigue con el siguiente lote
            except Exception as e_lote:
                print(f"   ❌ Error al procesar el lote {num_lote}: {e_lote}")
                lotes_fallidos.append(num_lote)
                continue


        # Une todos los lotes
        if resultados_lotes:
            df_final = pd.concat(resultados_lotes, ignore_index=True)
            df_final = df_final.drop_duplicates().dropna()
            nombre_archivo_salida = f'unique_gemini_{archivo}'
            df_final.to_csv(nombre_archivo_salida, index=False, encoding='utf-8')
            print(f"   -> Archivo final consolidado guardado en '{nombre_archivo_salida}'")
        else:
            print("   ⚠️  No se obtuvo ningún resultado válido de los lotes.")


        # Informa de lotes fallidos
        if lotes_fallidos:
            print(f"\n⚠️  Lotes fallidos o con errores: {lotes_fallidos}")


    except Exception as e:
        print(f"   ❌ Error al procesar {archivo}: {e}")
        if "exceeds the maximum token limit" in str(e).lower():
            print("   ⚠️  El contenido enviado probablemente supera el límite de tokens del modelo Gemini.")
            print("       Considera dividir el archivo o reducir el número de líneas.")


print("\n🎉 Proceso completado para archivos.")




