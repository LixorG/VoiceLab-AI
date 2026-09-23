# VoiceLab AI

Laboratorio local de clonación y generación de voz. Subes (o grabas) unos segundos de una voz, escribes un texto y
la aplicación lo lee con esa voz. Todo ocurre en tu ordenador: los audios, los modelos y la base de datos no salen
de aquí, y funciona sin conexión una vez descargados los pesos.

La interfaz está en español; el texto que generes puede estar en cualquier idioma que admita el motor que uses.

**Versión 1.1.0.**

---

## Índice

1. [Qué hace](#qué-hace)
2. [Qué necesitas](#qué-necesitas)
3. [Instalación](#instalación)
4. [Primeros pasos (5 minutos)](#primeros-pasos-5-minutos)
5. [Generar desde la terminal](#generar-desde-la-terminal)
6. [Las secciones de la aplicación](#las-secciones-de-la-aplicación)
7. [Trucos para que suene mejor](#trucos-para-que-suene-mejor)
8. [Marcas que entiende el texto](#marcas-que-entiende-el-texto)
9. [Motores de voz](#motores-de-voz)
10. [Dónde se guardan tus cosas](#dónde-se-guardan-tus-cosas)
11. [Ajustes opcionales](#ajustes-opcionales)
12. [Problemas frecuentes](#problemas-frecuentes)
13. [Para desarrolladores](#para-desarrolladores)
14. [Licencias y uso responsable](#licencias-y-uso-responsable)

---

## Qué hace

- **Clona una voz** a partir de una grabación corta (bastan 3–12 segundos limpios) y lee con ella el texto que
  escribas.
- **Transcribe tus grabaciones** automáticamente (Whisper `large-v3-turbo`), porque los motores necesitan saber qué
  dice la referencia. Puedes corregir la transcripción a mano.
- **Guarda perfiles de voz**: varias grabaciones por voz, una principal, referencias etiquetadas por emoción y los
  parámetros recomendados para cada motor. Se exportan e importan como un archivo `.voiceprofile`.
- **Entrena una voz** (opcional) con tus propias grabaciones, para que el motor ya no necesite audio de referencia.
- **Controla la interpretación** con marcas dentro del texto: pausas, énfasis, susurro, risas, emociones. También
  entiende las etiquetas de ElevenLabs v3.
- **Prepara el guion** antes de generar: arregla puntuación, párrafos y etiquetas **sin cambiar ni una palabra**, y
  avisa de lo que leerá mal (MAYÚSCULAS, enlaces, símbolos, frases demasiado largas).
- **Elige la mejor toma** automáticamente: genera 2, 3 o 5 lecturas de cada frase y conserva la que mejor lee el
  texto (palabras acertadas y parecido con tu voz).
- **Repite una sola frase** si el resto está bien: el audio se vuelve a montar y las demás frases no cambian.
- **Posprocesa el resultado** (ruido, silencios, velocidad, tono, sonoridad, pico, fundidos) siempre sobre una copia:
  el audio original del modelo nunca se toca y puedes comparar A/B.
- **Compara motores y ajustes** en Experimentos, con reproductores sincronizados, tabla comparativa, valoración
  manual y estimaciones automáticas (palabras falladas y parecido de voz).
- **Organiza guiones largos** en Proyectos (un segmento por frase o párrafo, voz y pausa por segmento, exportación a
  WAV/ZIP y subtítulos SRT/VTT).
- **Encuentra lo que generaste** en la Biblioteca: búsqueda por texto, filtros, favoritos, etiquetas y borrado
  múltiple.

Lo que **no** hace: no envía nada a ningún servicio externo, no dobla vídeos, no separa voces de una mezcla y no
convierte tu voz en tiempo real.

---

## Qué necesitas

| | Mínimo | Recomendado |
|---|---|---|
| Sistema | Windows 10/11 (también hay scripts para Linux/macOS y Docker) | Windows 11 |
| Python | 3.11 | 3.11 |
| Node.js | 20 (solo para compilar la interfaz) | LTS actual |
| FFmpeg | obligatorio | winget: `Gyan.FFmpeg` |
| Tarjeta gráfica | ninguna (CPU, muy lento) | NVIDIA con 8 GB de VRAM o más |
| Disco | ~10 GB con un motor | 30 GB si descargas todos los modelos y entrenas voces |

Sin GPU NVIDIA la aplicación funciona, pero generar una frase puede tardar minutos en vez de segundos.

---

## Instalación

En Windows, desde la carpeta del proyecto:

```bash
install.cmd
```

Eso crea el entorno de Python, instala PyTorch con CUDA y los motores, compila la interfaz y comprueba que todo
está en su sitio. Si te falta Python, Node o FFmpeg, deja que los instale por ti:

```bash
install.cmd -InstallPrereqs
```

Otras variantes útiles:

| Comando | Para qué |
|---|---|
| `install.cmd -Engines f5tts` | Instalar solo F5-TTS y E2-TTS |
| `install.cmd -Engines qwen3tts` | Instalar solo Qwen3-TTS |
| `install.cmd -Cpu` | Sin CUDA (solo para probar; muy lento) |
| `install.cmd -SkipFrontend` | No tocar Node ni recompilar la interfaz |

**Los pesos de los modelos no se descargan en la instalación.** Se bajan cuando tú lo pidas, desde la sección
**Modelos** de la aplicación o con:

```bash
backend\.venv\Scripts\python -m app.cli download f5tts
```

(`download qwen3tts`, `download asr` para el modelo de transcripción; `--variant` elige una variante concreta.)

### Arrancar la aplicación

```bash
VoiceLab.cmd
```

Se abre el navegador en <http://127.0.0.1:8000>. Para cerrar, cierra la ventana negra de la consola.

Si cambias de puerto o no quieres que abra el navegador: `scripts\start.ps1 -Port 8100 -NoBrowser`.

### Linux, macOS y Docker

```bash
bash scripts/setup.sh      # instalación
bash scripts/start.sh      # arranque

docker compose up          # CPU
docker compose -f docker-compose.gpu.yml up   # GPU NVIDIA
```

Los scripts de Linux/macOS y las imágenes de Docker están escritos pero no probados en la máquina de desarrollo.

---

## Primeros pasos (5 minutos)

1. **Descarga un motor.** Entra en **Modelos**, elige uno (Qwen3-TTS es el más rápido aquí) y pulsa **Descargar**.
   La primera vez también conviene descargar el modelo de transcripción desde **Configuración**.
2. **Crea una voz.** Ve a **Voces → Nueva voz**, ponle nombre y añade una grabación:
   - **Subir**: un WAV, MP3, FLAC, M4A, OGG, Opus o AIFF con 10–30 segundos de esa persona hablando sola, sin música ni ruido.
   - **Grabar**: pulsa *Grabar con el micrófono* y lee el texto que te propone la pantalla.
3. **Espera a la transcripción.** Se hace sola; revísala y corrígela si se equivocó en alguna palabra (importa más
   de lo que parece: el motor la usa como guía).
4. **Genera.** Ve a **Generar**, elige la voz, escribe tu texto y pulsa **Generar voz**.
   - **Vista previa** genera solo la primera frase: úsala para probar ajustes sin esperar.
5. **Escucha y ajusta.** Si algo no te convence:
   - *Repetir* vuelve a generarlo todo con otra semilla.
   - *Ver segmentos* → **Repetir frase** rehace solo la frase que falló.
   - *Posprocesar* limpia, iguala el volumen o cambia la velocidad sin tocar el original.
6. **Descarga** en WAV, MP3, OGG o FLAC desde el botón de descarga de cada audio.

La barra lateral tiene dos modos: **Sencillo** (lo imprescindible) y **Avanzado** (todos los parámetros). Empieza en
Sencillo.

---

## Generar desde la terminal

Lo mismo que hace la aplicación, sin abrirla, para automatizar (un script que lee una lista de frases, un flujo de
vídeo, un lote de noche). Lo que generes aparece igual en la Biblioteca.

```bash
backend\.venv\Scripts\python -m app.cli speak "Hola a todos" --voice "Hanna Miller" --out hola.wav
backend\.venv\Scripts\python -m app.cli speak --file guion.txt --engine qwen3tts --takes 3 --format mp3
backend\.venv\Scripts\python -m app.cli voices     # tus voces
backend\.venv\Scripts\python -m app.cli engines    # motores y variantes, y si están listos
```

| Opción | Para qué |
|---|---|
| `--voice`, `--reference` | La voz (por nombre o identificador) y, si quieres, una grabación concreta |
| `--engine`, `--variant` | Motor y variante; si no lo dices, los que tengas guardados para esa voz |
| `--param nombre=valor` | Cualquier parámetro real del motor; se puede repetir (`--param speed=1.1`) |
| `--seed`, `--takes` | Semilla exacta y cuántas tomas por frase (se queda con la mejor) |
| `--emotion`, `--intensity` | Emoción global, si el motor la admite |
| `--format`, `--out` | wav, mp3, ogg o flac, y dónde guardarlo |
| `--json`, `--quiet` | Una línea JSON con el resultado para scripts, o sin mensajes |

Cierra la aplicación mientras lo usas: los dos procesos cargarían el modelo en la misma tarjeta gráfica (la propia
orden te avisa si la detecta abierta).

---

## Las secciones de la aplicación

### Generar
El sitio donde trabajas casi siempre. Eliges voz y motor, escribes el texto y generas.

- **Preparar guion**: deja el texto listo para voz sin cambiar tus palabras y te avisa de lo que leerá mal.
- **Mejor toma automática**: 2, 3 o 5 lecturas por frase; se queda con la mejor. Tarda tantas veces más como tomas
  pidas.
- **Vista previa en directo**: empieza a sonar mientras se genera (con Qwen3-TTS, en 1–1,5 segundos).
- **Lote**: encola muchas generaciones de golpe (un texto por línea, varias variantes y repeticiones).
- **Cola**: qué se está generando, qué espera y cancelar lo que no quieras.
- **Historial**: lo último generado, con *Repetir*, *Copiar parámetros*, posprocesado y la lista de frases.

### Biblioteca
Todo lo que has generado desde *Generar*. Busca por lo que dice el audio, filtra por motor, voz, estado, etiqueta o
fecha, marca favoritos, pon tus propias etiquetas y borra varios audios a la vez.

### Voces
Tus perfiles de voz: referencias (subidas o grabadas), cuál es la principal, referencias etiquetadas por emoción,
transcripciones editables, configuración recomendada por motor, exportar/importar y **Entrenar esta voz**.

El entrenamiento (LoRA sobre Qwen3-TTS) necesita al menos 5 minutos de grabaciones transcritas; lo ideal son 15–30.
Al terminar compara la voz entrenada con la clonación normal usando frases tuyas que no utilizó para entrenar. Ayuda
sobre todo con voces con acento marcado; con voces muy estándar la clonación normal ya llega casi igual de lejos.

### Experimentos
El mismo texto leído por varios motores, variantes o ajustes, para decidir con criterio. Reproductores A/B
sincronizados, tabla comparativa, valoración de 1 a 5 y estimaciones automáticas: **palabras falladas** (se
transcribe el resultado y se compara con tu texto) y **parecido de voz** (modelo opcional de 405 MB que se descarga
desde Configuración). Son estimaciones, y así se etiquetan siempre.

### Proyectos
Para guiones largos: se importan por párrafos o frases, cada segmento tiene su voz, emoción, pausa y semilla, se
genera solo lo que falta o lo que quedó *desactualizado* al editar, y se exporta el audio completo (WAV, MP3, OGG,
FLAC), un ZIP con los segmentos sueltos o los subtítulos en SRT/VTT.

### Modelos
Motores instalados, pesos descargados, espacio que ocupan, licencia de cada uno y sus parámetros reales. También
puedes añadir **checkpoints propios** de F5-TTS/E2-TTS (por ejemplo, un ajuste en español de la comunidad).

### Configuración
Transcripción, diccionario de pronunciación (global o por voz), modelo de parecido de voz, apariencia (claro,
oscuro o automático), memoria de la GPU y limpieza de almacenamiento.

---

## Trucos para que suene mejor

- **La referencia manda.** Un audio limpio de 10–20 segundos, sin música, sin eco y sin otra persona hablando, vale
  más que cualquier ajuste. Para F5/E2 se envían 3–12 segundos; para Qwen3-TTS, hasta 20.
- **Revisa la transcripción de la referencia.** Si dice algo distinto de lo que se oye, la voz clonada se resiente.
- **Selecciona el mejor fragmento.** En la forma de onda puedes arrastrar para quedarte solo con la parte buena de
  una grabación larga.
- **Escribe como se habla.** Puntuación real, frases de menos de 20 palabras y párrafos separados por una línea en
  blanco: el motor respira donde tú lo marcas.
- **Números y abreviaturas**: se escriben como se leen automáticamente («15 €» → «quince euros»). Si algo se lee
  mal, añádelo al diccionario de pronunciación.
- **Semilla**: anótala si un resultado te gusta; con la misma semilla y los mismos ajustes vuelve a salir igual.
- **Vista previa primero, texto largo después.**

---

## Marcas que entiende el texto

Marcas propias de VoiceLab:

| Marca | Efecto |
|---|---|
| `[pausa:500ms]` | Silencio de la duración que indiques |
| `[emoción:feliz]…[/emoción]` | Aplica una emoción a ese tramo |
| `[énfasis]…[/énfasis]` | Marca una palabra o frase |
| `[susurro]…[/susurro]` | Baja a un susurro |
| `[risa]`, `[suspiro]`, `[respira]` | Sonidos sueltos |

También se aceptan las etiquetas de **ElevenLabs v3** (`[pause]`, `[long pause]`, `[laughs]`, `[sighs]`,
`[whispers]`, `[emphasized]`, `[stress on next word]`, `[excited]`, `[sad]`…) y los `<break time="1s"/>` de v2, que
se traducen solas. Las que ningún motor puede aplicar se ignoran con un aviso, no dan error.

**Importante**: si el motor elegido no sabe hacer algo (por ejemplo, cambiar la emoción), el botón aparece
desactivado y te dice por qué. La aplicación nunca finge un control que el modelo no tiene.

---

## Motores de voz

| Motor | Puntos fuertes | A tener en cuenta | Licencia de los pesos |
|---|---|---|---|
| **Qwen3-TTS** | El más rápido aquí (más que tiempo real), lectura frase a frase muy natural, se puede entrenar con tu voz | En clonación no acepta emoción ni velocidad nativas | Apache-2.0 (uso comercial permitido) |
| **F5-TTS** | Muy buena calidad, control de velocidad y de pasos de cálculo | Entrenado en inglés y chino: para español conviene un checkpoint ajustado | CC-BY-NC-4.0 (**no comercial**) |
| **E2-TTS** | Alternativa a F5 con el mismo paquete | Reproducción no oficial; mismas limitaciones de idioma | CC-BY-NC-4.0 (**no comercial**) |

Solo se carga un motor en la memoria de la GPU a la vez; se descarga solo cuando lleva un rato sin usarse.

---

## Dónde se guardan tus cosas

```
data/
  references/original/    tus grabaciones tal como las subiste (nunca se modifican)
  references/processed/   la versión que usa la aplicación (24 kHz, mono)
  references/analysis/    las métricas de calidad de cada grabación
  voices/                 los perfiles de voz y su manifiesto
  generated/              cada generación: audio del modelo, frases sueltas y audio final
  projects/               los proyectos
  training/               archivos de trabajo de los entrenamientos
  cache/                  transcripciones y conversiones (se puede borrar sin perder nada)
  database/               voicelab.db (SQLite con todo lo demás)
  logs/                   registro de la aplicación
models/                   pesos de los motores y de Whisper; models/voices/, las voces entrenadas
```

Para hacer copia de seguridad basta con copiar la carpeta `data/`. Los pesos (`models/`) se pueden volver a
descargar cuando quieras.

Desde **Configuración → Almacenamiento** ves cuánto ocupa cada cosa y puedes limpiar archivos huérfanos con
seguridad: solo borra lo que no está referenciado en la base de datos.

---

## Ajustes opcionales

Se ponen en un archivo `.env` en la raíz del proyecto (el instalador crea uno básico):

| Variable | Por defecto | Para qué |
|---|---|---|
| `PORT` | `8000` | Puerto de la aplicación |
| `DEVICE` | `auto` | `cuda`, `cpu`… si quieres forzarlo |
| `DATA_DIR` / `MODEL_DIR` | `./data`, `./models` | Mover los datos o los pesos a otro disco |
| `MODEL_IDLE_UNLOAD_MINUTES` | `15` | Minutos sin usar antes de liberar la GPU (0 = nunca) |
| `ASR_MODEL` | `large-v3-turbo` | Modelo de transcripción |
| `QWEN_CUDA_GRAPHS` | `true` | Aceleración de Qwen3-TTS; ponlo en `false` si da problemas |
| `MAX_AUDIO_SECONDS` | `600` | Duración máxima de una referencia |

El servidor escucha solo en `127.0.0.1`: no es accesible desde otros equipos de la red.

---

## Problemas frecuentes

**«Falta FFmpeg» o los audios no se procesan.**
Instálalo (`winget install Gyan.FFmpeg`) y **abre una consola nueva**: las ventanas ya abiertas no ven el PATH
nuevo. `VoiceLab.cmd` intenta encontrarlo solo.

**El modelo no está descargado.**
Ve a Modelos y pulsa *Descargar*, o usa `python -m app.cli download <motor>`. Los pesos nunca vienen incluidos.

**Se queda sin memoria de GPU.**
La aplicación libera la memoria y lo reintenta una vez. Si se repite: usa una variante más pequeña (Qwen 0.6B),
cierra otros programas que usen la GPU y baja `MODEL_IDLE_UNLOAD_MINUTES`.

**Va lentísimo.**
Comprueba en Configuración que está usando `cuda` y no `cpu`. Con Qwen3-TTS, que la aceleración esté activa.

**La voz clonada no se parece.**
Casi siempre es la referencia: prueba con otro fragmento más limpio, revisa la transcripción y, si la voz tiene
acento marcado, plantéate entrenarla.

**Se corta a mitad o repite palabras.**
Genera con 3 tomas (mejor toma automática) o repite solo esa frase desde *Ver segmentos*.

**Cambié algo y la aplicación sigue igual.**
Reinicia `VoiceLab.cmd` y recarga el navegador con Ctrl+F5.

**Diagnóstico completo:**

```bash
backend\.venv\Scripts\python -m app.cli doctor
```

---

## Para desarrolladores

```bash
scripts\dev.ps1     # backend en :8000 con recarga + interfaz en :5173
scripts\test.ps1    # ruff, pytest con cobertura, oxlint, tsc, vitest y build
```

- **Backend**: Python 3.11, FastAPI, Pydantic v2, SQLModel sobre SQLite, PyTorch, FFmpeg, faster-whisper. Una cola
  de trabajos asíncrona con un único trabajador de GPU en serie y progreso por SSE.
- **Frontend**: React + TypeScript, Vite, Tailwind, shadcn/ui, WaveSurfer.js, Zustand. Todos los textos visibles
  viven en `frontend/src/i18n/es.ts`.
- **Motores**: cada uno es un plugin que declara sus variantes, sus capacidades y sus parámetros reales; la interfaz
  se dibuja a partir de eso. Para añadir uno, lee `docs/models.md`.
- Documentación técnica en `docs/`: `architecture.md`, `audio-pipeline.md`, `model-analysis.md`, `models.md` y
  `testing.md`. El historial de cambios está en `CHANGELOG.md`.

La API REST está documentada sola: con la aplicación en marcha, <http://127.0.0.1:8000/docs>.

---

## Licencias y uso responsable

El código de VoiceLab AI es tuyo para usarlo; **los pesos de los modelos tienen su propia licencia** y no vienen
incluidos. F5-TTS y E2-TTS son **CC-BY-NC-4.0: no se pueden usar con fines comerciales**. Qwen3-TTS es Apache-2.0.
La aplicación muestra la licencia de cada motor antes de descargarlo.

Clona solo voces sobre las que tengas derecho: la tuya o la de alguien que te haya dado permiso explícito. Suplantar
a una persona puede ser ilegal donde vives, además de injusto para ella. Si publicas audio generado, dilo.
