import tkinter as tk
from tkinter import filedialog, simpledialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import sys
import os

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

class DenseWithQuantization(tf.keras.layers.Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantization_config = quantization_config

# Cargar modelo VGG16 para extracción de características
try:
    base_model = VGG16(weights='imagenet', include_top=True)
    feature_extractor = Model(inputs=base_model.input,
                              outputs=base_model.get_layer('fc2').output)
    print("Modelo VGG16 cargado correctamente para extracción de características")
except Exception as e:
    feature_extractor = None
    print(f"Error cargando modelo VGG16: {e}")

# Cargar el modelo H5
try:
    model = tf.keras.models.load_model(resource_path('modelo69.h5'), compile=False)
except Exception as e:
    if 'quantization_config' in str(e):
        try:
            model = tf.keras.models.load_model(
                resource_path('modelo69.h5'),
                custom_objects={'Dense': DenseWithQuantization},
                compile=False
            )
        except Exception as e2:
            model = None
            print("Error cargando modelo con fallback de quantization_config:", e2)
    else:
        model = None
        print("Error cargando modelo:", e)

# Definir SURF
SURF = {
    "O": (0.33, 0.33, 0.34, 0.34),
    "M": (0.00, 0.33, 0.33, 0.34),
    "D": (0.67, 0.33, 0.33, 0.34),
    "V": (0.33, 0.67, 0.34, 0.33),
    "L": (0.33, 0.00, 0.34, 0.33),
}

def add_rounded_corners(img, radius=20):
    """Agrega esquinas redondeadas a una imagen PIL"""
    # Convertir a RGBA si no lo está
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    # Crear máscaras para las esquinas redondeadas
    width, height = img.size
    
    # Crear una imagen con fondo transparente
    mask = Image.new('L', (width, height), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    
    # Dibujar un rectángulo redondeado blanco (opaco)
    draw.rounded_rectangle([(0, 0), (width, height)], radius=radius, fill=255)
    
    # Aplicar la máscara a la imagen
    img.putalpha(mask)
    return img

def draw_tooth(ax, x, y, size=1.0, label=None, marks=None):
    ax.add_patch(Rectangle((x, y), size, size, fill=False, linewidth=1.5, edgecolor='#4a5568'))
    for s, (rx, ry, rw, rh) in SURF.items():
        cx, cy, cw, ch = x + rx*size, y + ry*size, rw*size, rh*size
        facecolor = "#2d3748"
        alpha = 1.0
        if marks and s in marks:
            val = marks[s]
            if isinstance(val, tuple):
                facecolor, alpha = val
            else:
                facecolor = val
        ax.add_patch(Rectangle((cx, cy), cw, ch, facecolor=facecolor, edgecolor="#4a5568", linewidth=0.8, alpha=alpha))
        ax.text(cx+cw/2, cy+ch/2, s, ha="center", va="center", fontsize=6.5, color='#a0aec0', weight='bold')
    if label is not None:
        ax.text(x+size/2, y-0.15*size, str(label), ha="center", va="top", fontsize=8, color='#cbd5e0', weight='bold')

def create_odontogram_figure(marks_by_tooth=None, title="Odontograma (FDI 1994)"):
    UPPER = [18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28]
    LOWER = [48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38]
    fig = Figure(figsize=(13, 4.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, color='#e8eef5', pad=15, weight='bold')
    size = 1.0
    gap = 0.25
    y_upper = 2.0
    for i, t in enumerate(UPPER):
        x = i*(size+gap)
        draw_tooth(ax, x, y_upper, size=size, label=t, marks=(marks_by_tooth or {}).get(t))
    y_lower = 0.0
    for i, t in enumerate(LOWER):
        x = i*(size+gap)
        draw_tooth(ax, x, y_lower, size=size, label=t, marks=(marks_by_tooth or {}).get(t))
    ax.set_xlim(-0.5, 16*(size+gap))
    ax.set_ylim(-0.8, 3.4)
    ax.set_facecolor('#1a202c')
    fig.tight_layout()
    return fig

def get_odontogram_teeth_positions():
    UPPER = [18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28]
    LOWER = [48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38]
    size = 1.0
    gap = 0.25
    y_upper = 2.0
    y_lower = 0.0
    positions = {}
    for i, t in enumerate(UPPER):
        x = i * (size + gap)
        positions[t] = (x, y_upper, size)
    for i, t in enumerate(LOWER):
        x = i * (size + gap)
        positions[t] = (x, y_lower, size)
    return positions

class OdontogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Odontológico - IA Avanzada")
        self.root.geometry("1750x950")
        self.root.configure(bg="#0f1419")

        # Colores profesionales dentales
        self.color_primary = "#1e5a96"      # Azul dental profesional
        self.color_primary_light = "#2d7ab8"
        self.color_accent = "#4CAF50"       # Verde éxito
        self.color_bg = "#0f1419"           # Fondo oscuro profesional
        self.color_bg_secondary = "#1a202c" # Fondo secundario
        self.color_text = "#e8eef5"         # Texto claro
        self.color_text_dim = "#a0aec0"     # Texto tenue

        # Estilo ttk
        self.style = ttk.Style(root)
        try:
            self.style.theme_use('clam')
        except Exception:
            pass
        
        # Configurar tema oscuro profesional
        self.style.configure('TFrame', background=self.color_bg)
        self.style.configure('Card.TFrame', background=self.color_bg_secondary, relief='flat')
        
        self.style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=8, background=self.color_primary)
        self.style.map('TButton', 
                      background=[('active', self.color_primary_light), ('pressed', '#1a4470')],
                      foreground=[('active', self.color_text)])
        
        self.style.configure('Action.TButton', font=('Segoe UI', 10, 'bold'), foreground='white')
        self.style.configure('Danger.TButton', background='#dc3545')
        self.style.map('Danger.TButton', background=[('active', '#c82333')])
        
        self.style.configure('TLabel', background=self.color_bg, font=('Segoe UI', 10), foreground=self.color_text)
        self.style.configure('Title.TLabel', background=self.color_bg, font=('Segoe UI', 22, 'bold'), foreground=self.color_primary)
        self.style.configure('Subtitle.TLabel', background=self.color_bg, font=('Segoe UI', 11), foreground=self.color_text_dim)
        self.style.configure('Info.TLabel', background=self.color_bg, font=('Segoe UI', 9), foreground=self.color_text_dim)
        self.style.configure('Status.TLabel', background=self.color_bg_secondary, font=('Segoe UI', 9), foreground=self.color_accent)
        self.style.configure('TLabelFrame', background=self.color_bg, foreground=self.color_text, font=('Segoe UI', 11, 'bold'))
        self.style.configure('TLabelFrame.Label', background=self.color_bg, foreground=self.color_primary, font=('Segoe UI', 11, 'bold'))

        # Header profesional con logos
        header_frame = tk.Frame(root, bg=self.color_primary)
        header_frame.pack(side='top', fill='x', padx=0, pady=0)

        header_inner = tk.Frame(header_frame, bg=self.color_primary)
        header_inner.pack(side='top', fill='x', padx=20, pady=12)

        # Logo principal a la izquierda (más grande)
        try:
            logo_img = Image.open(resource_path('img/logo.png'))
            logo_img = logo_img.resize((250, 100), Image.Resampling.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(header_inner, image=self.logo_photo, bg=self.color_primary)
            logo_label.pack(side='left', padx=(0, 20))
        except Exception as e:
            print(f"Error cargando logo principal: {e}")

        # Título centrado
        title_frame = tk.Frame(header_inner, bg=self.color_primary)
        title_frame.pack(side='left', expand=True)

        title_label = tk.Label(title_frame, text='🦷  SISTEMA ODONTOLÓGICO - Detector de sarro',
                              font=('Segoe UI', 20, 'bold'),
                              background=self.color_primary, foreground='white')
        title_label.pack(anchor='center')

        subtitle_label = tk.Label(title_frame, text='Análisis de Piezas Dentales en Tiempo Real',
                                 font=('Segoe UI', 10),
                                 background=self.color_primary, foreground='#b8d4f1')
        subtitle_label.pack(anchor='center', pady=(2, 0))

        # Logo IES a la derecha
        try:
            ies_logo_img = Image.open(resource_path('img/logo_IESLomoDeLaHerradura_web.png'))
            ies_logo_img.thumbnail((200, 120), Image.Resampling.LANCZOS)
            self.ies_logo_photo = ImageTk.PhotoImage(ies_logo_img)
            ies_logo_label = tk.Label(header_inner, image=self.ies_logo_photo, bg=self.color_primary)
            ies_logo_label.pack(side='right', padx=(20, 0))
        except Exception as e:
            print(f"Error cargando logo IES: {e}")

        # Panel de control mejorado
        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(side='top', fill='x', padx=15, pady=12)

        btn_frame_row1 = ttk.Frame(ctrl_frame)
        btn_frame_row1.pack(side='top', fill='x', pady=(0, 0))

        self.btn_cargar = ttk.Button(btn_frame_row1, text='📷  Cargar Imágenes', command=self.cargar_imagenes)
        self.btn_cargar.pack(side='left', padx=5)

        self.btn_anterior = ttk.Button(btn_frame_row1, text='◀  Anterior', command=self.imagen_anterior)
        self.btn_anterior.pack(side='left', padx=5)

        self.btn_siguiente = ttk.Button(btn_frame_row1, text='Siguiente  ▶', command=self.imagen_siguiente)
        self.btn_siguiente.pack(side='left', padx=5)

        ttk.Separator(btn_frame_row1, orient='vertical').pack(side='left', padx=10, fill='y')

        self.btn_limpiar = ttk.Button(btn_frame_row1, text='🗑️  Limpiar', command=self.limpiar_imagen_actual)
        self.btn_limpiar.pack(side='left', padx=5)

        ttk.Separator(btn_frame_row1, orient='vertical').pack(side='left', padx=10, fill='y')

        self.btn_odontograma = ttk.Button(btn_frame_row1, text='🦷 Ver Odontograma', command=self.abrir_odontograma)
        self.btn_odontograma.pack(side='left', padx=5)

        ttk.Separator(btn_frame_row1, orient='vertical').pack(side='left', padx=10, fill='y')

        self.label_estado = ttk.Label(btn_frame_row1, text='⊙ Sin imágenes cargadas', style='Status.TLabel')
        self.label_estado.pack(side='left', padx=15)

        # Frame principal con editor de imagen
        main_frame = ttk.Frame(root)
        main_frame.pack(fill='both', expand=True, padx=15, pady=(8,15))

        # Imagen para dibujar
        left_frame = ttk.LabelFrame(main_frame, text='', padding=10)
        left_frame.pack(side='top', fill='both', expand=True, pady=(0, 12))

        self.canvas = tk.Canvas(left_frame, cursor='crosshair', bg='#1a202c', bd=0, highlightthickness=2, highlightbackground=self.color_primary)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda e: self.mostrar_imagen())

        self.canvas.bind("<ButtonPress-1>", self.iniciar_rect)
        self.canvas.bind("<B1-Motion>", self.dibujar_rect)
        self.canvas.bind("<ButtonRelease-1>", self.finalizar_rect)

        # Status bar profesional
        status_frame = tk.Frame(root, bg=self.color_bg_secondary)
        status_frame.pack(side='bottom', fill='x', padx=0, pady=0)
        
        status_inner = tk.Frame(status_frame, bg=self.color_bg_secondary)
        status_inner.pack(side='bottom', fill='x', padx=15, pady=8)
        
        self.status_label = ttk.Label(status_inner, text='ℹ️  Selecciona piezas dentales dibujando rectángulos sobre ellas', style='Info.TLabel')
        self.status_label.pack(side='left')

        # Footer con imágenes institucionales separadas
        footer_frame = tk.Frame(root, bg=self.color_bg)
        footer_frame.pack(side='bottom', fill='x', padx=0, pady=0)
        
        footer_left = tk.Frame(footer_frame, bg=self.color_bg)
        footer_left.pack(side='left', fill='x', expand=True, padx=(20, 0), pady=10)
        
        footer_right = tk.Frame(footer_frame, bg=self.color_bg)
        footer_right.pack(side='right', fill='x', expand=True, padx=(0, 20), pady=10)
        
        # Cargar imagen MEFPD en el pie izquierdo
        self.footer_images = []
        try:
            img_left = Image.open(resource_path('img/MEFPD.gif'))
            img_left.thumbnail((400, 240), Image.Resampling.LANCZOS)
            img_left = add_rounded_corners(img_left, radius=15)
            self.footer_left_photo = ImageTk.PhotoImage(img_left)
            self.footer_images.append(self.footer_left_photo)
            tk.Label(footer_left, image=self.footer_left_photo, bg=self.color_bg).pack(side='left')
        except Exception as e:
            print(f"Error cargando img/MEFPD.gif: {e}")
        
        # Cargar imagen Redes_ensenanzas en el pie derecho
        try:
            img_right = Image.open(resource_path('img/Redes_ensenanzas.jpg'))
            img_right.thumbnail((140, 80), Image.Resampling.LANCZOS)
            img_right = add_rounded_corners(img_right, radius=5)
            self.footer_right_photo = ImageTk.PhotoImage(img_right)
            self.footer_images.append(self.footer_right_photo)
            tk.Label(footer_right, image=self.footer_right_photo, bg=self.color_bg).pack(side='right')
        except Exception as e:
            print(f"Error cargando img/Redes_ensenanzas.jpg: {e}")

        # Atajos de teclado
        self.root.bind('<Left>', lambda e: self.imagen_anterior())
        self.root.bind('<Right>', lambda e: self.imagen_siguiente())
        self.root.bind('<Delete>', lambda e: self.limpiar_imagen_actual())

        self.imagenes = []
        self.rutas = []
        self.tk_imagenes = []
        self.boxes_por_imagen = []
        self.indice = 0
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.canvas_figure = None
        self.canvas_agg = None

    def actualizar_estado(self):
        if not self.imagenes:
            self.label_estado.config(text='⊙ Sin imágenes cargadas')
        else:
            num_dientes = len(self.boxes_por_imagen[self.indice])
            nombre = os.path.basename(self.rutas[self.indice])
            self.label_estado.config(text=f'📄 {nombre} • {self.indice+1}/{len(self.imagenes)} • 🦷 {num_dientes} pieza(s)')

    def cargar_imagenes(self):
        rutas = filedialog.askopenfilenames(title="Selecciona imágenes (máximo 5)", filetypes=[("Imágenes", "*.jpg *.png *.bmp *.tiff")])
        if not rutas:
            return
        rutas = rutas[:5]
        self.imagenes.clear()
        self.tk_imagenes.clear()
        self.boxes_por_imagen.clear()
        self.rutas = rutas
        self.indice = 0
        self.root.update_idletasks()
        screen_w = self.root.winfo_width() or self.root.winfo_screenwidth()
        screen_h = self.root.winfo_height() or self.root.winfo_screenheight()
        max_w = min(int(screen_w * 0.92), 1600)
        max_h = min(int((screen_h - 260) * 0.80), 1000)
        for ruta in rutas:
            img = cv2.imread(ruta)
            if img is None:
                messagebox.showerror("Error", f"No se pudo cargar {os.path.basename(ruta)}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            escala = min(max_w/w, max_h/h, 1.0)
            if escala < 1.0:
                nuevo_w = int(w * escala)
                nuevo_h = int(h * escala)
                img = cv2.resize(img, (nuevo_w, nuevo_h))
            self.imagenes.append(img)
            self.tk_imagenes.append(ImageTk.PhotoImage(Image.fromarray(img)))
            self.boxes_por_imagen.append([])
        self.mostrar_imagen()
        self.status_label.config(text=f'✓ Cargadas {len(self.imagenes)} imágenes • Dibuja rectángulos sobre las piezas dentales')
        self.actualizar_odontograma_ventana()

    def limpiar_imagen_actual(self):
        if not self.imagenes:
            return
        if messagebox.askyesno("Confirmar", "¿Limpiar todas las piezas de esta imagen?"):
            self.boxes_por_imagen[self.indice].clear()
            self.mostrar_imagen()
            self.actualizar_odontograma_ventana()
            self.status_label.config(text='✓ Imagen limpiada')

    def mostrar_imagen(self):
        self.canvas.delete("all")
        if not self.imagenes:
            return
        
        # Mostrar imagen
        img_tk = self.tk_imagenes[self.indice]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w > 0 and canvas_h > 0:
            self.canvas.create_image(canvas_w//2, canvas_h//2, anchor="center", image=img_tk)
        else:
            self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        
        # Mostrar boxes con colores profesionales
        colores_pieza = {
            0: "#4CAF50",      # Verde - Normal
            1: "#FF5722",      # Rojo - Sarro
            2: "#FFC107",      # Amarillo - Mancha
            3: "#2196F3"       # Azul - Otro
        }
        
        for box in self.boxes_por_imagen[self.indice]:
            x1, y1, x2, y2, clase, num = box
            color = colores_pieza.get(clase, "#4CAF50")
            
            # Dibujar rectángulo
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3)
            
            # Etiqueta con información
            self.canvas.create_rectangle(x1, y1-22, x1+45, y1, fill=color, outline=color)
            self.canvas.create_text(x1+22, y1-11, text=f"#{num}", anchor="center", 
                                   fill="white", font=("Segoe UI", 10, "bold"))
        
        self.actualizar_estado()

    def abrir_odontograma(self):
        if hasattr(self, 'odontograma_window') and self.odontograma_window.winfo_exists():
            self.odontograma_window.lift()
            self.actualizar_odontograma_ventana()
            return
        self.odontograma_window = tk.Toplevel(self.root)
        self.odontograma_window.title("Odontograma - FDI 1994")
        self.odontograma_window.geometry("1400x700")
        self.odontograma_window.configure(bg=self.color_bg)
        
        # Canvas para odontograma
        self.canvas_odontograma_ventana = tk.Canvas(self.odontograma_window, bg='#1a202c', bd=0, highlightthickness=2, highlightbackground=self.color_primary)
        self.canvas_odontograma_ventana.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Frame para controles
        ctrl_frame_ventana = ttk.Frame(self.odontograma_window)
        ctrl_frame_ventana.pack(fill='x', padx=20, pady=(0,20))
        
        ttk.Label(ctrl_frame_ventana, text="Editar pieza:", style='Info.TLabel').pack(side='left')
        
        self.tooth_combo_odontograma = ttk.Combobox(ctrl_frame_ventana, state='readonly', width=5)
        self.tooth_combo_odontograma.pack(side='left', padx=(5,10))
        
        ttk.Button(ctrl_frame_ventana, text='0', command=lambda: self.cambiar_clase_odontograma(0)).pack(side='left', padx=2)
        ttk.Button(ctrl_frame_ventana, text='1', command=lambda: self.cambiar_clase_odontograma(1)).pack(side='left', padx=2)
        ttk.Button(ctrl_frame_ventana, text='2', command=lambda: self.cambiar_clase_odontograma(2)).pack(side='left', padx=2)
        ttk.Button(ctrl_frame_ventana, text='3', command=lambda: self.cambiar_clase_odontograma(3)).pack(side='left', padx=2)
        
        ttk.Button(ctrl_frame_ventana, text='📥 Descargar', command=self.descargar_odontograma).pack(side='right', padx=10)
        
        self.actualizar_odontograma_ventana()
        
        self.odontograma_window.protocol("WM_DELETE_WINDOW", self.cerrar_odontograma)

    def cerrar_odontograma(self):
        self.odontograma_window.destroy()
        delattr(self, 'odontograma_window')

    def cambiar_clase_odontograma(self, nueva_clase):
        num_str = self.tooth_combo_odontograma.get()
        if not num_str:
            return
        num = int(num_str)
        for i, box in enumerate(self.boxes_por_imagen[self.indice]):
            if box[5] == num:
                x1, y1, x2, y2, _, _ = box
                self.boxes_por_imagen[self.indice][i] = (x1, y1, x2, y2, nueva_clase, num)
                break
        self.actualizar_odontograma_ventana()
        self.mostrar_imagen()
        self.status_label.config(text=f'✓ Clase de pieza #{num} cambiada')

    def on_click_odontograma(self, event):
        if event.button != 1 or event.inaxes is None:
            return
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return
        positions = get_odontogram_teeth_positions()
        for num, (x, y, size) in positions.items():
            margin = 0.25
            if x - margin <= xdata <= x + size + margin and y - 0.25 <= ydata <= y + size + margin:
                values = self.tooth_combo_odontograma['values']
                if str(num) in values:
                    self.tooth_combo_odontograma.set(str(num))
                    self.status_label.config(text=f'✓ Pieza #{num} seleccionada para edición')
                return

    def actualizar_odontograma_ventana(self):
        if not hasattr(self, 'odontograma_window') or not self.odontograma_window.winfo_exists():
            return
        marks_by_tooth = {}
        for img_boxes in self.boxes_por_imagen:
            for box in img_boxes:
                x1, y1, x2, y2, clase, num = box
                class_to_marks = {
                    0: {},
                    1: {"O": "#FF5722"},    # Rojo - Sarro
                    2: {"O": "#FFC107"},    # Amarillo - Mancha
                    3: {"O": "#2196F3"}     # Azul - Otro (oclusal)
                }
                marks = class_to_marks.get(clase, {})
                marks_by_tooth[num] = marks
        
        # Guardar el diente seleccionado actualmente
        diente_actual = self.tooth_combo_odontograma.get()
        
        # Actualizar combo
        if self.indice < len(self.boxes_por_imagen):
            nums = [box[5] for box in self.boxes_por_imagen[self.indice]]
        else:
            nums = []
        self.tooth_combo_odontograma['values'] = sorted(set(nums))
        
        # Restaurar el diente seleccionado si sigue existiendo, si no, selecciona el primero
        if diente_actual and diente_actual in self.tooth_combo_odontograma['values']:
            self.tooth_combo_odontograma.set(diente_actual)
        elif nums:
            self.tooth_combo_odontograma.set(nums[0])
        else:
            self.tooth_combo_odontograma.set('')
        
        # Crear figura
        fig = create_odontogram_figure(marks_by_tooth)
        fig.set_facecolor('#1a202c')
        fig.patch.set_alpha(1.0)
        
        # Limpiar canvas anterior
        if hasattr(self, 'canvas_agg_ventana'):
            self.canvas_agg_ventana.get_tk_widget().destroy()
        
        # Crear canvas de matplotlib
        self.canvas_agg_ventana = FigureCanvasTkAgg(fig, master=self.canvas_odontograma_ventana)
        self.canvas_agg_ventana.draw()
        self.canvas_agg_ventana.get_tk_widget().pack(fill='both', expand=True)
        
        # Conectar clic en el odontograma para seleccionar la pieza
        if hasattr(self, 'cid_odontograma_click'):
            try:
                self.canvas_agg_ventana.mpl_disconnect(self.cid_odontograma_click)
            except Exception:
                pass
        self.cid_odontograma_click = self.canvas_agg_ventana.mpl_connect('button_press_event', self.on_click_odontograma)

    def descargar_odontograma(self):
        if not hasattr(self, 'canvas_agg_ventana'):
            return
        fig = self.canvas_agg_ventana.figure
        ruta = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if ruta:
            fig.savefig(ruta, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Descargado", f"Odontograma guardado en {ruta}")

    def iniciar_rect(self, event):
        if not self.imagenes:
            return
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, 
                                                 outline="#2196F3", width=3, dash=(5, 5))

    def dibujar_rect(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def finalizar_rect(self, event):
        if not self.rect or not self.imagenes:
            return
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        # Validar tamaño mínimo
        if x2 - x1 < 15 or y2 - y1 < 15:
            messagebox.showwarning("Área muy pequeña", "El rectángulo debe ser más grande")
            self.rect = None
            return
        
        # Crop
        img = self.imagenes[self.indice]
        h, w = img.shape[:2]
        img_tk = self.tk_imagenes[self.indice]
        img_w, img_h = img_tk.width(), img_tk.height()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        # Imagen centrada en el canvas
        offset_x = max((canvas_w - img_w) // 2, 0)
        offset_y = max((canvas_h - img_h) // 2, 0)

        crop_x1 = int((x1 - offset_x) * (w / img_w))
        crop_y1 = int((y1 - offset_y) * (h / img_h))
        crop_x2 = int((x2 - offset_x) * (w / img_w))
        crop_y2 = int((y2 - offset_y) * (h / img_h))

        crop_x1 = max(0, min(w, crop_x1))
        crop_y1 = max(0, min(h, crop_y1))
        crop_x2 = max(0, min(w, crop_x2))
        crop_y2 = max(0, min(h, crop_y2))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            messagebox.showwarning("Área inválida", "El rectángulo debe estar dentro de la imagen")
            self.rect = None
            return

        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            messagebox.showwarning("Área inválida", "No se pudo recortar la imagen. Dibuja el rectángulo dentro de la imagen.")
            self.rect = None
            return
        
        # Predict
        predicted_class = 0
        if model is not None and feature_extractor is not None:
            try:
                # Preprocesar imagen para VGG16
                crop_vgg = cv2.resize(crop, (224, 224))
                crop_vgg = cv2.cvtColor(crop_vgg, cv2.COLOR_BGR2RGB)
                crop_vgg = preprocess_input(np.expand_dims(crop_vgg, axis=0))
                
                # Extraer características con VGG16
                features = feature_extractor.predict(crop_vgg, verbose=0)
                
                # Usar características como entrada para el modelo entrenado
                prediction = model.predict(features, verbose=0)
                predicted_class = np.argmax(prediction[0])
            except Exception as e:
                print(f"Error en predicción: {e}")
        elif model is not None:
            try:
                # Fallback: usar el método original si VGG16 falla
                crop_resized = cv2.resize(crop, (128, 128))
                crop_norm = np.expand_dims(crop_resized, axis=0) / 255.0
                prediction = model.predict(crop_norm, verbose=0)
                predicted_class = np.argmax(prediction[0])
                print("Usando método de predicción original (sin extracción de características)")
            except Exception as e:
                print(f"Error en predicción fallback: {e}")
        
        # Pedir número del diente con validación mejorada
        num = simpledialog.askinteger("Número FDI", 
                                     "Ingresa el número FDI de la pieza:\n\n11-18 (cuadrante 1)\n21-28 (cuadrante 2)\n31-38 (cuadrante 3)\n41-48 (cuadrante 4)")
        
        if num is not None:
            validos = list(range(11,19)) + list(range(21,29)) + list(range(31,39)) + list(range(41,49))
            if num not in validos:
                messagebox.showerror('Número inválido', 'El número debe estar entre:\n11-18, 21-28, 31-38 o 41-48')
            else:
                existing_nums = [box[5] for box in self.boxes_por_imagen[self.indice]]
                if num in existing_nums:
                    messagebox.showwarning('Pieza duplicada', f'La pieza #{num} ya fue registrada en esta imagen')
                else:
                    self.boxes_por_imagen[self.indice].append((x1, y1, x2, y2, predicted_class, num))
                    self.actualizar_odontograma_ventana()
                    self.status_label.config(text=f'✓ Pieza #{num} registrada correctamente')
        
        self.rect = None
        self.mostrar_imagen()

    def imagen_anterior(self):
        if not self.imagenes:
            return
        self.indice = max(0, self.indice - 1)
        self.mostrar_imagen()
        self.actualizar_odontograma_ventana()

    def imagen_siguiente(self):
        if not self.imagenes:
            return
        self.indice = min(len(self.imagenes) - 1, self.indice + 1)
        self.mostrar_imagen()
        self.actualizar_odontograma_ventana()

root = tk.Tk()
app = OdontogramApp(root)
root.state('zoomed')  # Maximizar la ventana por defecto
root.mainloop()