#
# KP - PT.PINDAD - FTMD ITB
# Author: linkedin.com/in/juan-aaron-norata-47242a222
#
# Hak Cipta © 2025 Juan Aaron Norata - Hak Cipta Dilindungi Undang-Undang
#
# Program dan kode sumber ini dilindungi berdasarkan Konvensi Bern untuk
# Perlindungan Karya Sastra dan Seni, TRIPS Agreement (1994), Undang-Undang Republik Indonesia Nomor 28
# Tahun 2014 tentang Hak Cipta, serta peraturan perundang-undangan hak kekayaan
# intelektual lainnya yang berlaku, baik nasional maupun internasional.
#
# Dilarang keras menyalin, mendistribusikan, memodifikasi, mensublisensikan,
# merekayasa balik (reverse engineering), atau menggunakan perangkat lunak ini,
# baik sebagian maupun seluruhnya, tanpa izin tertulis sebelumnya dari pemilik hak cipta.
#
# Program ini memanfaatkan modul pihak ketiga (misalnya PaddleOCR, YOLO, CustomTkinter,
# dan lainnya) yang dilisensikan secara terpisah di bawah lisensi sumber terbuka
# masing-masing. Hak cipta atas modul-modul tersebut tetap dimiliki oleh pemiliknya.
#
# Perangkat lunak ini disediakan "SEBAGAIMANA ADANYA" tanpa jaminan apapun, baik
# tersurat maupun tersirat, termasuk namun tidak terbatas pada jaminan
# kelayakan untuk diperdagangkan, kesesuaian untuk tujuan tertentu, dan bebas
# dari pelanggaran hak pihak ketiga. Dalam keadaan apapun, pemilik hak cipta
# tidak bertanggung jawab atas klaim, kerugian, atau kewajiban lain yang timbul
# dari atau sehubungan dengan perangkat lunak ini atau penggunaan maupun
# interaksi lainnya dengan perangkat lunak ini.
#
#---------------------------------------------------------------------------------------------------------------------------//
#
# Copyright (c) 2025 Juan Aaron Norata - All Rights Reserved
# 
# This program and its source code are protected under the Berne Convention for
# the Protection of Literary and Artistic Works, TRIPS Agreement (1994), the Indonesian Copyright Law
# (Undang-Undang Nomor 28 Tahun 2014 tentang Hak Cipta), and other applicable
# international and national intellectual property laws.
#
# Unauthorized copying, distribution, modification, sublicensing, reverse engineering,
# or any other use of this software, in whole or in part, is strictly prohibited
# without the prior written consent of the author.
#
# This program makes use of third-party modules (such as PaddleOCR, YOLO,
# CustomTkinter, and others) which are separately licensed under their respective
# open-source licenses. Copyrights for these modules remain with their respective
# owners.
#
# This software is provided "AS IS", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a
# particular purpose, and noninfringement. In no event shall the author be liable
# for any claim, damages, or other liability arising from, out of, or in connection
# with the software or the use or other dealings in the software.

#---------------------------------------------------------------------------------------------------------------------------//

# MANUAL INSTALLATION GUIDE
# Please download cuda 12.9, then enable a venv environment 
# For first install, please run the command: pip install -r requirements.txt
# requirements.txt can be installed at: https://github.com/JuanA26/PINDAD-OSerialSP1/blob/main/requirements.txt

#---------------------------------------------------------------------------------------------------------------------------//

import tkinter
import customtkinter
from customtkinter import CTkImage
import pywinstyles
from PIL import Image
import time
import cv2
import os
import csv
import datetime
import threading
import shutil
import numpy as np
import requests
import zipfile
import io
import tempfile
import json, re
from pathlib import Path

def _daily_csv_path(ts: float) -> str:
    dt = datetime.datetime.fromtimestamp(ts)
    base = Path("logs") / "csv" / f"{dt:%Y}" / f"{dt:%m}"
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"{dt:%Y-%m-%d}.csv")

def _ensure_header(path: str):
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["Timestamp", "No", "Status", "PIN", "Accuracy", "Weight"])

FOLDER_DATA = "DATA"

# --- Inlined former utils.py helpers (was: DATA/py_files/utils.py) ---
def download_with_progress(url, desc="Downloading"):
    import sys, time
    response = requests.get(url, stream=True, timeout=(5, 30))
    total_length = response.headers.get('content-length')

    if total_length is None:  # no content length header
        content = response.content
        print(f"{desc}: Size unknown. Downloaded.")
        return content

    total_length = int(total_length)
    chunk_size = 8192
    downloaded = 0
    chunks = []
    start_time = time.time()
    print(f"{desc}:")
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            chunks.append(chunk)
            downloaded += len(chunk)
            done = int(50 * downloaded / total_length)
            percent = int(100 * downloaded / total_length)
            speed = downloaded / (time.time() - start_time + 1e-5)
            sys.stdout.write(
                f"\r[{'=' * done}{' ' * (50 - done)}] {percent}% "
                f"{downloaded//1024}KB/{total_length//1024}KB "
                f"({speed/1024:.1f} KB/s)"
            )
            sys.stdout.flush()
    sys.stdout.write("\n")
    return b"".join(chunks)

def download_and_merge_data(folder_names, base_url, target_dir="DATA"):
    os.makedirs(target_dir, exist_ok=True)
    for folder in folder_names:
        zip_url = f"{base_url}/{folder}.zip"
        print(f"Preparing to download {folder} from {zip_url}...")
        try:
            content = download_with_progress(zip_url, desc=f"Downloading {folder}")
            if content and len(content) > 0:
                with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
                    with tempfile.TemporaryDirectory() as temp_extract_dir:
                        zip_ref.extractall(temp_extract_dir)
                        for extracted_name in os.listdir(temp_extract_dir):
                            src_path = os.path.join(temp_extract_dir, extracted_name)
                            dest_path = os.path.join(target_dir, folder)
                            if os.path.isdir(src_path):
                                if os.path.exists(dest_path):
                                    shutil.rmtree(dest_path)
                                shutil.move(src_path, dest_path)
                                print(f"Extracted into: {dest_path}")
            else:
                print(f"Failed to download {zip_url}: No content received.")
        except Exception as e:
            print(f"Error downloading {folder}: {e}")

def check_and_restore_data():
    github_zip_base_url = "https://github.com/JuanA26/PINDAD-OSerialSP1/raw/refs/heads/main"
    required_subfolders = ["images", "TEMP", "YOLOonnx", "py_files", "PPOCRonnxrec", "PPOCRonnxdet", "dict", "RFDETRonnx"]

    missing = []
    if not os.path.isdir(FOLDER_DATA):
        print(f"'{FOLDER_DATA}' folder is missing.")
        missing = required_subfolders
    else:
        for sub in required_subfolders:
            sub_path = os.path.join(FOLDER_DATA, sub)
            if not os.path.isdir(sub_path):
                print(f"Missing subfolder: {sub_path}")
                missing.append(sub)

    if missing:
        print(f"Downloading missing folders: {missing}")
        download_and_merge_data(missing, github_zip_base_url, target_dir=FOLDER_DATA)
    else:
        print(f"'{FOLDER_DATA}' and all required subfolders are present.")
# --- end inlined utils ---

if not os.path.isdir(FOLDER_DATA):
    print(f"The folder '{FOLDER_DATA}' does not exist. Attempting to download components...")
    github_zip_base_url = "https://github.com/JuanA26/PINDAD-OSerialSP1/raw/refs/heads/main"
    folders_to_download = ["images", "TEMP", "YOLOonnx", "py_files", "PPOCRonnxrec", "PPOCRonnxdet", "dict", "RFDETRonnx"]
    download_and_merge_data(folders_to_download, github_zip_base_url, target_dir=FOLDER_DATA)
else:
    print(f"The folder '{FOLDER_DATA}' exists.")
    check_and_restore_data()

## Import separate modules + check
if(not os.path.isdir(FOLDER_DATA)):
    print(f"Error: The folder '{FOLDER_DATA}' does not exist. Please ensure the DATA folder is present.")
    exit(1)
else:
    from DATA.py_files import globals
    from DATA.py_files import ocrmodels
    from DATA.py_files import serial_comm
    from DATA.py_files import imagetool
    from DATA.py_files import stats_utils
    from DATA.py_files import doubledata
    from DATA.py_files import ocrerrorutils
    from DATA.py_files import pputil
    from serial import SerialException
    from tkinter import ttk
    from collections import deque

cam = cv2.VideoCapture(0)

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

esp32 = serial_comm.init_serial()

# --- model sessions as globals so App() can use them after preload ---
onnxdet = None
onnxrec = None
model   = None
model2  = None
model3  = None

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        try:
            self.attributes("-alpha", 0.0)
        except Exception:
            pass

        self._closing = False
        self._after_tokens = set()

        self.streaming_active = False
        self.cam_lock = threading.Lock()
        self._frame_q = deque(maxlen=1)

        self.overlay_yolo = True         # set False to disable overlay
        self._boxes_live = []            # cached boxes from last inference
        self._infer_interval = 0.08      # seconds between YOLO passes (throttle)
        self._last_infer_t = 0.0
        self._box_ttl = 0.5
        self._conf_thresh = 0.5

        self.title("OSerial 2025 - SP1 - PINDAD MRI")
        self.geometry("{0}x{1}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))
        self.state('zoomed')

        self.protocol("WM_DELETE_WINDOW", self.handle_close)

        self._last_rgb = None  # last frame rendered to the label

        # root!
        self.main_container = customtkinter.CTkFrame(self, fg_color="#000000")
        self.main_container.pack(fill=tkinter.BOTH, expand=True, padx=0, pady=0)

        pywinstyles.change_header_color(self, color="#000000")  

        # left side panel -> for frame selection
        self.left_side_panel = customtkinter.CTkFrame(self.main_container, width=380, corner_radius=10, fg_color="#0B0C27")
        self.left_side_panel.pack(side=tkinter.LEFT, fill=tkinter.Y, expand=False, padx=5, pady=5)
        self.left_side_panel.pack_propagate(False)
        
        self.left_side_panel.grid_columnconfigure((0, 1), weight=1)
        self.left_side_panel.grid_rowconfigure((0, 1, 2, 3), weight=0)
        self.left_side_panel.grid_rowconfigure((4, 5, 6), weight=0)
        self.left_side_panel.grid_rowconfigure((7, 8), weight=0)
        self.left_side_panel.grid_rowconfigure(9, weight=1)
        self.left_side_panel.grid_propagate(False)

        self.right_side_panel = customtkinter.CTkFrame(self.main_container, corner_radius=10, fg_color="#0B0C27")
        self.right_side_panel.pack(side=tkinter.RIGHT, fill=tkinter.BOTH, expand=True, padx=(0, 5), pady=5)
        self.right_side_panel.grid_columnconfigure(0, weight=0)              # stretch
        self.right_side_panel.grid_columnconfigure(1, weight=1)  # fixed 800 px
        self.right_side_panel.grid_rowconfigure(0, weight=3)      # top can grow more
        self.right_side_panel.grid_rowconfigure(1, weight=0, minsize=250)      # keep bottom tall enough
        self.right_side_panel.grid_rowconfigure(2, weight=0, minsize=28)       # small fixed footer

        self.top_right_panel = customtkinter.CTkFrame(self.right_side_panel, corner_radius=10, fg_color="#000000", height=400)
        self.top_right_panel.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="nsew", columnspan=2)
        self.top_right_panel.grid_columnconfigure((0), weight=1)
        self.top_right_panel.grid_rowconfigure((0), weight=1)

        self.bottom_right_panel1 = customtkinter.CTkFrame(self.right_side_panel, corner_radius=10, fg_color="#232121", width=300, height=250)
        self.bottom_right_panel1.grid(row=1, column=0, padx=(10, 5), pady=(5, 5), sticky="nsew")
        self.bottom_right_panel1.grid_columnconfigure((0), weight=1)
        self.bottom_right_panel1.grid_rowconfigure((0), weight=1)
        self.bottom_right_panel1.grid_propagate(False)

        self.status_label = customtkinter.CTkLabel(self.bottom_right_panel1, text="-", font=customtkinter.CTkFont(family='Bahnschrift SemiLight', size=60, weight="bold"), text_color="white")
        self.status_label.grid(row=0, column=0, padx=40, pady=40, sticky="nsew")

        self.bottom_right_panel2 = customtkinter.CTkFrame(self.right_side_panel, corner_radius=10, fg_color="#232121", height=250)
        self.bottom_right_panel2.grid(row=1, column=1, padx=(5, 10), pady=(5, 5), sticky="nsew")
        self.bottom_right_panel2.grid_columnconfigure((0), weight=1)
        self.bottom_right_panel2.grid_rowconfigure((0), weight=1)
        self.bottom_right_panel2.grid_propagate(False)

        self.bottom_rightIpanel3 = customtkinter.CTkFrame(self.right_side_panel, corner_radius=10, fg_color="#232121")
        self.bottom_rightIpanel3.grid(row=2, column=0, columnspan=2, padx=10, pady=(5, 10), sticky="nsew")
        self.bottom_rightIpanel3.grid_columnconfigure((0), weight=1)

        self.copyright = customtkinter.CTkLabel(self.bottom_rightIpanel3, text="© 2025 Juan Aaron Norata - FTMD ITB", font=customtkinter.CTkFont(family='Bahnschrift SemiLight', size=10, weight="bold"), text_color="white")
        self.copyright.grid(row=0, column=0, padx=(10,10), pady=(5, 5), sticky="nsew")

        self.top_left_panel = customtkinter.CTkFrame(self.left_side_panel, corner_radius=10, fg_color="#232121")
        self.top_left_panel.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 10), sticky="nsew")

        self.mid_left_panel = customtkinter.CTkFrame(self.left_side_panel, corner_radius=60, fg_color="yellow", height=60)
        self.mid_left_panel.grid(row=1, column=0, columnspan=2, padx=30, pady=(10, 0), sticky="nsew")
        self.mid_left_panel.grid_columnconfigure((0), weight=1)
        self.mid_left_panel.grid_rowconfigure((0), weight=1)
        self.mid_left_panel.grid_propagate(False)

        self.mid_left_panel2 = customtkinter.CTkFrame(self.left_side_panel, corner_radius=10, fg_color="#232121", height = 100)
        self.mid_left_panel2.grid(row=3, column=0, columnspan=2, padx=10, pady=(10, 0), sticky="nsew")
        self.mid_left_panel2.grid_columnconfigure((0), weight=1)
        self.mid_left_panel2.grid_rowconfigure((0), weight=1)
        self.mid_left_panel2.grid_propagate(False)

        self.prevocr_label = customtkinter.CTkLabel(self.mid_left_panel2, text=None, corner_radius=10)
        self.prevocr_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.bottom_left_panel = customtkinter.CTkFrame(self.left_side_panel, corner_radius=10, fg_color="#232121", height=298, width=380)
        self.bottom_left_panel.grid(row=10, column=0, columnspan=2, padx=10, pady=(0, 10))
        self.bottom_left_panel.grid_propagate(False)

        img = CTkImage(light_image=Image.open("DATA/images/pindad.png"),
               dark_image=Image.open("DATA/images/pindad.png"),
               size=(88, 59))
        label = customtkinter.CTkLabel(self.top_left_panel, text=None, image=img)
        label.grid(row=1, column=0, padx=(15,0), pady=(0, 0))
        
        self.image_path = "DATA/images/startuplogo.png"
        self.ocrimage = CTkImage(light_image=Image.open(self.image_path),   
                            size=(990, 500))
        self.main_display = customtkinter.CTkLabel(self.top_right_panel, text=None, image=self.ocrimage, corner_radius=10)
        self.main_display.grid(row=0, column=0, padx=10, pady=20, sticky="nsew")

        self.top_right_panel.bind("<Configure>", self._on_top_right_resize)

        self.T_W, self.T_H = 990, 500

        self.ctk_img = CTkImage(light_image=Image.new("RGB", (self.T_W, self.T_H)),
                        size=(self.T_W, self.T_H))
        self.main_display.configure(image=self.ocrimage)

        self.after(15, self._ui_tick)
        
        # self.left_side_panel WIDGET
        self.logo_label = customtkinter.CTkLabel(self.top_left_panel, text="OSerial 2025 - SP1 \n", font=customtkinter.CTkFont(family='Bahnschrift SemiLight', size=15, weight="bold"), text_color="white")
        self.logo_label.grid(row=1, column=1, padx=(20,0), pady=(25, 10), sticky="w")

        self.status_text = customtkinter.CTkLabel(self.mid_left_panel, text="Idle", font=customtkinter.CTkFont(family='Bahnschrift SemiLight', size=15, weight="bold"), text_color="black")
        self.status_text.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.status_text2 = customtkinter.CTkLabel(self.bottom_left_panel, text="Terminal Output:", font=customtkinter.CTkFont(family='Bahnschrift SemiLight', size=15, weight="bold"), text_color="white")
        self.status_text2.grid(row=0, column=0, padx=(10,10), pady=(5, 5), sticky="w")

        self.textbox = customtkinter.CTkTextbox(self.bottom_left_panel, width=340, height = 220, corner_radius=10)
        self.textbox.grid(row=1, column=0, padx=(10,10), pady=(2, 0))
        self.textbox.insert(customtkinter.END, "Welcome to OSerial 2025 - SP1\n")
        
        columns = ("NO", "STATUS", "PIN", "ACCURACY", "WEIGHT")
        self.tree = ttk.Treeview(self.bottom_right_panel2, columns=columns, show="headings")

        # Define column headings and widths

        self.tree.heading("NO", text="NO")
        self.tree.heading("STATUS", text="STATUS")
        self.tree.heading("PIN", text="PIN")
        self.tree.heading("ACCURACY", text="ACCUR")
        self.tree.heading("WEIGHT", text="WEIGHT")

        self.tree.column("NO", anchor="center")       # small for numbering
        self.tree.column("STATUS", anchor="center")  # wider for status text
        self.tree.column("PIN", anchor="center")
        self.tree.column("ACCURACY", anchor="center")
        self.tree.column("WEIGHT", anchor="center")

        self.tree.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        self.tree.bind("<Configure>", self._resize_tree_columns)
        self.bottom_right_panel2.bind("<Configure>", self._resize_tree_columns)
        self.after(100, self._resize_tree_columns)  # initial sizing once widgets exist

        style = ttk.Style(self)
        style.theme_use("clam")  # important: many Windows themes ignore custom colors
        style.configure(
            "Treeview",
            font=("Bahnschrift SemiLight", 18),   # <-- Table content font
            rowheight=40,                          # <-- Increase row height
            background="#070716",      # table bg
            fieldbackground="#070716", # cells bg
            foreground="#E5E7EB",      # text
            bordercolor="#070716",
            borderwidth=0
        )
        style.configure(
            "Treeview.Heading",
            background="#070716",
            foreground="#E5E7EB",
            font=("Bahnschrift SemiLight", 20, "bold")
        )
        style.map(
            "Treeview",
            background=[("selected", "#1D4ED8")],
            foreground=[("selected", "#FFFFFF")]
        )
        # Optional: dark scrollbars for ttk
        style.configure(
            "Vertical.TScrollbar",
            background="#0B0C27",
            troughcolor="#070716",
        )
        style.configure(
            "Horizontal.TScrollbar",
            background="#0B0C27",
            troughcolor="#070716",
        )
        # after creating self.tree
        self.tree.tag_configure("evenrow", background="#111827")
        self.tree.tag_configure("oddrow",  background="#0F172A")
        self.tree.tag_configure("rejected", background="#FF0000", foreground="white")  # dark red
        self.tree.tag_configure("accepted", background="#26FF00", foreground="black")  # dark green
        self.tree.tag_configure("warning",  background="#FFEE00", foreground="black")  # yellow-brown

        available_ports = serial_comm.list_ports()

        for port in available_ports:
            self.textbox.insert(customtkinter.END, f"Device: {port.device}\n") # Full device name/path (e.g., /dev/ttyUSB0, COM3)
            self.textbox.insert(customtkinter.END, f"Description: {port.description}") # Human-readable description
            self.textbox.insert(customtkinter.END, "\n" + "-" * 80 + "\n")
        self.textbox.configure(state="disabled")  # Make it read-only

        # Auto-detect CH340 port
        self.ch340_port = "COM1"  # default fallback
        for port in serial_comm.list_ports():
            if "CH340" in port.description.upper():
                self.ch340_port = port.device
                break
        
        port_list = serial_comm.list_ports()
        self.optionmenu_var = customtkinter.StringVar(value=self.ch340_port)
        self.optionmenu = customtkinter.CTkOptionMenu(
            self.left_side_panel,
            values=[port.device for port in port_list],
            command=self.optionmenu_callback,
            variable=self.optionmenu_var,
            width=140,
            fg_color="#232121",
            button_color="#080404",
            dropdown_fg_color="#232121",
            dropdown_hover_color="#2A2A2A"
        )
        self.optionmenu.grid(row=2, column=0, padx=2, pady=(15, 0), columnspan=1)
        self.optionmenu.grid_propagate(False)

        self.optionmenu2_var = customtkinter.StringVar(value="CAM0")
        self.optionmenu2 = customtkinter.CTkOptionMenu(
            self.left_side_panel,
            values=["CAM0", "CAM1", "CAM2"],
            command=self.optionmenu_callback2,
            variable=self.optionmenu2_var,
            width=140,
            fg_color="#232121",
            button_color="#080404",
            dropdown_fg_color="#232121",
            dropdown_hover_color="#2A2A2A"
        )
        self.optionmenu2.grid(row=2, column=1, padx=2, pady=(15, 0), columnspan=1)
        self.optionmenu2.grid_propagate(False)

        self._init_device_menus()

        self.optionmenu3_var = customtkinter.StringVar(value=f"TOLERANSI {globals.tolerance:.2f}")
        self.optionmenu3 = customtkinter.CTkOptionMenu(
            self.left_side_panel,
            values=["TOLERANSI 0.05", "TOLERANSI 0.03"],
            command=self.optionmenu_callback3,
            variable=self.optionmenu3_var,
            width=260,
            fg_color="#232121",
            button_color="#080404",
            dropdown_fg_color="#232121",
            dropdown_hover_color="#2A2A2A"
        )
        self.optionmenu3.grid(row=8, column=0, padx=2, pady=(0, 0), columnspan=2)
        self.optionmenu3.grid_propagate(False)

        self.bt_Start = customtkinter.CTkButton(self.left_side_panel, text="START", fg_color= "#00761A", hover_color = "#00340F", command= self.start_system)
        self.bt_Start.grid(row=4, column=0, padx=60, pady=(15,0), columnspan = 2, sticky="ew")
    
        self.bt_Stop = customtkinter.CTkButton(self.left_side_panel, text="STOP", fg_color= "#570000", hover_color = "#460101", command= self.stop_system)
        self.bt_Stop.grid(row=5, column=0, padx=60, pady=(10,10), columnspan = 2, sticky="ew")

        self.bt_statistics = customtkinter.CTkButton(self.left_side_panel, text="STATISTICS", fg_color= "#007A78", hover_color = "#004549", command= self.statistics)
        self.bt_statistics.grid(row=6, column=0, padx=60, pady=(0,10), columnspan = 2, sticky="ew")

        self.bt_refresh = customtkinter.CTkButton(self.left_side_panel, text="REFRESH", fg_color= "#3D007A", hover_color = "#320049", command=self.refresh_devices)
        self.bt_refresh.grid(row=7, column=0, padx=60, pady=(0,10), columnspan = 2, sticky="ew")

    def _init_device_menus(self):
        """Initialize serial & camera menus on first launch:
        - Serial: prefer CH340; if none, show '(no ports)'
        - Camera: default to CAM0
        """
        # ----- Serial (prefer CH340, else '(no ports)') -----
        ports = serial_comm.list_ports()
        ch340 = next((p for p in ports if "CH340" in p.description.upper()), None)

        if ch340:
            # show all ports but preselect the CH340
            port_values = [p.device for p in ports]
            self.optionmenu.configure(values=port_values)
            self.optionmenu_var.set(ch340.device)
        else:
            # explicitly show "(no ports)"
            self.optionmenu.configure(values=["(no ports)"])
            self.optionmenu_var.set("(no ports)")

        # ----- Camera (default to CAM0) -----
        self.optionmenu2.configure(values=["CAM0", "CAM1"])
        self.optionmenu2_var.set("CAM0")

    def _on_top_right_resize(self, event):
        # account for the label's internal padding (padx=10, pady=20)
        new_w = max(100, event.width  - 1000)
        new_h = max(100, event.height - 400)

        if (new_w, new_h) != (self.T_W, self.T_H):
            self.T_W, self.T_H = new_w, new_h

            # resize the CTkImage holder to the new target box
            if getattr(self, "ctk_img", None) is not None:
                self.ctk_img.configure(size=(self.T_W, self.T_H))

            # if we have a last frame, redraw it at the new size immediately
            if self._last_rgb is not None:
                rgb = self._cover_resize_crop_cv(self._last_rgb, self.T_W, self.T_H)
                self._set_main_display_from_rgb(rgb, self.T_W, self.T_H)

    def _resize_tree_columns(self, event=None):
        # how wide is the tree right now?
        w = max(1, self.tree.winfo_width())
        # pick proportions (sum ≈ 1.0); status is widest
        ratios = {
            "NO": 0.08,
            "STATUS": 0.46,
            "PIN": 0.18,
            "ACCURACY": 0.14,
            "WEIGHT": 0.14,
        }
        # keep things readable on small windows
        mins = {"NO": 70, "STATUS": 220, "PIN": 140, "ACCURACY": 120, "WEIGHT": 120}

        for col in ("NO", "STATUS", "PIN", "ACCURACY", "WEIGHT"):
            target = int(w * ratios[col])
            self.tree.column(
                col,
                width=max(target, mins[col]),
                stretch=(col == "STATUS"),   # only STATUS soaks up extra slack
                anchor="center",
            )

    # --- Tk main-thread scheduler helpers ---
    def ui_call(self, fn, *args, **kwargs):
        if self._closing:  # don't schedule after close begins
            return
        try:
            self.after(0, lambda: None)  # quick check that Tk is alive
            self.after(0, lambda: fn(*args, **kwargs))
        except Exception:
            pass
    
    def ui_after(self, ms, fn, *args, **kwargs):
        if self._closing:
            return
        def _wrapped():
            self._after_tokens.discard(token)
            if not self._closing:
                try:
                    fn(*args, **kwargs)
                except Exception:
                    pass
        try:
            token = self.after(ms, _wrapped)
            self._after_tokens.add(token)
        except Exception:
            pass
    # ----------------------------------------

    def handle_close(self):
        """Handle window close event with delay"""
        self._closing = True
        # cancel all scheduled callbacks
        for t in list(self._after_tokens):
            try: self.after_cancel(t)
            except Exception: pass
        self._after_tokens.clear()

        self.stop_streaming()
        self.stop_system()
        self.textbox.configure(state="normal")
        self.textbox.insert(customtkinter.END, "Closing System...\n")
        self.textbox.configure(state="disabled")
        self.textbox.see("end")

        # Disable the close button so user can't press it again
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        # Schedule the actual close
        self.after(300, self.force_close)

    def force_close(self):
        global cam
        with getattr(self, "cam_lock", threading.Lock()):
            if cam and cam.isOpened():
                cam.release()
        # Explicitly drop Tk image refs so their __del__ doesn’t run after Tcl dies
        try:
            self.main_display.configure(image=None)
            self.prevocr_label.configure(image=None)
        except Exception:
            pass
        self.ctk_img = None
        # If you stored other CTkImage objects on attributes, set them to None too.
        self.destroy()
    
    def _ui_tick(self):
        """Main-thread UI ticker: render the newest frame if available."""
        try:
            if self._frame_q:
                try:
                    rgb = self._frame_q.pop()
                except IndexError:
                    rgb = None
                if rgb is not None:
                    if rgb is not None:
                        self._last_rgb = rgb
                        self._set_main_display_from_rgb(rgb, self.T_W, self.T_H)
        except Exception as e:
            # Avoid crashing the loop; log and continue
            print(f"[UI tick] render error: {e}")
        finally:
            # Schedule next tick (~66 FPS request; real FPS depends on workload)
            self.after(15, self._ui_tick)

    def _set_main_display_from_rgb(self, rgb_img, target_w, target_h):
        if rgb_img.shape[1] != target_w or rgb_img.shape[0] != target_h:
            interp = cv2.INTER_AREA if (rgb_img.shape[1] > target_w or rgb_img.shape[0] > target_h) else cv2.INTER_LINEAR
            rgb_img = cv2.resize(rgb_img, (target_w, target_h), interpolation=interp)

        pil_img = Image.fromarray(rgb_img)
        # reuse the same CTkImage object, just swap its PIL image
        self.ctk_img.configure(light_image=pil_img, size=(target_w, target_h))
        # no need to recreate or reassign on the label every frame

    def _cover_resize_crop_cv(self, rgb_img, target_w, target_h):
        """Scale to cover target box, then center-crop to exact size."""
        h, w = rgb_img.shape[:2]
        scale = max(target_w / w, target_h / h)      # cover, not fit
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        x1 = max(0, (new_w - target_w) // 2)
        y1 = max(0, (new_h - target_h) // 2)
        cropped = resized[y1:y1 + target_h, x1:x1 + target_w]
        return cropped  # still RGB, (target_h, target_w, 3)

    def _run_onnx_box(self, session, frame_bgr):
        """
        Run a YOLO-style ONNX model (pinregion/textbox) on a BGR frame and
        return (x1, y1, x2, y2, conf). If no box, returns None.
        """
        h0, w0 = frame_bgr.shape[:2]

        # Input metadata (assumes NCHW with fixed H,W)
        inp = session.get_inputs()[0]
        in_name = inp.name
        _, _, in_h, in_w = inp.shape

        # Preprocess  BGR->RGB, resize, float32, [0..1], CHW, BCHW
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        # Inference -> (1, 5, 8400) with [cx, cy, w, h, conf]
        out = session.run(None, {in_name: img})[0]
        pred = out[0].T  # (8400, 5)

        if pred.shape[0] == 0:
            return None

        best_idx = int(np.argmax(pred[:, 4]))
        cx, cy, w, h, conf = pred[best_idx]

        # xyxy in resized space
        x1r, y1r = cx - w/2.0, cy - h/2.0
        x2r, y2r = cx + w/2.0, cy + h/2.0

        # map back to original image space
        sx, sy = w0 / float(in_w), h0 / float(in_h)
        x1, y1 = int(round(x1r * sx)), int(round(y1r * sy))
        x2, y2 = int(round(x2r * sx)), int(round(y2r * sy))

        # clip
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0 - 1, x2), min(h0 - 1, y2)

        return (x1, y1, x2, y2, float(conf))

    def refresh_devices(self, max_cam_index=5):
        """Refresh serial port list and camera list, updating both option menus."""
        global cam

        # ---------- Serial ports ----------
        ports = serial_comm.list_ports()
        port_values = [p.device for p in ports] or ["(no ports)"]
        current_port = self.optionmenu_var.get()

        # Auto-pick CH340 if present, else keep current, else first item
        ch340 = next((p.device for p in ports if "CH340" in p.description.upper()), None)
        new_port_sel = ch340 or (current_port if current_port in port_values else (port_values[0] if port_values else ""))

        self.optionmenu.configure(values=port_values)
        self.optionmenu_var.set(new_port_sel)

        # ---------- Cameras ----------
        cam_values = []
        # Release current cam while probing to avoid device locking
        with self.cam_lock:
            try:
                if cam is not None and cam.isOpened():
                    cam.release()
            except Exception:
                pass

        for i in range(max_cam_index + 1):  # probe 0..N
            try:
                test = cv2.VideoCapture(i)
                ok = test.isOpened()
                # Try a quick read to filter ghost devices
                if ok:
                    ok, _ = test.read()
                test.release()
                if ok:
                    cam_values.append(f"CAM{i}")
            except Exception:
                # ignore bad indices
                pass

        if not cam_values:
            cam_values = ["(no cameras)"]

        current_cam = self.optionmenu2_var.get()
        new_cam_sel = current_cam if current_cam in cam_values else cam_values[0]

        self.optionmenu2.configure(values=cam_values)
        self.optionmenu2_var.set(new_cam_sel)

        # ---------- Log to terminal ----------
        self.textbox.configure(state="normal")
        self._append_textbox("\n=== REFRESH ===\n")
        self._append_textbox(f"Serial ports: {', '.join(port_values)}\n")
        self._append_textbox(f"Cameras: {', '.join(cam_values)}\n")
        self.textbox.configure(state="disabled")
        self.textbox.see("end")

        # If streaming is active and the selected camera changed due to refresh,
        # reopen the selected cam so preview keeps working.
        if self.streaming_active:
            if new_cam_sel.startswith("CAM"):
                self._open_cam_for_current_choice()

    def _apply_cam_fast_settings(self, cap):
        """Fast-to-open tweaks without changing resolution or FPS."""
        # Keep buffer tiny to reduce latency/stalls right after switching
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # Prefer MJPG (opens faster on many USB cams; avoids slow YUY2 negotiation)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

    def _open_camera_blocking(self, cam_index):
        """Open a camera (fast path) and warm it up. Returns an opened cap or None."""
        # Prefer DirectShow on Windows; fall back to MSMF.
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, 0):
            cap = None
            try:
                cap = cv2.VideoCapture(cam_index, backend)
            except Exception:
                cap = cv2.VideoCapture(cam_index)  # last-resort generic
            if not cap or not cap.isOpened():
                if cap: cap.release()
                continue
            # Force fast settings & warm-up a few frames (prevents first read stall)
            self._apply_cam_fast_settings(cap)
            ok = False
            for _ in range(6):
                ok, _ = cap.read()
                if ok: break
                time.sleep(0.01)
            if ok:
                return cap
            cap.release()
        return None

    def switch_camera_async(self, cam_index):
        """Open the new camera off the UI thread; hot-swap when ready."""
        # Avoid starting multiple openers at once
        if getattr(self, "_cam_switching", False):
            return
        self._cam_switching = True

        def _worker():
            try:
                new_cap = self._open_camera_blocking(cam_index)
                if not new_cap:
                    self.ui_call(self._append_textbox, f"[ERROR] Camera {cam_index} failed to open.\n")
                    return
                # Hot-swap under lock to prevent read() races
                global cam
                with self.cam_lock:
                    old = cam
                    cam = new_cap
                # Release old after swap (outside lock)
                try:
                    if old is not None and old.isOpened():
                        old.release()
                except Exception:
                    pass
                self.ui_call(self._append_textbox, f"Camera switched to index {cam_index}\n")
            finally:
                self._cam_switching = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def camera_loop(self):
        global cam, model, model2
        self._last_infer_t = 0.0  # (re)start throttle timer

        while self.streaming_active:
            with self.cam_lock:
                ret, frame_bgr = cam.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Throttle ONNX inference so UI stays smooth
            now = time.perf_counter()
            do_infer = (now - self._last_infer_t) >= getattr(self, "_infer_interval", 0.05)

            if do_infer:
                self._last_infer_t = now
                try:
                    # run both sessions
                    box1 = self._run_onnx_box(model,  frame_bgr)   # pinregion.onnx
                    box2 = self._run_onnx_box(model2, frame_bgr)   # textbox.onnx

                    new_cache = []
                    th = getattr(self, "_conf_thresh", 0.5)
                    if box1 and box1[4] >= th:
                        new_cache.append((*box1, "M1", (0, 255, 0), now))
                    if box2 and box2[4] >= th:
                        new_cache.append((*box2, "M2", (255, 0, 0), now))
                    self._boxes_live = new_cache

                except Exception as e:
                    # never crash the camera thread
                    print(f"[camera_loop] inference error: {e}")

            # ---- ALWAYS draw the most recent boxes (until TTL expires) ----
            thick = max(2, int(round(min(frame_bgr.shape[0], frame_bgr.shape[1]) * 0.003)))
            kept = []
            th = getattr(self, "_conf_thresh", 0.5)
            for (x1, y1, x2, y2, conf, label, color, t0) in self._boxes_live:
                if (now - t0 <= getattr(self, "_box_ttl", 1.0)) and (conf >= th):
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thick)
                    cv2.putText(frame_bgr, f"{label} {conf:.2f}",
                                (x1+3, max(y1-5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    kept.append((x1, y1, x2, y2, conf, label, color, t0))
            self._boxes_live = kept  # <-- make sure to keep this assignment

            # hand off the newest frame to the UI thread
            # (convert to RGB and resize here so UI just displays)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tw, th = self.T_W, self.T_H
            try:
                rgb = self._cover_resize_crop_cv(rgb, tw, th)  # your existing helper
            except Exception:
                # if helper not available, fall back to plain resize
                rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)

            # keep only the newest frame for UI
            self._frame_q.append(rgb)

            # ~30–33 FPS pacing
            time.sleep(0.030)

    def start_streaming(self):
        self.streaming_active = True
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def stop_streaming(self):
        if not getattr(self, "streaming_active", False):
            return
        self.streaming_active = False
        t = getattr(self, "camera_thread", None)
        if t and t.is_alive():
            t.join(timeout=3.0)

    def statistics(self):
        # Let stats_utils discover today's CSV in logs/csv/YYYY/MM/YYYY-MM-DD.csv
        stats_utils.show_statistics_window(self, tolerance=globals.tolerance, base_dir="logs/csv")

    def optionmenu_callback2(self, choice):
        global cam
        if choice.startswith("CAM"):
            cam_index = int(choice.replace("CAM", ""))
            # Non-blocking switch (UI stays responsive)
            self.switch_camera_async(cam_index)
        else:
            print("Serial Port Selected:", choice)
            self._append_textbox(f"Serial Port Selected: {choice}\n")

    def optionmenu_callback3(self, choice):
        val = float(choice.replace("TOLERANSI ", ""))
        globals.tolerance = val
        self._append_textbox(f"Tolerance Level Selected: {val:.2f} kg\n")

        # --- Quick hack: overwrite tolerance in globals.py ---
        try:
            file_path = os.path.join("DATA", "py_files", "globals.py")
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(file_path, "w", encoding="utf-8") as f:
                for line in lines:
                    if line.strip().startswith("tolerance ="):
                        f.write(f"tolerance = {val:.2f}  # in Kg\n")
                    else:
                        f.write(line)
            self._append_textbox("Globals.py updated successfully.\n")
        except Exception as e:
            self._append_textbox(f"Error updating globals.py: {e}\n")

    def revert_maindisp(self):
        self.image_path = "DATA/images/startuplogo.png"
        pil_img = Image.open(self.image_path)
        ctk_img = CTkImage(light_image=Image.open(self.image_path), size=(990, 500))
        self.main_display.configure(image=ctk_img)
        self.main_display.image = ctk_img  # keep a reference so it doesn't get garbage-collected

    # stop serial connection  
    def stop_system(self): 
        global esp32
        self.stop_streaming()  # stops the camera thread
        self._release_cam()
        if not self._closing:
            self.ui_after(500, self.revert_maindisp)
        self.status_text.configure(text="Stopping Connection...", text_color="white", font=customtkinter.CTkFont(family='Bahnschrift SemiLight', size=12, weight="bold"))
        self.ui_call(self.mid_left_panel.configure, fg_color="red")
        # Robust: Only stop serial_thread if it exists and serial is open
        if hasattr(self, "serial_thread") and self.serial_thread is not None:
            try:
                # Only stop if port is open (prevents pyserial cancel_read bug)
                if getattr(esp32, "is_open", False):
                    self.serial_thread.stop()  # Stop the reader thread
                    self.serial_thread.join(timeout=1)
            except Exception as e:
                print(f"Error stopping serial thread: {e}")
            self.serial_thread = None  # Mark as closed
        if getattr(esp32, "is_open", False):
            esp32.close()  # Close the serial port connection
            self._append_textbox(f"Disconnecting from serial port {self.optionmenu_var.get()}\n")
            self._append_textbox("Serial port closed successfully.\n")
            self.status_text.configure(text="Connection Stopped", font=customtkinter.CTkFont(family='Bahnschrift SemiLight', size=12, weight="bold"), text_color="black")
            self.ui_call(self.mid_left_panel.configure, fg_color="yellow")
        else:
            self.after(2000, lambda: self.status_text.configure(
                text="No Serial Connection to Stop",
                font=customtkinter.CTkFont(family='Bahnschrift SemiLight', size=12, weight="bold"),
                text_color="white"
                ))
            
    def _release_cam(self):
        """Release the current OpenCV camera handle safely."""
        global cam
        with self.cam_lock:
            try:
                if cam is not None and cam.isOpened():
                    cam.release()
            except Exception:
                pass
    
    def _open_cam_for_current_choice(self):
        """(Re)open the camera based on the option menu selection."""
        global cam
        # parse e.g. "CAM1" -> 1
        choice = self.optionmenu2_var.get()
        cam_index = 0
        try:
            if str(choice).upper().startswith("CAM"):
                cam_index = int(str(choice).upper().replace("CAM", ""))
        except Exception:
            cam_index = 0

        with self.cam_lock:
            # make sure previous handle is closed first
            try:
                if cam is not None and cam.isOpened():
                    cam.release()
            except Exception:
                pass
            cam = cv2.VideoCapture(cam_index)

    def optionmenu_callback(self, choice):
        print("Serial Port Selected:", choice)
        self._append_textbox(f"Serial Port Selected: {choice}\n")

    def start_system(self):
        global esp32  # use the global serial object 

        self.ctk_img = customtkinter.CTkImage(light_image=Image.new("RGB", (self.T_W, self.T_H)),
                            size=(self.T_W, self.T_H))
        self.main_display.configure(image=self.ctk_img)
        self._open_cam_for_current_choice()
        self.start_streaming()
        self.status_text.configure(text="Initializing...", text_color="black", font=customtkinter.CTkFont(family='Bahnschrift SemiLight', size=12, weight="bold"))
        try:
            selected_port = self.optionmenu_var.get()

            # Get the description for the selected port
            description = "Unknown"
            for p in serial_comm.list_ports():
                if p.device == selected_port:
                    description = p.description
                    break

            esp32 = serial_comm.open_serial(port=selected_port, baudrate=115200)

            # Start threaded reader
            self.serial_thread = serial_comm.ReaderThread(
                esp32,
                lambda: serial_comm.SerialReader(self, ignore_secs=1.0)  # adjust if you want longer/shorter
            )
            self.serial_thread.start()

            self._append_textbox("Serial port opened successfully.\n")
            self._append_textbox(f"Connected to {selected_port}\n")
            self._append_textbox(f"Description: {description}\n")

            self.ui_call(self.mid_left_panel.configure, fg_color="green")
            if getattr(esp32, "is_open", False):
                self.status_text.configure(
                    text="Serial Connection Established",
                    font=customtkinter.CTkFont(family='Bahnschrift SemiLight', size=10, weight="bold"), text_color="black"
                )   
        except SerialException as e:
            print(f"Error opening serial port: {e}")
            self.status_text.configure(text="Serial Connection Error", font=customtkinter.CTkFont(family='Bahnschrift SemiLight', size=12, weight="bold"), text_color="white")
            self._append_textbox(f"Error opening serial port: {e}\n")
            self.ui_call(self.mid_left_panel.configure, fg_color="red")
            pass
    
    def append_row(self, timestamp, no, status, pin, accuracy, weight, retries=2):
        row = [timestamp, no, status, pin, accuracy, weight]
        ts_epoch = time.mktime(datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').timetuple())
        path = _daily_csv_path(ts_epoch)
        _ensure_header(path)

        for i in range(retries + 1):
            try:
                with open(path, mode="a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
                return
            except Exception as e:
                if i == retries:
                    print(f"[CSV] write failed: {e}")
                else:
                    time.sleep(0.2)

    def doubledata(self):
        # Let stats_utils discover today's CSV in logs/csv/YYYY/MM/YYYY-MM-DD.csv
        doubledata.show_ddata_window(self, tolerance=globals.tolerance, base_dir="logs/csv")
                   
    def updateLabelData(self, data):
        print("Updating data:")
        try:
            val = data.decode('utf-8').strip()
        except Exception as e:
            print("Error parsing data:", e)
        if "|" in val:
            button_str, mass_str = val.split("|")
            button_state = int(button_str)
            mass_value = float(mass_str)
            print(f"Status: {button_state}, Mass: {mass_value}")
            self.ui_call(self._append_textbox, f"Status: {button_state}, Mass: {mass_value}\n")
        else:
            print("Invalid format:", val)
            self.ui_call(self._append_textbox, f"Invalid format:" + val + "\n")
            return
        globals.tempdata5 = mass_value
        if button_state == 1 and (5 - globals.tolerance) < mass_value < (5 + globals.tolerance):
            self.ui_call(self.bottom_right_panel1.configure, fg_color="#00FF08")
            self.ui_call(self.status_label.configure, text="ACC")
            globals.state = "ACCEPTED"
        else:
            self.ui_call(self.bottom_right_panel1.configure, fg_color="#FF0000")
            # choose short text once, then:
            txt = "RJT"
            if button_state == 0 and mass_value < 5 - globals.tolerance: txt = "RJT\nLM-LT"
            elif button_state == 0 and mass_value > 5 + globals.tolerance: txt = "RJT\nHM-LT"
            elif button_state == 0: txt = "RJT\nLT"
            elif mass_value < 5 - globals.tolerance: txt = "RJT\nLM"
            elif mass_value > 5 + globals.tolerance: txt = "RJT\nHM"
            self.ui_call(self.status_label.configure, text=txt)
            # set state2 once:
            if "LM" in txt:
                globals.state2 = "-MASS LOW"
            elif "HM" in txt:
                globals.state2 = "-MASS HIGH"
            if "LT" in txt:
                globals.state2 += "-LEAK TEST FAIL"
            globals.state = "REJECTED"
        
        self.capture_image()  # Call the capture_image function to take a picture
        self.yolo_detect()  # Call the YOLO detection function
        self.rfdetr_detect()
        ocr_outcome = self.ocr_detect()   # <--- get outcome flag
        
        self.ui_call(self._append_textbox, "Processing Done")

        self.image2_path = "DATA/TEMP/yolotemp_ocr_res_img.jpg"

        img_bgr2 = cv2.imread(self.image2_path)
        if img_bgr2 is not None:
            img_rgb2 = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB)
            pil2 = Image.fromarray(img_rgb2)
            self.ui_call(
                self.prevocr_label.configure,
                image=CTkImage(light_image=pil2, size=(int(780/2.5), int(215/2.5)))
            )

        self.stop_streaming()

        self.image_path = "DATA/TEMP/yolotemp2_ocr_res_img.jpg"

        img_bgr = cv2.imread(self.image_path)
        if img_bgr is not None:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            rgb = self._cover_resize_crop_cv(rgb, self.T_W, self.T_H)
            self.ui_call(self._set_main_display_from_rgb, rgb, self.T_W, self.T_H)
        if not self._closing:
            self.ui_after(4000, self.start_streaming)  # Restart streaming after 4 seconds

        accur = (float(globals.tempdata3) + float(globals.tempdata4))/2
        acc_str = f"{accur*100:.2f}%"

        # --- Normalize and compare the OCR PINs from YOLO, RFDETR, and largeYOLO as 7-digit strings ---
        # tempdata1 = YOLO pin,  tempdata3 = YOLO conf
        # tempdata2 = RFDETR pin, tempdata4 = RFDETR conf
        # tempdata6 = largeYOLO pin, tempdata7 = largeYOLO conf
        def norm7(s: str) -> str:
            # keep exactly 7 digits; anything else => ""
            print(s)
            s = (s or "").strip()
            m = re.findall(r"(?<!\d)(\d{7})(?!\d)", s)
            if m:
                candidate = m[0].lstrip("0")
                return candidate if len(candidate) == 7 else ""
            s_digits = re.sub(r"\D", "", s).lstrip("0")
            return s_digits if len(s_digits) == 7 else ""

        def is_valid(pin: str) -> bool:
            return bool(pin) and pin != "0000000"

        pin_yolo_raw   = str(globals.tempdata1 or "")
        pin_rfdetr_raw = str(globals.tempdata2 or "")
        pin_large_raw  = str(globals.tempdata6 or "")

        conf_yolo      = float(globals.tempdata3 or 0.0)
        conf_rfdetr    = float(globals.tempdata4 or 0.0)
        conf_large     = float(globals.tempdata7 or 0.0)

        # Normalize then treat "0000000" as NO DETECTION
        p_yolo  = norm7(pin_yolo_raw)
        p_rfd   = norm7(pin_rfdetr_raw)
        p_large = norm7(pin_large_raw)

        if p_yolo  == "0000000": p_yolo  = ""
        if p_rfd   == "0000000": p_rfd   = ""
        if p_large == "0000000": p_large = ""

        agree_all  = (is_valid(p_yolo) and is_valid(p_rfd) and is_valid(p_large) and (p_yolo == p_rfd == p_large))
        agree_yr   = (is_valid(p_yolo) and is_valid(p_rfd) and (p_yolo == p_rfd))
        none_yr    = (not is_valid(p_yolo) and not is_valid(p_rfd))
        large_ok   = (is_valid(p_large) and conf_large >= 0.90)
        agree_yl = (is_valid(p_yolo) and is_valid(p_large) and (p_yolo == p_large))
        agree_rl = (is_valid(p_rfd)  and is_valid(p_large) and (p_rfd  == p_large))

        # choose a single row number for Treeview + CSV
        no = globals.detnumber

        if agree_all:
            # 1) RFDETR + YOLO + largeYOLO agree -> use that PIN
            usedPIN = p_yolo
            status_to_log = globals.state

        elif agree_yr:
            # 2) RFDETR & YOLO agree (regardless of largeYOLO) -> use their PIN
            usedPIN = p_yolo
            status_to_log = globals.state
            # acc_str already based on (yolo+rfdetr); keep as-is

        elif agree_yl and not agree_all:
            # NEW: YOLO & largeYOLO agree; RFDETR disagrees or is invalid -> use Y+L and mark mismatch
            usedPIN = p_yolo  # == p_large
            pair_conf = (float(conf_yolo) + float(conf_large)) / 2.0
            acc_str = f"{pair_conf * 100:.2f}%"
            status_to_log = globals.state

        elif agree_rl and not agree_all:
            # NEW: RFDETR & largeYOLO agree; YOLO disagrees or is invalid -> use R+L and mark mismatch
            usedPIN = p_rfd  # == p_large
            pair_conf = (float(conf_rfdetr) + float(conf_large)) / 2.0
            acc_str = f"{pair_conf * 100:.2f}%"
            status_to_log = globals.state

        elif none_yr and large_ok:
            # 3) RFDETR & YOLO have no PIN, but largeYOLO detects with conf > 0.90
            usedPIN = p_large
            status_to_log = globals.state
            acc_str = f"{conf_large * 100:.2f}%"

        else:
            # 4) mismatch or no valid PIN overall -> open manual resolver window
            # Decide error label for logging and screenshot
            any_valid = any([is_valid(p_yolo), is_valid(p_rfd), is_valid(p_large)])
            err_lbl = "MSMTCH" if any_valid else "NOPIN"

            # Open resolver window (modal) and wait for user PIN
            # Show the raw capture image (change to yolotemp*.jpg if you prefer)
            tempimg_path = "DATA/TEMP/tempimg.jpg"
            done = threading.Event()
            self.ui_call(self._open_ocr_error_modal, err_lbl, tempimg_path, "", done)
            done.wait()  # waits in the worker thread; UI stays responsive
            win = getattr(self, "_last_ocr_error_window", None)  # optional if you store it there
            manual_pin = getattr(win, "result_pin", "") if win is not None else ""
            if manual_pin:
                # Use the manually entered PIN
                usedPIN = manual_pin
                status_to_log = globals.state
                acc_str = "MANUAL"
                self.save_failure_image("MANUAL")
            else:
                # User canceled -> graceful fallback to highest-confidence valid PIN (if any)
                candidates = []
                if is_valid(p_yolo):
                    candidates.append(("YOLO", p_yolo, conf_yolo))
                if is_valid(p_rfd):
                    candidates.append(("RFDETR", p_rfd, conf_rfdetr))
                if is_valid(p_large):
                    candidates.append(("LARGEYOLO", p_large, conf_large))

                if not candidates:
                    usedPIN = ""      # truly no PIN
                    status_to_log = globals.state + " NOPIN"
                    # keep acc_str as-is or mark as 0%
                    acc_str = "0.00%"
                    self.save_failure_image(err_lbl)
                else:
                    # Consolidate identical PINs (take max confidence per PIN)
                    by_pin = {}
                    for _, pin, c in candidates:
                        by_pin[pin] = max(by_pin.get(pin, 0.0), float(c))
                    usedPIN, best_conf = max(by_pin.items(), key=lambda kv: kv[1])
                    acc_str = f"{best_conf * 100:.2f}%"
                    status_to_log = globals.state + " MSMTCH"
                    self.save_failure_image(err_lbl)

        # --- DUPLICATE CHECK across ALL historical CSV logs ---
        if usedPIN:
            dup_info = doubledata.find_existing_pin(usedPIN, base_dir="logs/csv")
            if dup_info:
                done = threading.Event()
                self.ui_call(self._open_duplicate_pin_modal, usedPIN, dup_info, done)
                done.wait()  # wait in worker thread; UI stays responsive

                win = getattr(self, "_last_duplicate_pin_window", None)
                action = getattr(win, "result_action", None) if win is not None else "erase"
                replacement_pin = getattr(win, "result_pin", "") if win is not None else ""

                if action == "erase":
                    # Operator chose to discard this reading entirely.
                    self.ui_call(self._append_textbox, f"Duplicate PIN {usedPIN} -> entry discarded.\n")
                    return  # <- stop here: don't add row or append CSV
                elif action == "replace":
                    usedPIN = replacement_pin  # <- continue to log with new PIN
                    acc_str = "MANUAL"

        # record to UI + CSV
        self.ui_call(self.add_row_to_tree, 
            no, 
            status_to_log + globals.state2, 
            usedPIN, 
            acc_str, 
            round(mass_value, 4)
        )

        print(f"Row {no}: {status_to_log}, PIN: {usedPIN}, Accuracy: {acc_str}, Mass: {mass_value}")
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        self.append_row(timestamp, no, status_to_log + globals.state2, usedPIN, acc_str, mass_value)
        doubledata.pin_index_add(usedPIN)  # optional: keep cache updated
        globals.detnumber += 1

    def _open_ocr_error_modal(self, err_lbl, tempimg_path, preset_pin, done_event):
        try:
            win = ocrerrorutils.show_ocr_error_window(self, err_lbl, tempimg_path, preset_pin=preset_pin)
            self._last_ocr_error_window = win
            if win is None:
                done_event.set()
                return
            win.bind("<Destroy>", lambda e: done_event.set())
        except Exception as e:
            print(f"[OCR modal] failed: {e}")
            done_event.set()
    
    def _open_duplicate_pin_modal(self, pin, dup_info, done_event):
        try:
            win = doubledata.show_duplicate_pin_window(self, pin, dup_info)
            self._last_duplicate_pin_window = win
            if win is None:
                done_event.set()
                return
            win.bind("<Destroy>", lambda e: done_event.set())
        except Exception as e:
            print(f"[Dup PIN modal] failed: {e}")
            done_event.set()

    def capture_image(self):
        global cam
        img_name = imagetool.capture_image(cam)
        return img_name
    
    def save_failure_image(self, case_label):
        # case_label: e.g., "NOPIN", "MSMTCH", etc.
        src_img = "DATA/TEMP/tempimg.jpg"
        now = datetime.datetime.now()
        folder = f"FailureLog/{now.strftime('%Y-%m-%d')}_{case_label}"
        os.makedirs(folder, exist_ok=True)
        # Save with timestamp in filename to avoid overwrite
        filename = now.strftime("%Y%m%d_%H%M%S") + ".jpg"
        dest_img = os.path.join(folder, filename)
        shutil.copy2(src_img, dest_img)
        print(f"Failure image saved to: {dest_img}")
        self.ui_call(self._append_textbox, f"Failure image saved to: {dest_img}\n")
        # --- Prune old folders: only keep last month ---
        base_dir = "FailureLog"
        if not os.path.isdir(base_dir):
            return  # nothing to prune on first run
        cutoff = now - datetime.timedelta(days=31)
        for entry in os.scandir(base_dir):
            if entry.is_dir():
                # Try to parse the date from the folder name
                try:
                    folder_date = datetime.datetime.strptime(entry.name[:10], '%Y-%m-%d')
                    if folder_date < cutoff:
                        shutil.rmtree(entry.path)
                        print(f"Deleted old folder: {entry.path}")
                except Exception as e:
                    print(f"Skipping folder {entry.name}: {e}")

    def yolo_detect(self):
        """
        Run ONNX (YOLO-style) sessions 'model' and 'model2' on DATA/TEMP/tempimg.jpg,
        crop to the best box if conf >= threshold, otherwise save the original.
        Outputs:
        - DATA/TEMP/yolotemp.jpg   (from model  / pinregion)
        - DATA/TEMP/yolotemp2.jpg  (from model2 / textbox)
        """
        global model, model2
        image_path = "DATA/TEMP/tempimg.jpg"
        img = cv2.imread(image_path)
        if img is None:
            self.ui_call(self._append_textbox, f"Missing image: {image_path}\n")
            return

        H0, W0 = img.shape[:2]
        conf_thr = getattr(self, "_conf_thresh", 0.5)

        def _best_box_crop(session, src_bgr, save_path):
            """
            Session outputs shape (1,5,8400) with [cx, cy, w, h, conf] per anchor (as in test3.py).
            Returns (found, nboxes, conf_used).
            """
            try:
                inp = session.get_inputs()[0]
                in_name = inp.name
                _, _, in_h, in_w = inp.shape

                # preprocess: BGR->RGB, resize to model size, 0..1, NCHW
                rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
                blob = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

                out = session.run(None, {in_name: blob})[0]  # expected (1,5,8400)
                pred = out[0].T                              # (8400,5) -> [cx,cy,w,h,conf]
                if pred.shape[0] == 0:
                    cv2.imwrite(save_path, src_bgr)
                    return False, 0, 0.0

                i = int(np.argmax(pred[:, 4]))
                cx, cy, w, h, conf = pred[i].tolist()

                # convert to xyxy (resized space)
                x1r, y1r = cx - w/2.0, cy - h/2.0
                x2r, y2r = cx + w/2.0, cy + h/2.0

                # map back to original image space
                sx, sy = W0 / float(in_w), H0 / float(in_h)
                x1 = int(round(x1r * sx)); y1 = int(round(y1r * sy))
                x2 = int(round(x2r * sx)); y2 = int(round(y2r * sy))

                # clip
                x1 = max(0, min(W0 - 1, x1)); y1 = max(0, min(H0 - 1, y1))
                x2 = max(0, min(W0 - 1, x2)); y2 = max(0, min(H0 - 1, y2))

                if conf >= conf_thr and x2 > x1 and y2 > y1:
                    crop = src_bgr[y1:y2, x1:x2]
                    cv2.imwrite(save_path, crop)
                    return True, pred.shape[0], float(conf)
                else:
                    # low confidence or invalid box -> save original
                    cv2.imwrite(save_path, src_bgr)
                    return False, pred.shape[0], float(conf)

            except Exception as e:
                print(f"[yolo_detect] ONNX error: {e}")
                # on any error, save original to keep pipeline going
                cv2.imwrite(save_path, src_bgr)
                return False, 0, 0.0

        # Model 1 -> yolotemp.jpg
        ok1, n1, c1 = _best_box_crop(model,  img, "DATA/TEMP/yolotemp.jpg")
        # Model 2 -> yolotemp2.jpg
        ok2, n2, c2 = _best_box_crop(model2, img, "DATA/TEMP/yolotemp2.jpg")

        self.ui_call(self._append_textbox,
            f"\nYOLO(ONNX) — M1: {'crop' if ok1 else 'full'} (conf={c1:.2f})\n"
            f"YOLO(ONNX) — M2: {'crop' if ok2 else 'full'} (conf={c2:.2f})\n"
        )

    def rfdetr_detect(self):
        global model3
        IMAGE_PATH = "DATA/TEMP/tempimg.jpg"
        CONF_THRESH = 0.5
        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        def letterbox(img, new_shape):
            h0, w0 = img.shape[:2]
            w, h = new_shape
            r = min(w / w0, h / h0)
            new_size = (int(w0 * r), int(h0 * r))
            resized = cv2.resize(img, new_size)
            pad_w, pad_h = w - new_size[0], h - new_size[1]
            out = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(114,114,114))
            return out, r, (w0, h0)
        def cxcywh_to_xyxy(box):
            cx, cy, w, h = box
            return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
        in_name = model3.get_inputs()[0].name
        _, _, in_h, in_w = model3.get_inputs()[0].shape

        orig = cv2.imread(IMAGE_PATH)
        H0, W0 = orig.shape[:2]

        img, scale, (W0_, H0_) = letterbox(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), (in_w, in_h))
        x = img.astype(np.float32)/255.0
        x = (x - MEAN)/STD
        x = x.transpose(2,0,1)[None]

        outputs = model3.run(None, {in_name: x})
        # Find boxes/logits among outputs
        boxes, logits = None, None
        for o in outputs:
            if o.ndim == 3 and o.shape[-1] == 4: boxes = o[0]
            elif o.ndim == 3: logits = o[0]

        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        scores = probs.max(-1)

        best_idx = scores.argmax()
        print(f"Best detection index: {best_idx}, score: {scores[best_idx]}")
        if scores[best_idx] < CONF_THRESH:
            print("No detection above threshold.")
            return

        b = boxes[best_idx] * np.array([in_w, in_h, in_w, in_h])
        xyxy = cxcywh_to_xyxy(b)
        xyxy /= scale  # undo resize
        x1,y1,x2,y2 = np.clip(xyxy, [0,0,0,0], [W0-1,H0-1,W0-1,H0-1]).astype(int)

        crop = orig[y1:y2, x1:x2]
        cv2.imwrite(r"DATA/TEMP/yolotemp3.jpg", crop)

    def ocr_detect(self):
        def _mean_conf(char_conf_list):
            # char_conf_list is a list of per-character confidences; fall back to 0 if empty
            if not char_conf_list:
                return 0.0
            # if it’s a nested structure (e.g., list of arrays), flatten first
            vals = []
            for c in char_conf_list:
                try:
                    vals.append(float(c))
                except Exception:
                    try:
                        vals.append(float(np.asarray(c).item()))
                    except Exception:
                        pass
            return float(np.mean(vals)) if vals else 0.0

        def _run_one(image_path, out_img_path, out_json_path):
            """
            Runs det+rec on one image and returns (texts, conf_lists).
            Also saves a visualization and a minimal JSON (rec_texts/rec_scores).
            """
            frame = cv2.imread(image_path)
            if frame is None:
                return [], []

            image_vis = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # inplace

            # DETECT boxes (polygons), then sort for nicer reading order
            points = onnxdet(frame)
            points = pputil.sort_polygon(list(points))

            # draw polygons (green)
            int_points = [np.asarray(pt, dtype=np.int32) for pt in points]
            for pt in int_points:
                cv2.polylines(image_vis, [pt], True, (0, 255, 0), 2)

            # crop each polygon for recognition
            cropped_images = [pputil.crop_image(frame, x) for x in points]

            # RECOGNIZE each crop
            texts, conf_lists = ([], [])
            if cropped_images:
                texts, conf_lists = onnxrec(cropped_images)

            # draw recognized text near each polygon
            for pt, text in zip(int_points, texts):
                x, y, w, h = cv2.boundingRect(pt)
                cv2.putText(image_vis, text, (int(x), max(int(y) - 5, 12)),
                            cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            # save visualization
            cv2.imwrite(out_img_path, image_vis)

            # build JSON compatible with previous PaddleOCR outputs
            rec_texts  = list(map(str, texts))
            rec_scores = [ _mean_conf(c) for c in conf_lists ]
            tmp = out_json_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"rec_texts": rec_texts, "rec_scores": rec_scores}, f, ensure_ascii=False)
            os.replace(tmp, out_json_path)

            return rec_texts, rec_scores

        t0 = time.perf_counter()

        # Run for both sides
        texts1, scores1 = _run_one(
            "DATA/TEMP/yolotemp.jpg",
            "DATA/TEMP/yolotemp_ocr_res_img.jpg",
            "DATA/TEMP/yolotemp_res.json",
        )
        texts2, scores2 = _run_one(
            "DATA/TEMP/yolotemp2.jpg",
            "DATA/TEMP/yolotemp2_ocr_res_img.jpg",
            "DATA/TEMP/yolotemp2_res.json",
        )
        texts3, scores3 = _run_one(
            "DATA/TEMP/yolotemp3.jpg",
            "DATA/TEMP/yolotemp3_ocr_res_img.jpg",
            "DATA/TEMP/yolotemp3_res.json",
        )

        # Pick best 7-digit numeric token from each side
        def _best_pin(texts, scores):
            best_pin, best_conf = "0000000", 0.0
            for txt, sc in zip(texts, scores):
                # keep only digits when searching for 7-digit groups
                # try exact 7-digit tokens first
                m = re.findall(r"(?<!\d)(\d{7})(?!\d)", txt)
                if not m:
                    # relaxed: strip non-digits and check length==7
                    s = re.sub(r"\D", "", txt)
                    m = [s] if len(s) == 7 else []
                for cand in m:
                    if sc > best_conf:
                        best_pin, best_conf = cand, float(sc)
            return best_pin, best_conf

        pin1, conf1 = _best_pin(texts1, scores1)
        pin2, conf2 = _best_pin(texts2, scores2)
        pin3, conf3 = _best_pin(texts3, scores3)

        # Publish to globals (keeps your later logic intact)
        globals.tempdata1 = pin1
        globals.tempdata3 = conf1
        globals.tempdata2 = pin3
        globals.tempdata4 = conf3
        globals.tempdata6 = pin2
        globals.tempdata7 = conf2

        self.ui_call(self._append_textbox, f"7-digit ID (Method 1 - YOLOv11): {pin1} | Confidence: {conf1:.2%}\n")
        self.ui_call(self._append_textbox, f"7-digit ID (Method 2 - largeYOLO): {pin2} | Confidence: {conf2:.2%}\n")
        self.ui_call(self._append_textbox, f"7-digit ID (Method 3 - RFDETR): {pin3} | Confidence: {conf3:.2%}\n")
        
        t1 = time.perf_counter()
        self.ui_call(self._append_textbox, f"\nONNX OCR Completed in {t1 - t0:.2f} seconds\n")
        if pin1 == "0000000" and pin3 == "0000000":
            print("No valid PIN found in methods 1 and 3.")

        return "NOPIN" if (pin1 == "0000000" and pin3 == "0000000") else "OK"
    
    def add_row_to_tree(self, no, status, pin, accuracy, weight):
        # Decide color tag
        if "REJECTED" in status and "MSMTCH" not in status and "NOPIN" not in status:
            tag = "rejected"
        elif "ACCEPTED" in status and "MSMTCH" not in status and "NOPIN" not in status:
            tag = "accepted"
        elif "MSMTCH" in status or "NOPIN" in status:
            tag = "warning"
        else:
            # fallback to zebra if no match
            idx = len(self.tree.get_children())
            tag = "evenrow" if idx % 2 == 0 else "oddrow"

        iid = self.tree.insert("", "end",
                            values=(no, status, pin, accuracy, weight),
                            tags=(tag,))
        self.tree.see(iid)

    def _append_textbox(self, s: str):
        self.textbox.configure(state="normal")
        self.textbox.insert(customtkinter.END, s)
        self.textbox.configure(state="disabled")
        self.textbox.see("end")

# --- 2) Splash as CTkToplevel (NOT a CTk root) ---
class LoaderSplash(customtkinter.CTkToplevel):
    def _center_geometry(self, width: int, height: int, scale_factor: float = 1.0) -> str:
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = int(((sw/2) - (width/2)) * scale_factor)
        y = int(((sh/2) - (height/1.5)) * scale_factor)
        return f"{width}x{height}+{x}+{y}"

    def __init__(self, master, message="Memuat komponen…", video_path="DATA/images/splash.mp4"):
        super().__init__(master)
        self.overrideredirect(True)
        self.attributes("-topmost", True)

        # DON'T use self._w / self._h (Tk uses _w internally!)
        self._vw, self._vh = 800, 400

        # use the instance helper (not LoaderSplash.CenterWindowToDisplay(...))
        self.geometry(self._center_geometry(self._vw, self._vh, self._get_window_scaling()))
        self.deiconify()
        self.configure(fg_color="#061324")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- video background holder ---
        self._video_label = customtkinter.CTkLabel(self, text=None, fg_color="transparent")
        self._video_label.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._video_img = CTkImage(light_image=Image.new("RGB", (self._vw, self._vh)),
                                   size=(self._vw, self._vh))
        self._video_label.configure(image=self._video_img)

        self._video_path = video_path
        self._cap = None
        self._video_after_id = None
        self._video_running = False
        self.bind("<Configure>", self._on_resize)

        # Foreground widgets (text + progress bar) — appear on top
        self.label2 = customtkinter.CTkLabel(
            self._video_label, text="OSerial 2025 SP1",
            font=customtkinter.CTkFont("Bahnschrift SemiLight", 16, "bold"),
            text_color="white",
            bg_color="transparent",   # << add
            fg_color="transparent",   # << add
        )
        self.label2.place(relx=0.86, rely=0.1, anchor="center")

        self.label = customtkinter.CTkLabel(
            self._video_label, text=message,
            font=customtkinter.CTkFont("Bahnschrift SemiLight", 16, "bold"),
            text_color="white",
            bg_color="transparent",   # << add
            fg_color="transparent",   # << add
        )
        self.label.place(relx=0.85, rely=0.15, anchor="center")

        self.pb = customtkinter.CTkProgressBar(self._video_label, mode="indeterminate", bg_color="transparent")
        self.pb.place(relx=0.815, rely=0.2, anchor="center")
        self.pb.start()

        # Optional: prevent closing via [X]
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        # Start the video loop (if the file exists)
        self._start_video()

    def _on_resize(self, _evt):
        # track target size and resize the CTkImage container
        try:
            w = max(50, self.winfo_width())
            h = max(50, self.winfo_height())
            self._vw, self._vh = w, h
            self._video_img.configure(size=(self._vw, self._vh))
        except Exception:
            pass

    def _start_video(self):
        if not self._video_path or not os.path.isfile(self._video_path):
            # No video file – keep a flat bg color
            return
        try:
            self._cap = cv2.VideoCapture(self._video_path)
            if not self._cap or not self._cap.isOpened():
                # Failed to open – bail gracefully
                self._cap = None
                return
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._video_delay = int(200 / fps) if fps and fps > 1 else 33  # ~30 FPS fallback
            self._video_running = True
            self._video_tick()  # kick off loop
        except Exception:
            self._cap = None

    def _video_tick(self):
        # Guard against race with close/destroy
        if not self._video_running or not self.winfo_exists():
            return
        ret, frame = (False, None)
        try:
            ret, frame = self._cap.read()
            if not ret:
                # Loop from start
                try:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                except Exception:
                    ret = False
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = self._cover_resize_crop(rgb, self._vw, self._vh)
                pil_img = Image.fromarray(rgb)
                self._video_img.configure(light_image=pil_img, size=(self._vw, self._vh))
        except Exception:
            pass
        # Schedule next frame if still alive
        if self._video_running and self.winfo_exists():
            self._video_after_id = self.after(self._video_delay, self._video_tick)

    @staticmethod
    def _cover_resize_crop(rgb_img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """Scale to cover target box, then center-crop to exact size (like CSS object-fit: cover)."""
        h, w = rgb_img.shape[:2]
        scale = max(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x1 = max(0, (new_w - target_w) // 2)
        y1 = max(0, (new_h - target_h) // 2)
        return resized[y1:y1 + target_h, x1:x1 + target_w]

    def set_text(self, txt: str):
        self.label.configure(text=txt)

    def stop_and_close(self):
        # Stop progress first
        try:
            self.pb.stop()
        except Exception:
            pass
        # Stop the video loop safely
        self._video_running = False
        try:
            if self._video_after_id is not None:
                self.after_cancel(self._video_after_id)
                self._video_after_id = None
        except Exception:
            pass
        try:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        except Exception:
            pass
        # Destroy the splash
        try:
            self.destroy()
        except Exception:
            pass

# --- 3) Bootstrap: run one mainloop (the App), splash is a child window ---
def preload_and_launch():
    app = App()

    splash = LoaderSplash(
        app,
        "Menyiapkan 2 OCR",
        video_path=r"DATA\images\small-vecteezy_digital-plexus-wave-triangulation-shapes-futuristic_20937878_small.mp4",   # <-- put your MP4 here
    )

    def ui_set(msg: str):
        # Safe UI update from worker thread
        app.after(1000, lambda: splash.winfo_exists() and splash.set_text(msg))

    def worker():
        global onnxdet, onnxrec, model, model2, model3
        try:
            ui_set("Memuat modl-OCR")
            onnxdet, onnxrec = ocrmodels.load_ocr()
            time.sleep(0.5)

            ui_set("Memuat mod-YOLO")
            model, model2 = ocrmodels.load_yolo()
            time.sleep(0.5)

            ui_set("Memuat RF-DETR…")
            model3 = ocrmodels.Load_rfdetr()
            time.sleep(0.5)

            ui_set("Inisialisasi Sistem")

        except Exception as e:
            ui_set(f"Gagal memuat: {e}")
            time.sleep(1.0)
        finally:
            def finalize():
                if splash.winfo_exists():
                    splash.stop_and_close()
                # Show the main window only now
                app.deiconify()
                try:
                    app.attributes("-alpha", 1.0)
                except Exception:
                    pass
                app.lift()
                app.focus_force()
            app.after(5000, finalize)

    threading.Thread(target=worker, daemon=True).start()
    app.mainloop()

# --- 4) Entry point ---
if __name__ == "__main__":
    preload_and_launch()