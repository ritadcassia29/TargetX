from pymavlink import mavutil
import time
import pymap3d.enu as enu
import math

import sys

import threading

import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
from pathlib import Path

import drone_control as drone_control

# from typing import List

from typing import Tuple, List

from ultralytics import YOLO
from utils.geodetic import Geodetic
from utils.geometry import Geometry
import numpy as np
import utm
import cv2
from utils.timestamped_buffer import TimestampedBuffer

import pandas as pd
from sklearn.cluster import DBSCAN

from datetime import datetime

import json

# posicao = None
# posicao_timestamp = None

# atitude = None
# atitude_timestamp = None

# posicao_lock = threading.Lock()
# atitude_lock = threading.Lock()

atitude_buffer = TimestampedBuffer(maxlen=10)
posicao_buffer = TimestampedBuffer(maxlen=10)

frame_global = None
frame_lock = threading.Lock()

add_deteccao = True
add_deteccao_lock = threading.Lock()

lat_gabarito = -22.9452397
long_gabarito = -43.1567356
alt_gabarito = 0
# relativa = 4      # drone sobre alvo
# nivel_mar = 8     # drone sobre alvo
# elipsoide = 1.283 # drone sobre alvo

deteccoes_latlon = []  # lista de dicts: {"ts": float, "lat": float, "lon": float}
deteccoes_lock = threading.Lock()

east_gabarito, north_gabarito, zn_gab, zl_gab = utm.from_latlon(lat_gabarito, long_gabarito)

model = YOLO("YOLO_Alessandra.pt")
geometry = Geometry()
geodetic = Geodetic(None, None, None)
K = np.array([[635.2751011647164, 0, 664.3862127816539],
              [0, 636.6952787846122, 347.9655674631052],
              [0, 0, 1]])

# Coeficientes de distorção
dist_coeff_fisheye = (0.022909464284721036, -0.1485221115561539, 0.19800195776761587, -0.10861853479661913)

K_inv = geometry.inv_K(K)

# PID DE CONTROLE VISUAL

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

CENTRO_X = CAMERA_WIDTH / 2.0
CENTRO_Y = CAMERA_HEIGHT / 2.0

PIXEL_DEAD_ZONE = 30.0

KP_X = 0.0018
KI_X = 0.0
KD_X = 0.0

KP_Y = 0.0018
KI_Y = 0.0
KD_Y = 0.0

VELOCIDADE_MAXIMA = 1.5  # m/s

class PID:
    def __init__(self, kp, ki, kd, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.output_limit = output_limit

        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def update(self, error):
        agora = time.monotonic()

        if self.previous_time is None:
            self.previous_time = agora
            self.previous_error = error

            return self.kp * error

        dt = agora - self.previous_time

        if dt <= 0:
            return 0.0

        # Termo proporcional
        P = self.kp * error

        # Termo integral
        self.integral += error * dt
        I = self.ki * self.integral

        # Termo derivativo
        D = self.kd * (error - self.previous_error) / dt

        saida = P + I + D

        # Limitação da saída
        if self.output_limit is not None:
            saida = max(
                -self.output_limit,
                min(saida, self.output_limit)
            )

        self.previous_error = error
        self.previous_time = agora

        return saida

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

pid_x = PID(
    kp=KP_X,
    ki=KI_X,
    kd=KD_X,
    output_limit=VELOCIDADE_MAXIMA
)

pid_y = PID(
    kp=KP_Y,
    ki=KI_Y,
    kd=KD_Y,
    output_limit=VELOCIDADE_MAXIMA
)

def adicionar_deteccao_latlon(lat: float, lon: float, lat_drone: float, lon_drone: float, ts: float | None = None) -> None:
    """
    Adiciona uma detecção (lat/lon) na lista global de forma thread-safe.
    """
    if ts is None:
        ts = time.time()

    dist_drone_alvo = 0

    alvo = (lat,lon,0)
    drone = (lat_drone, lon_drone,0)

    dist_drone_alvo = distance_between_geo_points(alvo, drone)

    with deteccoes_lock:
        deteccoes_latlon.append({"ts": float(ts), "lat": float(lat), "lon": float(lon), "lat_drone": float(lat_drone), "lon_drone": float(lon_drone), "dist_drone_alvo": float(dist_drone_alvo)})

def salvar_media_cluster_deteccoes(lista_ordenada_truncada: list[dict]):
    ############ Faz a média das lat e long ############
    try:
        latitudes = [d["lat"] for d in lista_ordenada_truncada]
        longitudes = [d["lon"] for d in lista_ordenada_truncada]

        lat_media = sum(latitudes) / len(latitudes)
        lon_media = sum(longitudes) / len(longitudes)

        print(f"--> Alvo média: lat={lat_media}, lon={lon_media}")

        ts_humano = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        payload = {
                "alvo_centro": {
                    "timestamp": ts_humano,
                    "lat": float(lat_media),
                    "lon": float(lon_media)
                }
                }

        caminho_saida = "alvo_media.json"
        path = Path(caminho_saida)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    
    except Exception as e:
        print(f"[ERRO] Falha inesperada ao calcular a média: {e}")
    ############################################################


    ############ Cluster ############
    try:

        df_res, center_latlon, center_utm, mean_dist_drone_alvo = filtrar_outliers_dbscan_latlon(lista_ordenada_truncada, eps_m=3.0, min_samples=8)

        if center_latlon is not None:
            (lat_cluster, lon_cluster) = center_latlon

        else: 
            lat_cluster = 0 
            lon_cluster = 0


        print(f"--> Alvo cluster: lat={lat_cluster}, lon={lon_cluster}")

        ts_humano = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        payload = {
                "alvo_centro": {
                    "timestamp": ts_humano,
                    "lat": float(lat_cluster),
                    "lon": float(lon_cluster)
                }
                }

        caminho_saida = "alvo_cluster.json"
        path = Path(caminho_saida)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    
    except Exception as e:
        print(f"[ERRO] Falha inesperada Cluster {e}")
    ############################################################

def get_cluster_deteccoes(lista_ordenada_truncada: list[dict]):
    
    ############ Cluster ############
    try:

        df_res, center_latlon, center_utm, mean_dist_drone_alvo = filtrar_outliers_dbscan_latlon(lista_ordenada_truncada, eps_m=3.0, min_samples=8)

        if center_latlon is not None:
            (lat_cluster, lon_cluster) = center_latlon
            return lat_cluster, lon_cluster, mean_dist_drone_alvo

        else: 
            return None


        print(f"--> Alvo cluster: lat={lat_cluster}, lon={lon_cluster}")

    except Exception as e:
        print(f"[ERRO] Falha inesperada Cluster {e}")
        return None
    ############################################################


def snapshot_deteccoes_latlon(n_menores: int = 20) -> list[dict]:
    """
    Retorna uma cópia (snapshot) da lista global, thread-safe,
    ordenada por dist_drone_alvo (crescente) e limitada aos N menores.
    """
    if n_menores <= 0:
        return []

    with deteccoes_lock:
        snap = list(deteccoes_latlon)

    if len(snap) == 0:
        return []
    
    # Ordena fora do lock (evita travar writers por muito tempo)
    snap.sort(key=lambda d: d.get("dist_drone_alvo", float("inf")))

    return snap[:n_menores]

# def snapshot_deteccoes_latlon() -> list[dict]:
#     """
#     Retorna uma cópia (snapshot) da lista global, thread-safe.
#     """
#     with deteccoes_lock:
#         return list(deteccoes_latlon)

def salvar_deteccoes_json(caminho_arquivo: str, n_menores: int = 20):
    """
    Salva as N menores detecções em um arquivo JSON.
    """
    # with deteccoes_lock:
    #     snap = list(deteccoes_latlon)

    snap = snapshot_deteccoes_latlon(n_menores=n_menores)

    if not snap:
        print("Nenhuma detecção para salvar.")
        return

    path = Path(caminho_arquivo)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)

    print(f"{len(snap)} detecções salvas em {caminho_arquivo}")   

def filtrar_outliers_dbscan_latlon(
    lista_latlon: list[dict],
    eps_m: float = 3.0,
    min_samples: int = 8
):
    """
    Recebe lista de dicts com estrutura:
      {"ts", "lat", "lon", "lat_drone", "lon_drone", "dist_drone_alvo"}

    Faz clusterização APENAS de (lat, lon) do ALVO:
      lat/lon -> UTM (metros) -> DBSCAN
      - remove ruídos (label=-1)
      - mantém apenas o MAIOR cluster
      - calcula centroide em UTM do cluster principal
      - converte centroide de volta para lat/lon

    Retorna:
      (df_result, center_latlon, center_utm)

    df_result inclui colunas:
      ts, lat, lon, lat_drone, lon_drone, dist_drone_alvo,
      alvo_utm_e, alvo_utm_n, zn, zl, alvo_label, is_outlier, is_main_cluster
    """
    # 0) lista vazia
    if not lista_latlon:
        df = pd.DataFrame(columns=[
            "ts","lat","lon","lat_drone","lon_drone","dist_drone_alvo",
            "alvo_utm_e","alvo_utm_n","zn","zl",
            "alvo_label","is_outlier","is_main_cluster"
        ])
        return df, None, None

    df = pd.DataFrame(lista_latlon).copy()

    # 1) garante tipos numéricos
    for col in ["lat","lon","lat_drone","lon_drone","dist_drone_alvo"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # remove linhas sem lat/lon válidos
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if len(df) == 0:
        df["alvo_label"] = []
        df["is_outlier"] = []
        df["is_main_cluster"] = []
        return df, None, None

    # 2) converte (lat,lon) do ALVO para UTM
    e_list, n_list, zn_list, zl_list = [], [], [], []
    for lat, lon in zip(df["lat"].astype(float), df["lon"].astype(float)):
        e, n, zn, zl = utm.from_latlon(lat, lon)
        e_list.append(e)
        n_list.append(n)
        zn_list.append(zn)
        zl_list.append(zl)

    df["alvo_utm_e"] = e_list
    df["alvo_utm_n"] = n_list
    df["zn"] = zn_list
    df["zl"] = zl_list

    pts = df[["alvo_utm_e", "alvo_utm_n"]].to_numpy(dtype=float)
    if len(pts) == 0:
        df["alvo_label"] = []
        df["is_outlier"] = []
        df["is_main_cluster"] = []
        return df, None, None

    # 3) DBSCAN em UTM
    db = DBSCAN(eps=eps_m, min_samples=min_samples)
    labels = db.fit_predict(pts)  # -1 = outlier

    df["alvo_label"] = labels
    df["is_outlier"] = df["alvo_label"] == -1

    # 4) acha maior cluster (ignorando -1)
    valid_labels = labels[labels != -1]
    if len(valid_labels) == 0:
        df["is_main_cluster"] = False
        return df, None, None

    unique, counts = np.unique(valid_labels, return_counts=True)
    main_label = unique[np.argmax(counts)]
    df["is_main_cluster"] = df["alvo_label"] == main_label

    # 5) centroide UTM do cluster principal
    main_pts = df.loc[df["is_main_cluster"], ["alvo_utm_e", "alvo_utm_n"]].to_numpy(dtype=float)
    center_utm = main_pts.mean(axis=0)  # (e_c, n_c)

    # 6) converte centroide UTM -> lat/lon usando a zona do 1º ponto do cluster principal
    first_main = df.loc[df["is_main_cluster"]].iloc[0]
    zn = int(first_main["zn"])
    zl = str(first_main["zl"])

    center_lat, center_lon = utm.to_latlon(
        float(center_utm[0]),
        float(center_utm[1]),
        zone_number=zn,
        zone_letter=zl
    )

    # return df, (float(center_lat), float(center_lon)), (float(center_utm[0]), float(center_utm[1]))

    # 7) média da dist_drone_alvo no cluster principal
    mean_dist_drone_alvo = (
        df.loc[df["is_main_cluster"], "dist_drone_alvo"]
        .dropna()
        .mean()
    )

    if np.isnan(mean_dist_drone_alvo):
        mean_dist_drone_alvo = None
    else:
        mean_dist_drone_alvo = float(mean_dist_drone_alvo)

    return (
        df,
        (float(center_lat), float(center_lon)),
        (float(center_utm[0]), float(center_utm[1])),
        mean_dist_drone_alvo
    )

def salvamento_automatico_deteccoes_thread():
    while True:
        time.sleep(10)
        try:
            lista_ordenada_truncada = snapshot_deteccoes_latlon(n_menores = 20)
            salvar_media_cluster_deteccoes(lista_ordenada_truncada)
        except Exception as e:
            print(f"[ERRO] Falha inesperada no salvamento automático: {e}")

def set_velocidade_drones(drones, velocidade):
    for drone in drones:
        ok = drone_control.set_param_async(drone, "WPNAV_SPEED", velocidade, tentativas=3)
        if not ok:
            print(f"Drone {drone.target_system}: não confirmou WPNAV_SPEED, seguindo mesmo assim...")

def rota_automatica(drones, rota, altitudes):

    # delay(2)
    global add_deteccao
    global add_deteccao_lock

    set_velocidade_drones(drones, 400)
    tamanho_rota = len(rota)

    for i in range(len(rota) - 1):

        if i == 1:
            set_velocidade_drones(drones, 250)
        
        atual = rota[i]
        proximo = rota[i+1]

        raio = 8.0

        erro_aproximacao_waypoint = 7

        # lider, esq, dir = gerar_triangulo_formacao(atual, proximo, raio)
        lider, esq = seguidor_atras_do_lider_formacao(atual, proximo, raio)

        print(f"Waypoint {i+1} de {tamanho_rota}:")
        # print(f"   Líder (frente)   → {lider[0]:.8f}, {lider[1]:.8f}")
        # print(f"   Esquerda (trás)  → {esq[0]:.8f}, {esq[1]:.8f}")


        th_goto_0 = threading.Thread(target=goto_location_and_wait_fpv, args=(drones[0], lider, altitudes[0], erro_aproximacao_waypoint))

        # th_goto_1 = threading.Thread(target=drone_control.goto_location_and_wait, args=(drones[1], esq, altitudes[1], erro_aproximacao_waypoint))

        # th_goto_0 = threading.Thread(target=goto_location, args=(drones[0], lider , altitudes[0]))
        # th_goto_1 = threading.Thread(target=goto_location, args=(drones[1], esq, altitudes[1]))
        # th_goto_2 = threading.Thread(target=goto_location, args=(drones[2], dir, altitude_drone2))

        th_goto_0.start()
        # th_goto_1.start()
        # th_goto_2.start()

        th_goto_0.join() 
        # th_goto_1.join()
        # th_goto_2.join()

        print("Chegou")

        with deteccoes_lock:
            deteccoes_realizadas = len(deteccoes_latlon)
            if deteccoes_realizadas >=20:
                break

    # salvar_deteccoes_json("deteccoes_filtradas.json", n_menores=20)
    eps_mean_dist_drone_alvo = 3.0

    while True:

        # Pegar as detecções realizadas
        lista_ordenada_truncada = snapshot_deteccoes_latlon(n_menores = 20)
        lat_cluster, lon_cluster, mean_dist_drone_alvo = get_cluster_deteccoes(lista_ordenada_truncada)

        #mandar ir pra latlongcluster
        goto_location_and_wait_fpv(drones[0], (lat_cluster, lon_cluster), altitudes[0], erro_aproximacao_waypoint, wait=False)

        if mean_dist_drone_alvo < eps_mean_dist_drone_alvo:
            break

        time.sleep(0.5)


    # salvar_media_cluster_deteccoes(lista_ordenada_truncada)

    set_velocidade_drones(drones, 400)


    
    rtl_threads = []


    print("Configurando parametros de retorno dos drones...")

    for drone, altitude  in zip(drones, altitudes):

        # Modificar parametro do drone para apontar para o home
        ok = drone_control.set_param(drone, "WP_YAW_BEHAVIOR", 1)  # 1 = Face next WP incluindo HOME no RTL
        if not ok:
            print(f"Drone {drone.target_system}: não confirmou WP_YAW_BEHAVIOR, seguindo mesmo assim...")


        altitude_cm = int(altitude * 100.0)
        ok = drone_control.set_param(drone, "RTL_ALT", altitude_cm)  # 1 = Face next WP incluindo HOME no RTL
        if not ok:
            print(f"Drone {drone.target_system}: não confirmou RTL_ALT, seguindo mesmo assim...")


    # 4) RTL + LAND
    print("Retornando com os drones")
    # return_and_land(drone, drone_id)
    for drone in drones:
        # set_flight_mode(drone, "RTL")
        th = threading.Thread(target=set_flight_mode, args=(drone, "RTL"))
        rtl_threads.append(th)
    
    for th in rtl_threads:
        th.start()
    
    for th in rtl_threads:
        th.join()

    while True:
        with add_deteccao_lock:
            add_deteccao = False
            print("Desativado salvamento de pontos da deteccao")
            break
        time.sleep(0.2)

    print("Missão concluída.")

def print_on_pixel(image, label, x, y, cor):
    font_scale = 1  # Tamanho da fonte
    font_thickness = 2  # Espessura da fonte
    font = cv2.FONT_HERSHEY_SIMPLEX  # Fonte
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    image_height, image_width, image_channels = image.shape
    text_x = x  # Alinhar à esquerda do retângulo
    text_y = y - baseline - 5  # Acima do retângulo (-5 para espaçamento)

    if text_y < 0:
        text_y = text_height + 5
    if text_x + text_width > image_width:  # Ultrapassa a borda direita
        text_x = image_width - text_width - 5  # Ajustar para a borda direita
    if text_x < 0:  # Ultrapassa a borda esquerda
        text_x = 5  # Ajustar para a borda esquerda


    cv2.putText(image, label, (text_x, text_y), font, font_scale, cor, font_thickness)

def mav_position_copy(msg):
    if msg is None:
        return None
    
    msg_copy = mavutil.mavlink.MAVLink_global_position_int_message(
        time_boot_ms=msg.time_boot_ms,
        lat=msg.lat,
        lon=msg.lon,
        alt=msg.alt,
        relative_alt=msg.relative_alt,
        vx=msg.vx,
        vy=msg.vy,
        vz=msg.vz,
        hdg=msg.hdg,
    )

    return msg_copy

def mav_atitude_copy(att_msg):
    if att_msg is None:
        return None
    
    att_copy = mavutil.mavlink.MAVLink_attitude_message(
        time_boot_ms=att_msg.time_boot_ms,
        roll=att_msg.roll,
        pitch=att_msg.pitch,
        yaw=att_msg.yaw,
        rollspeed=att_msg.rollspeed,
        pitchspeed=att_msg.pitchspeed,
        yawspeed=att_msg.yawspeed,
    )

    return att_copy

def camera_thread(drone_controle_visual=None):
    global frame_global
    global posicao_buffer
    global atitude_buffer
    global add_deteccao
    global add_deteccao_lock

    print("Carregando câmera...")
    cap = cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera carregada")

    if not cap.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        return
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar frame.")
            break

        # with posicao_lock:
        # posicao_frame = mav_position_copy(posicao)
        posicao_frame = posicao_buffer.get_closest(time.time())

        # with atitude_lock:
        # atitude_frame = mav_atitude_copy(atitude)
        atitude_frame = atitude_buffer.get_closest(time.time())

        detect = detecta_alvo(frame)

        if detect is not None:
            x, y, conf, x1, y1, x2, y2, label = detect
            lat = lon = erro_dist = 0

            if conf >= 0.6:  # FILTRO DE CONFIANÇA
                
                # ========================================================
                # ERRO VISUAL
                # ========================================================

                erro_x = CENTRO_X - float(x)
                erro_y = CENTRO_Y - float(y)

                # ========================================================
                # DEAD ZONE
                # ========================================================

                if abs(erro_x) < PIXEL_DEAD_ZONE:
                    erro_x = 0.0

                if abs(erro_y) < PIXEL_DEAD_ZONE:
                    erro_y = 0.0

                # ========================================================
                # PID
                # ========================================================

                velocidade_x = pid_x.update(erro_x)
                velocidade_y = pid_y.update(erro_y)

                print(
                    f"Alvo: ({x:.1f}, {y:.1f}) | "
                    f"Erro: ({erro_x:.1f}, {erro_y:.1f}) | "
                    f"PID: ({velocidade_x:.2f}, {velocidade_y:.2f})"
                )

                # ========================================================
                # ENVIO DO COMANDO DE VELOCIDADE
                # ========================================================

                # Só executamos se tivermos o drone disponível.
                if drone_controle_visual is not None:

                    drone_control.enviar_velocidade_visual(
                        drone_controle_visual,
                        velocidade_x,
                        velocidade_y
                    )

                # ========================================================
                # SUA LOCALIZAÇÃO ORIGINAL CONTINUA AQUI
                # ========================================================

                if posicao_frame is not None and atitude_frame is not None:

                    lat, lon, east, north = localiza_alvo(
                        posicao_drone=posicao_frame,
                        atitude_drone=atitude_frame,
                        pixel_x=x,
                        pixel_y=y
                    )

                    erro_dist = np.sqrt(
                        (east - east_gabarito)**2 +
                        (north - north_gabarito)**2
                    )

                    lat_drone = posicao_frame.lat / 1e7
                    lon_drone = posicao_frame.lon / 1e7

                    with add_deteccao_lock:

                        if add_deteccao:
                            adicionar_deteccao_latlon(
                                lat,
                                lon,
                                lat_drone,
                                lon_drone
                            )

                    # f_csv = open("logs_deteccao.csv","a")
                    # # log = f"{time.time()}; alvo: {lat}, {lon}; erro: {erro_dist}; pixel: {x}, {y}, drone: {posicao_frame.lat / 1e7}, {posicao_frame.lon / 1e7}, {posicao_frame.relative_alt / 1000.0}"
                    # # print(log)
                    # log_csv = f"{time.time()};{lat};{lon};{erro_dist};{x};{y};{posicao_frame.lat / 1e7};{posicao_frame.lon / 1e7};{posicao_frame.relative_alt / 1000.0}\n"
                    # f_csv.write(log_csv)

                # Desenhar retângulo
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 0, 255),
                    2
                )

                # Texto com nome + confiança
                print_on_pixel(frame,f"{label} {conf:.2f}, LAT: {lat:.7f}, LONG: {lon:.7f}", int(x1), int(y1) - 10,(0, 0, 255))
                # cv2.putText(
                #     frame,
                #     f"{label} {conf:.2f}, LAT: {lat:.7f}, LONG: {lon:.7f}, erro: {erro_dist:.2f}",
                #     (int(x1), int(y1) - 10),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     0.7,
                #     (0, 0, 255),
                #     2
                # )

        # Exibe a imagem anotada
        with frame_lock:
            frame_global = frame.copy()

        
    return
    
def rad2deg(rad):
    return rad * 180.0 / math.pi

def detecta_alvo(frame):
    
    results = model(frame, verbose=False)
    
    for box in results[0].boxes:
        conf = float(box.conf[0])           # confiança
        cls_id = int(box.cls[0])            # ID da classe
        label = model.names[cls_id]         # nome da classe
        x1, y1, x2, y2 = box.xyxy[0]         # coordenadas
        x = (x1 + x2)/2
        y = (y1 + y2)/2
        return (x, y, conf, x1, y1, x2, y2, label)
    
    return None

def localiza_alvo(posicao_drone, atitude_drone, pixel_x, pixel_y):

    if pixel_x is None or pixel_y is None:
        return None

    yaw_graus = rad2deg(atitude_drone.yaw)
    pitch_graus = rad2deg(atitude_drone.pitch) - 90
    roll_graus = rad2deg(atitude_drone.roll)

    R_drone = geometry.yaw_pitch_roll_to_rotation_matrix(yaw_graus, pitch_graus, roll_graus)
    R_drone_T = np.transpose(R_drone)
    R = geometry.droneToCameraR @ R_drone_T @ geometry.mundoToDroneR

    lat = posicao_drone.lat / 1e7          # latitude * 1e7
    lon = posicao_drone.lon / 1e7         # longitude * 1e7
    alt = posicao_drone.alt / 1000.0 
    rel_alt = posicao_drone.relative_alt / 1000.0

    east, north, zn, zl = utm.from_latlon(lat, lon)

    utm_position = np.array([[east], [north], [rel_alt]])

    intersection = geodetic.get_intersection_from_pixel((pixel_x, pixel_y), K_inv, R_drone, utm_position, dist_coeff=dist_coeff_fisheye, fisheye=True, K=K)
    detection_east = intersection[0,0]
    detection_north = intersection[1,0]

    detect_lat, detect_lon = utm.to_latlon(detection_east, detection_north, zone_number=zn, zone_letter=zl)
    print(f"detectado lat: {detect_lat}, long: {detect_lon}")

    return (detect_lat, detect_lon, east, north)

def get_position(drone):    

    msg = drone.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)

    if msg is not None:
        # msg_json = json.dumps(msg.to_dict())
        # return msg_json
        return msg
    
    return None

def get_attitude(drone):  

    msg = drone.recv_match(type='ATTITUDE', blocking=True, timeout=1)

    if msg is not None:
        # msg_json = json.dumps(msg.to_dict())
        # return msg_json
        return msg
    
    return None

def posicao_thread(drone):
    global posicao_buffer

    while True:

        p = mav_position_copy(get_position(drone))
        ts = time.time()

        if p is not None:
            posicao_buffer.add(p, ts)

        #ATENÇAO: Não colocar delay aqui. Pois deixa a fila do buffer da serial cheia
        # time.sleep(0.1)

def atitude_thread(drone):
    global atitude_buffer

    while True:

        a = mav_atitude_copy(get_attitude(drone))
        ts = time.time()

        if a is not None:
            atitude_buffer.add(a, ts)

        #ATENÇAO: Não colocar delay aqui. Pois deixa a fila do buffer da serial cheia
        # time.sleep(0.1)

def telemetria_thread(drone):
    global atitude_buffer
    global posicao_buffer

    while True:
        msg = drone.recv_match(blocking=True, timeout=1.0)  # lê QUALQUER msg
        if msg is None:
            continue

        ts = time.time()
        t = msg.get_type()

        if t == "GLOBAL_POSITION_INT":
            posicao_buffer.add(mav_position_copy(msg), ts)
        elif t == "ATTITUDE":
            atitude_buffer.add(mav_atitude_copy(msg), ts)

def ler_waypoints_qgc_para_lista(
    caminho_arquivo: str,
    incluir_home: bool = False,
    com_altitude: bool = False
) -> List[List[float]]:
    """
    Lê arquivo .waypoints do QGroundControl (QGC WPL 110/120)
    e retorna uma lista simples de listas com as coordenadas.

    Parâmetros:
        caminho_arquivo: caminho do arquivo
        incluir_home: True para incluir o ponto Home (índice 0)
        com_altitude: True para retornar [lat, lon, alt], False para [lat, lon]

    Retorna:
        [[lat, lon], ...] ou [[lat, lon, alt], ...]
    """
    waypoints = []

    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    # Pula cabeçalho
    if not linhas or "QGC WPL" not in linhas[0]:
        raise ValueError("Arquivo não é um arquivo de missão QGC válido.")

    for linha in linhas[1:]:  # começa depois do cabeçalho
        linha = linha.strip()
        if not linha or linha.startswith('#'):
            continue

        colunas = linha.split('\t')  # separador é TAB
        if len(colunas) < 12:
            continue

        try:
            idx = int(colunas[0])
            lat = float(colunas[8])
            lon = float(colunas[9])
            alt = float(colunas[10])
            flag = int(colunas[11])
        except:
            continue

        # Só inclui waypoints válidos
        # if flag == 1
        if flag != 1:
            continue

        # Pula o Home se não quiser incluir
        if idx == 0 and not incluir_home:
            continue

        if com_altitude:
            waypoints.append([lat, lon, alt])
        else:
            waypoints.append([lat, lon])

    print(f"Carregados {len(waypoints)} waypoints do arquivo '{caminho_arquivo}'")
    return waypoints

def lancar_carga(drones):
    """
    Aciona um servo na porta AUX3 da Pixhawk 6C em modo reverso.
    
    - ativado=True → envia PWM baixo (ativado)
    - ativado=False → envia PWM alto (desativado)
    - pwm_ativado=900 → valor de PWM para "ligado"
    - pwm_desativado=1900 → valor de PWM para "desligado"
    """

    # Porta AUX3 = canal 10 (1-8 = MAIN, 9-14 = AUX)
    canal = 11
    pwm_baixo = 1000
    pwm_alta = 2000
    tentativas = 3

    for i in range(0,tentativas):

        print(f"LAMÇAMENTO DE CARGA: Tentativa {i +1}")
    
        for drone in drones:
            drone.mav.command_long_send(
                drone.target_system,
                drone.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                canal,     # Número do servo
                pwm_baixo,       # PWM a enviar
                0, 0, 0, 0, 0
            )

        time.sleep(3)

        for drone in drones:
            drone.mav.command_long_send(
                drone.target_system,
                drone.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                canal,     # Número do servo
                pwm_alta,       # PWM a enviar
                0, 0, 0, 0, 0
            )

        time.sleep(3)

def delay(total_time, bar_length=30):
    """
    Aguarda total_time segundos, exibindo uma barra de progresso de bar_length caracteres.
    
    :param total_time: duração total do delay em segundos.
    :param bar_length: número de caracteres da barra de progresso.
    """

    print(f"Delay de {total_time}")

    # Tempo entre cada atualização da barra
    update_interval = total_time / bar_length
    
    # Começa a barra vazia
    sys.stdout.write('\r[' + ' ' * bar_length + ']')
    sys.stdout.flush()
    
    # Volta o cursor para dentro dos colchetes
    sys.stdout.write('\r[')
    for i in range(bar_length):
        time.sleep(update_interval)
        sys.stdout.write('█')      # preenche um bloco
        sys.stdout.flush()
    sys.stdout.write(']\n')        # fecha a barra e pula linha

# https://mavlink.io/en/messages/common.htm

delay_comando = 0.5

def get_velocidade(drone, timeout=1.0):
    """
    Retorna um dicionário com velocidades horizontal, vertical e 3D em m/s.
    """
    msg = drone.recv_match(type=['GLOBAL_POSITION_INT'],
                           blocking=True, timeout=timeout)
    if msg is None:
        return 0,0,0

    elif msg.get_type() == 'GLOBAL_POSITION_INT':
        vx, vy, vz = msg.vx/100.0, msg.vy/100.0, msg.vz/100.0  # cm/s -> m/s
    
    return vx, vy, vz

# Função para obter localização atual usando pymavlink
def get_current_location(drone):
    
    id_drone = drone.target_system
    count = 0

    drone.mav.command_long_send(
        id_drone, drone.target_component,
        mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
        0, mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        0, 0, 0, 0, 0, 0
    )

    while count <4:

        count+=1

        msg = drone.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=2)
        
        if msg:
            # return {
            #     'lat': msg.lat / 1e7,
            #     'lon': msg.lon / 1e7,
            #     'alt': msg.relative_alt / 1000.0
            # }
            return msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0
        
    return None


def distance_between_geo_points(point1, point2):
    """
    Calcula a distância entre dois pontos geográficos (latitude, longitude, altitude)
    usando o sistema de coordenadas locais ENU (East‑North‑Up) via pymap3d.

    Parâmetros:
        lat1, lon1: latitude e longitude do ponto de referência (graus decimais)
        lat2, lon2: latitude e longitude do ponto alvo (graus decimais)
        alt1: altitude do ponto de referência (metros) — default 0.0
        alt2: altitude do ponto alvo (metros) — default 0.0

    Retorna:
        Distância em metros (float)
    """
    lat1, lon1, alt1 = point1
    lat2, lon2, alt2 = point2
    # Converte ponto alvo para coordenadas ENU, com origem no ponto 1
    e, n, u = enu.geodetic2enu(lat2, lon2, alt2, lat1, lon1, alt1)

    # Distância euclidiana no sistema ENU
    return math.sqrt(e**2 + n**2 + u**2)

# Função para conectar ao drone
def connect_drone_IP(connection_string):

    drone = mavutil.mavlink_connection(connection_string)
    drone.wait_heartbeat()

    print("Enxame conectado!")

    return drone

def connect_drone(port, baud):

    drone = mavutil.mavlink_connection(port, baud=baud)
    drone.wait_heartbeat()

    print("Conectado ao drone!")

    return drone

def connect_drone_sitl(host="127.0.0.1", port=5762):
    """
    Conecta ao SITL ArduCopter via TCP.
    host: IP do SITL (normalmente localhost)
    port: porta SERIAL0 do SITL (5760 por padrão)
    """
    connection_string = f"tcp:{host}:{port}"
    print(f"Conectando ao SITL em {connection_string}...")
    drone = mavutil.mavlink_connection(connection_string)
    drone.wait_heartbeat()

    print("Conectado ao SITL!")

    return drone


# Função para armar e decolar
def arm_and_takeoff(drone, altitude):
    
    id_drone = drone.target_system

    print(f"Drone {id_drone} - Armando e decolando a {altitude}m...")

    drone.mav.command_long_send(
        id_drone, drone.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)
    
    time.sleep(3)

    msg = drone.recv_match(type='STATUSTEXT', blocking=True, timeout=2)
    if msg:
        print(f"STATUSTEXT: {msg.text}")

    drone.mav.command_long_send(
        id_drone, drone.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude)
    
    msg = drone.recv_match(type='STATUSTEXT', blocking=True, timeout=2)
    if msg:
        print(f"STATUSTEXT: {msg.text}")

    
    time.sleep(delay_comando)

def is_armed(drone, timeout=1):
    """
    Verifica se o veículo `target_system` está armado, lendo HEARTBEATs
    e descartando mensagens de outros sistemas.

    master: conexão pymavlink (mavutil.mavlink_connection)
    target_system: sysid do veículo a checar
    timeout: tempo máximo em segundos para aguardar um heartbeat válido
    """
    id_drone = drone.target_system
    start = time.time()
    while True:
        # aguarda até timeout para qualquer HEARTBEAT
        msg = drone.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
        if msg is None:
            raise RuntimeError(f"Nenhum HEARTBEAT do sistema {id_drone} em {timeout}s")
        # filtra só do system que me interessa
        if msg.get_srcSystem() == id_drone:
            return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        # se ainda não chegou o heartbeat do target, verifica timeout geral
        if time.time() - start > timeout:
            raise RuntimeError(f"Nenhum HEARTBEAT do sistema {id_drone} em {timeout}s")

# Função para definir modo de voo
def set_flight_mode(drone, mode_name):

    id_drone = drone.target_system
    
    print(f"Drone {id_drone}: Alterando para modo {mode_name}...")

    while drone.flightmode != mode_name:

        drone.wait_heartbeat()

        mode_id = drone.mode_mapping()[mode_name]

        drone.mav.set_mode_send(
            id_drone,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)
        
        time.sleep(delay_comando)
    
    print(f"Drone {id_drone}: Modo alterado para {mode_name}")

def get_bateria_v(drone, timeout=2):
    """Imprime e retorna a tensão da bateria em volts."""

    msg = drone.recv_match(type='SYS_STATUS', blocking=True, timeout=timeout)

    if not msg or msg.voltage_battery == 65535:
        # print("Sem leitura de bateria.")
        return 0
    
    v = msg.voltage_battery / 1000.0

    return v
    

# Função para ir a uma localização
# def goto_location(drone, id_drone, lat, lon, alt):
#     print(f"Drone {id_drone} - Indo para lat: {lat}, lon: {lon}, alt: {alt}...")
#     drone.mav.set_position_target_global_int_send(
#         0, id_drone, drone.target_component,
#         mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
#         0b110111111000,  # Máscara para posição
#         int(lat * 10**7), int(lon * 10**7), alt,
#         0, 0, 0, 0, 0, 0, 0, 0)
#     time.sleep(5)

# Função para ir a uma localização mantendo a altitude atual
# https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html

def goto_location(drone, point, altitude):
    # Desempacota as coordenadas do ponto (lat, lon)
    lat, lon = point

    id_drone = drone.target_system

    vel_x = 0
    vel_y = 0
    vel_z = 0

    graus_rate = 0
    yaw_rate = graus_rate * 2 * math.pi / 360

    # Obtém a altitude atual do drone
    # current_altitude = get_altitude(drone)
    
    print(f"Drone {id_drone}: Indo para lat: {lat}, lon: {lon}, altitude: {altitude}m...")

    # Bitmask to indicate which fields should be ignored by the vehicle (see POSITION_TARGET_TYPEMASK enum)
    # bit1:PosX, bit2:PosY, bit3:PosZ, bit4:VelX, bit5:VelY, bit6:VelZ, bit7:AccX, bit8:AccY, bit9:AccZ, bit11:yaw, bit12:yaw rate
    # When providing Pos, Vel and/or Accel all 3 axis must be provided. At least one of Pos, Vel and Accel must be provided (e.g. providing Yaw or YawRate alone is not supported)
    # Use Position : 0b110111111000 / 0x0DF8 / 3576 (decimal)
    # Use Velocity : 0b110111000111 / 0x0DC7 / 3527 (decimal)
    # Use Acceleration : 0b110000111000 / 0x0C38 / 3128 (decimal)
    # Use Pos+Vel : 0b110111000000 / 0x0DC0 / 3520 (decimal)
    # Use Pos+Vel+Accel : 0b110000000000 / 0x0C00 / 3072 (decimal)
    # Use Yaw : 0b100111111111 / 0x09FF / 2559 (decimal)
    # Use Yaw Rate : 0b010111111111 / 0x05FF / 1535 (decimal)

    	
    # Valid options are:
    # MAV_FRAME_GLOBAL (0): alt is in meters above sea level
    # MAV_FRAME_GLOBAL_INT (5): alt is in meters above sea level
    # MAV_FRAME_GLOBAL_RELATIVE_ALT (3): alt is in meters above home
    # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT (6): alt is in meters above home
    # MAV_FRAME_GLOBAL_TERRAIN_ALT (10): alt is in meters above terrain
    # MAV_FRAME_GLOBAL_TERRAIN_ALT_INT (11): alt is in meters above terrain

    # Envia comando 1 vez
    drone.mav.set_position_target_global_int_send(
        0, id_drone, drone.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        #   0b110111000000,  #Posição + velocidade
        # 0b110111000000 
        0b110111111000,  # Máscara para posição
        int(lat * 10**7), 
        int(lon * 10**7), 
        altitude,
        vel_x, 
        vel_y, 
        vel_z, 
        0, 
        0, 
        0, 
        0, 
        yaw_rate)
    
    time.sleep(delay_comando)

    # Envia comando 2 vez
    drone.mav.set_position_target_global_int_send(
        0, id_drone, drone.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        #   0b110111000000,  #Posição + velocidade
        # 0b110111000000 
        0b110111111000,  # Máscara para posição
        int(lat * 10**7), 
        int(lon * 10**7), 
        altitude,
        vel_x, 
        vel_y, 
        vel_z, 
        0, 
        0, 
        0, 
        0, 
        yaw_rate)
    
 
def distancia_ate_old(drone, ponto):
    
    lat, lon = ponto

    tentativas = 3
    i=0

    while i<tentativas:

        i=i+1

        msg = drone.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        
        if msg is not None:

            if msg.get_type() == 'GLOBAL_POSITION_INT':
                # Posição absoluta em milímetros e velocidade em cm/s
                current_lat = msg.lat / 1e7          # graus
                current_long = msg.lon / 1e7          # graus
                alt = msg.relative_alt / 1000.0  # metros (relativo ao home)
                vx = msg.vx / 100.0          # m/s
                vy = msg.vy / 100.0          # m/s
                vz = msg.vz / 100.0          # m/s
                hdg = msg.hdg / 100.0        # graus (0-360)

                # print(f"Pos: {lat:.7f}, {lon:.7f} | Alt: {alt:.2f}m | Vel: ({vx:.1f}, {vy:.1f}, {vz:.1f}) m/s | Hdg: {hdg:.1f}°")

                current_point = current_lat, current_long, 0
                dest_point = lat, lon, 0

                distance = distance_between_geo_points(current_point, dest_point)
                # voltagem_bateria = get_bateria_v(drone)

                # print(f"Distancia ate waypoint: {distance}")

                return distance

    return None

                
def wait(drone, ponto):

    lat, lon = ponto

    i=0

    while i<10:

        i=i+1

        msg = drone.recv_match(blocking=True, timeout=0.5)  # bloqueia no máximo 1s
        
        if msg is not None:

            if msg.get_type() == 'GLOBAL_POSITION_INT':
                # Posição absoluta em milímetros e velocidade em cm/s
                current_lat = msg.lat / 1e7          # graus
                current_long = msg.lon / 1e7          # graus
                alt = msg.relative_alt / 1000.0  # metros (relativo ao home)
                vx = msg.vx / 100.0          # m/s
                vy = msg.vy / 100.0          # m/s
                vz = msg.vz / 100.0          # m/s
                hdg = msg.hdg / 100.0        # graus (0-360)

                print(f"Pos: {lat:.7f}, {lon:.7f} | Alt: {alt:.2f}m | Vel: ({vx:.1f}, {vy:.1f}, {vz:.1f}) m/s | Hdg: {hdg:.1f}°")

                current_point = current_lat, current_long, 0
                dest_point = lat, lon, 0
                distance = distance_between_geo_points(current_point, dest_point)
                # voltagem_bateria = get_bateria_v(drone)

                print(f"Distancia ate waypoint: {distance}")
                # print(f"Drone {id_drone}: Altura: {current_alt}; Lat: {current_lat}; Lon: {current_long}")
                # print(f"Drone {id_drone}: Tensão: {voltagem_bateria:.2f} V")
                # print("----")

                if distance < 2.5:
                    print(f" Chegou ao destino")
                    break
     

    


def wait_goto_location(drone, ponto):
    
    lat, lon = ponto
    while True:
        id_drone = drone.target_system
        current_lat, current_long, current_alt = get_current_location(drone)
        current_point = current_lat, current_long, 0
        dest_point = lat, lon, 0
        distance = distance_between_geo_points(current_point, dest_point)
        voltagem_bateria = get_bateria_v(drone)

        print(f"Drone {id_drone}: Distancia ate waypoint: {distance}")
        print(f"Drone {id_drone}: Altura: {current_alt}; Lat: {current_lat}; Lon: {current_long}")
        print(f"Drone {id_drone}: Tensão: {voltagem_bateria:.2f} V")
        print("----")

        if distance < 2.5:
            print(f"Drone {id_drone}: Chegou ao destino")
            break

def wait_drones_goto_location(drones, ponto):

    lat, lon = ponto
    ok_dict = {}

    while True:
        for drone in drones:

            id_drone = drone.target_system
            current_lat, current_long, current_alt = get_current_location(drone)
            current_point = current_lat, current_long, 0
            dest_point = lat, lon, 0
            distance = distance_between_geo_points(current_point, dest_point)
            voltagem_bateria = get_bateria_v(drone)

            print(f"Drone {id_drone}: Distancia ate waypoint: {distance}")
            print(f"Drone {id_drone}: Altura: {current_alt}; Lat: {current_lat}; Lon: {current_long}")
            print(f"Drone {id_drone}: Tensão: {voltagem_bateria:.2f} V")
            print("----")

            if distance < 2.5:
                print(f"Drone {id_drone}: Chegou ao destino")
                ok_dict[f"{id_drone}"] = True

        print(ok_dict)

        if len(ok_dict) == len(drones):
            break

def set_altitude_relativa(drone, altitude_m):
    """
    Envia um SET_POSITION_TARGET_LOCAL_NED para manter o drone na altitude desejada.
    altitude_m: metros acima do home (positivo)
    """
    # Em NED, Z positivo é pra baixo → z = -altitude_m
    ned_z = -altitude_m

    print(f"Altitude relativa para {altitude_m}")

    duracao_s = 15

    inicio = time.time()

    while (time.time() - inicio) < duracao_s:

        print(f"[{time.time() - inicio:.1f}s] Ajustando altitude para {altitude_m} m")
        
        drone.mav.set_position_target_local_ned_send(
            0,                            # time_boot_ms (0 = ignora)
            drone.target_system,          # target_system
            drone.target_component,       # target_component
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b110111111000,               # type_mask (usar somente PosX/PosY/PosZ)
            0, 0, ned_z,                  # x, y, z
            0, 0, 0,                      # vx, vy, vz
            0, 0, 0,                      # afx, afy, afz
            0, 0                          # yaw, yaw_rate
        )
        
        # espera 1 segundo antes do próximo envio
        time.sleep(1)

def set_altitude_fabio(drone, target_altitude):
    """
    Define altitude relativa ao home, capturando lat/lon atuais automaticamente.
    - target_altitude: Altitude desejada em metros acima do home.
    Usa SET_POSITION_TARGET_GLOBAL_INT com frame 6.
    """
    # Captura posição atual (lat, lon, alt atual para log)
    # msg = drone.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=5)
    
    # if not msg:
    #     print("Erro: Não conseguiu capturar posição atual. Verifique conexão.")
    #     return
    
    # current_lat = msg.lat  # Já em int * 1e7
    # current_lon = msg.lon  # Já em int * 1e7
    # current_alt = msg.relative_alt / 1000.0  # Para log, em metros

    current_lat = 0
    current_lon = 0
    
    # print(f"Posição atual capturada: Lat={current_lat/1e7}, Lon={current_lon/1e7}, Alt relativa={current_alt}m")

    # Configurações do comando
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT  # 6: Alt relativa ao home
    # type_mask = 3576  # Usa posição (X/Y/Z), ignora vel/acc/yaw
    type_mask = 0b110111111011 # ignorar x e y

    duracao_s = 20

    inicio = time.time()
    
    while (time.time() - inicio) < duracao_s:

    # Envia o comando: Mantém lat/lon atuais, ajusta apenas alt
        drone.mav.set_position_target_global_int_send(
            0,  # Tempo de boot (0 para agora)
            drone.target_system, drone.target_component,
            frame,
            type_mask,
            current_lat,  # Lat atual
            current_lon,  # Lon atual
            target_altitude,  # Altitude alvo relativa ao home
            0, 0, 0,  # Velocidades (ignoradas)
            0, 0, 0,  # Acelerações (ignoradas)
            0, 0  # Yaw e yaw_rate (0 para manter heading)
        )
        print(f"[{time.time() - inicio:.1f}s] Ajustando altitude para {target_altitude} m")

        # espera 1 segundo antes do próximo envio
        time.sleep(1)

def set_altitude(drone, target_altitude):
    
    current_lat, current_long, current_alt = get_current_location(drone, id_drone)
    id_drone = drone.target_system

    print(f"Drone {id_drone}: ajustando altitude para {target_altitude} metros.")
    
    
    while True:

        print("1")

        drone.mav.set_position_target_global_int_send(
            0, id_drone, drone.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            0b111111011011,  # Máscara para altura
            int(current_lat * 10**7), int(current_long * 10**7), target_altitude,
            0, 0, 0.2, 0, 0, 0, 0, 0)
        
        time.sleep(delay_comando)

        print("2")
   
        _, _, current_alt = get_current_location(drone, id_drone)
        print(f"Drone {id_drone}: Altura atual: {current_alt}; Altura alvo: {target_altitude}")
        distance_alt = abs(current_alt - target_altitude)
        # _, _, vz = get_velocidade(drone)
        
        if distance_alt < 0.3:
            print(f"Drone {id_drone}: Altitude ajustada")
            break

    drone.mav.set_position_target_global_int_send(
    0, id_drone, drone.target_component,
    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
    0b110000000111,  # Máscara para velocidade e aceleração
    int(current_lat * 10**7), int(current_long * 10**7), target_altitude,
    0, 0, 0, 0, 0, 0, 0, 0)

    

    time.sleep(delay_comando)

    print(f"Drone {id_drone}: Setando aceleração e velocidades para ZERO")


# Função para obter a altitude atual
def get_altitude(drone):
    msg = drone.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    return msg.relative_alt / 1000.0  # Altitude em metros


# Função para retornar e pousar
def return_and_land(drone):
    id_drone = drone.target_system
    print(f"Drone {id_drone} - Retornando ao ponto de origem e pousando...")
    set_flight_mode(drone, id_drone, "RTL")
    time.sleep(5)
    drone.mav.command_long_send(
        id_drone, drone.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)
    time.sleep(5)

def set_circle_mode(enxame, drone_id):
    print(f"Drone {drone_id} alterando para o modo CIRCLE com velocidade vertical zerada.")

    # Mudar para o modo CIRCLE
     # o número do modo CIRCLE pode variar dependendo da configuração; confirme no seu setup
    set_flight_mode(enxame, drone_id, "CIRCLE")

    time.sleep(0.2)

    set_altitude(enxame, drone_id, 20)

    # enxame.mav.set_position_target_global_int_send(
    #     0, drone_id, enxame.target_component,
    #     mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
    #     0b000111111000,  # Máscara para posição
    #     0,
    #     0,
    #     20,
    #     0, 0, 0, 0, 0, 0, 0, 0)
    
    print(f"Drone {drone_id} em modo CIRCLE com velocidade vertical estabilizada.")
    time.sleep(0.5)

def pressionar_para_continuar():
    """
    Pausa a execução até o usuário pressionar qualquer tecla.
    Funciona em Windows, Linux e macOS.
    """
    try:
        # Windows
        import msvcrt
        print("Pressione qualquer tecla para continuar...")
        msvcrt.getch()
    except ImportError:
        # Linux / macOS
        print("Pressione Enter para continuar...")
        input()


def seguidor_atras_do_lider_formacao(
    ponto_atual: Tuple[float, float],
    ponto_lider: Tuple[float, float],
    distancia_atras_m: float = 10.0
) -> List[Tuple[float, float]]:
    """
    Formação com 2 drones: [lider, seguidor]

    - O líder vai para `ponto_lider`.
    - O seguidor fica ATRÁS do líder, na direção oposta ao vetor (ponto_atual -> ponto_lider),
      a uma distância `distancia_atras_m` (em metros).

    Retorna:
        [lider, seguidor] como (lat, lon)
    """
    lat_atual, lon_atual = ponto_atual
    lat_lider, lon_lider = ponto_lider

    # --- 1) Vetor direção em metros: ponto_atual -> ponto_lider ---
    metros_por_grau_lat = 111320.0
    metros_por_grau_lon = 111320.0 * math.cos(math.radians(lat_atual))

    dlat = lat_lider - lat_atual
    dlon = lon_lider - lon_atual

    dx = dlon * metros_por_grau_lon  # leste
    dy = dlat * metros_por_grau_lat  # norte

    dist = math.hypot(dx, dy)

    if dist < 0.1:
        # Sem direção definida: assume "para o norte" como referência
        ux, uy = 0.0, 1.0
    else:
        ux = dx / dist
        uy = dy / dist

    # --- 2) Seguidor atrás do líder (vetor oposto) ---
    dx_seg = -distancia_atras_m * ux
    dy_seg = -distancia_atras_m * uy

    # Conversão metros -> graus em torno do líder
    metros_por_grau_lat_lider = 111320.0
    metros_por_grau_lon_lider = 111320.0 * math.cos(math.radians(lat_lider))

    def offset_metros_para_latlon(lat0: float, lon0: float, dx_m: float, dy_m: float) -> Tuple[float, float]:
        lat = lat0 + dy_m / metros_por_grau_lat_lider
        lon = lon0 + dx_m / metros_por_grau_lon_lider
        return lat, lon

    lider = (lat_lider, lon_lider)
    seguidor = offset_metros_para_latlon(lat_lider, lon_lider, dx_seg, dy_seg)

    return [lider, seguidor]

def gerar_triangulo_formacao(
    ponto_atual: Tuple[float, float],
    ponto_lider: Tuple[float, float],
    raio: float = 10.0
) -> List[Tuple[float, float]]:
    """
    Gera uma formação triangular a partir de um ponto de líder.

    - O drone líder vai para `ponto_lider`.
    - O drone da direita fica a +120° (sentido horário) em relação à direção
      ponto_atual -> ponto_lider.
    - O drone da esquerda fica a -120° (sentido anti-horário).
    - A distância dos drones da esquerda/direita até o líder é `raio` (em metros).

    Parâmetros:
        ponto_atual: (lat, lon) posição atual do líder (para definir a direção de voo)
        ponto_lider: (lat, lon) ponto onde o líder deve ir (centro da formação)
        raio: distância em metros dos drones laterais até o líder

    Retorna:
        [lider, esquerda, direita] como (lat, lon)
    """
    lat_atual, lon_atual = ponto_atual
    lat_lider, lon_lider = ponto_lider

    # --- 1) Vetor direção em metros: ponto_atual -> ponto_lider ---
    metros_por_grau_lat = 111320.0
    metros_por_grau_lon = 111320.0 * math.cos(math.radians(lat_atual))

    dlat = lat_lider - lat_atual
    dlon = lon_lider - lon_atual

    dx = dlon * metros_por_grau_lon  # leste
    dy = dlat * metros_por_grau_lat  # norte

    dist = math.hypot(dx, dy)

    if dist < 0.1:
        # Se o ponto_atual ~ ponto_lider, assume direção "para o norte"
        ux, uy = 0.0, 1.0
    else:
        ux = dx / dist
        uy = dy / dist

    # --- 2) Gira o vetor direção em ±120° ---
    def rotacionar(vx: float, vy: float, ang_rad: float) -> Tuple[float, float]:
        cos_a = math.cos(ang_rad)
        sin_a = math.sin(ang_rad)
        rx = vx * cos_a - vy * sin_a
        ry = vx * sin_a + vy * cos_a
        return rx, ry

    # Ângulos em radianos
    ang_direita = -140 * math.pi / 180.0   # +120° horário
    ang_esquerda =  140 * math.pi / 180.0  # -120° anti-horário

    # Vetores unidade rotacionados
    ux_dir, uy_dir = rotacionar(ux, uy, ang_direita)
    ux_esq, uy_esq = rotacionar(ux, uy, ang_esquerda)

    # --- 3) Calcula offsets em metros em torno do líder ---
    # Líder está exatamente no ponto_lider
    dx_dir = raio * ux_dir
    dy_dir = raio * uy_dir

    dx_esq = raio * ux_esq
    dy_esq = raio * uy_esq

    # Conversão metros -> graus em torno do líder
    metros_por_grau_lat_lider = 111320.0
    metros_por_grau_lon_lider = 111320.0 * math.cos(math.radians(lat_lider))

    def offset_metros_para_latlon(lat0: float, lon0: float, dx_m: float, dy_m: float) -> Tuple[float, float]:
        lat = lat0 + dy_m / metros_por_grau_lat_lider
        lon = lon0 + dx_m / metros_por_grau_lon_lider
        return lat, lon

    lider = (lat_lider, lon_lider)
    direita = offset_metros_para_latlon(lat_lider, lon_lider, dx_dir, dy_dir)
    esquerda = offset_metros_para_latlon(lat_lider, lon_lider, dx_esq, dy_esq)

    # Mantendo a ordem [lider, esquerda, direita] como você já usava
    return [lider, esquerda, direita]

def gerar_triangulo_formacao_old(
    ponto_atual: Tuple[float, float],
    ponto_proximo: Tuple[float, float],
    distancia_lado: float = 25.0
) -> List[Tuple[float, float]]:
    """
    Gera os 3 vértices de um triângulo equilátero centrado no ponto_atual,
    com o vértice da frente apontando para ponto_proximo.

    Retorna:
        [lider, esquerda, direita]  → todos como (lat, lon)
        - líder    = vértice da frente
        - esquerda = traseira esquerda (visto de trás)
        - direita  = traseira direita
    """
    lat1, lon1 = ponto_atual
    lat2, lon2 = ponto_proximo

    # --- 1) Coordenadas locais em metros (aprox. boa pra distâncias pequenas) ---
    metros_por_grau_lat = 111320.0
    metros_por_grau_lon = 111320.0 * math.cos(math.radians(lat1))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    dx = dlon * metros_por_grau_lon  # eixo leste
    dy = dlat * metros_por_grau_lat  # eixo norte

    distancia = math.hypot(dx, dy)
    if distancia < 0.1:
        # Muito perto -> não dá pra definir direção
        return [ponto_atual, ponto_atual, ponto_atual]

    # Vetor unitário na direção do voo
    ux = dx / distancia
    uy = dy / distancia

    # Vetor perpendicular (90° à esquerda)
    vx = -uy
    vy =  ux

    # --- 2) Geometria do triângulo equilátero (em metros) ---
    L = distancia_lado
    h = L * math.sqrt(3) / 2.0   # altura

    # Distâncias em relação ao CENTRO (que é ponto_atual):
    # - líder:  2h/3 à frente
    # - traseiros: h/3 para trás ± L/2 lateral
    front_offset =  2.0 * h / 3.0
    back_offset  = -1.0 * h / 3.0
    lateral      = L / 2.0

    # Offsets em metros (x=leste, y=norte)
    front_x = front_offset * ux
    front_y = front_offset * uy

    back_center_x = back_offset * ux
    back_center_y = back_offset * uy

    left_x  = back_center_x + lateral * vx
    left_y  = back_center_y + lateral * vy

    right_x = back_center_x - lateral * vx
    right_y = back_center_y - lateral * vy

    def meters_to_latlon(x_m: float, y_m: float) -> Tuple[float, float]:
        lat = lat1 + y_m / metros_por_grau_lat
        lon = lon1 + x_m / metros_por_grau_lon
        return (lat, lon)

    lider   = meters_to_latlon(front_x, front_y)
    esquerda = meters_to_latlon(left_x, left_y)
    direita  = meters_to_latlon(right_x, right_y)

    return [lider, esquerda, direita]

def distancia_ate_fpv(drone, ponto):
    global posicao_buffer
    
    lat, lon = ponto

    tentativas = 1
    i=0

    while i<tentativas:

        i=i+1

        # with posicao_lock:
        #     msg = mav_position_copy(posicao)
        # msg = drone.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        msg = posicao_buffer.get_closest(time.time())

        if msg is not None:

            if msg.get_type() == 'GLOBAL_POSITION_INT':
                # Posição absoluta em milímetros e velocidade em cm/s
                current_lat = msg.lat / 1e7          # graus
                current_long = msg.lon / 1e7          # graus
                alt = msg.relative_alt / 1000.0  # metros (relativo ao home)
                vx = msg.vx / 100.0          # m/s
                vy = msg.vy / 100.0          # m/s
                vz = msg.vz / 100.0          # m/s
                hdg = msg.hdg / 100.0        # graus (0-360)

                # print(f"Pos: {lat:.7f}, {lon:.7f} | Alt: {alt:.2f}m | Vel: ({vx:.1f}, {vy:.1f}, {vz:.1f}) m/s | Hdg: {hdg:.1f}°")

                current_point = current_lat, current_long, 0
                dest_point = lat, lon, 0

                distance = distance_between_geo_points(current_point, dest_point)
                # voltagem_bateria = get_bateria_v(drone)

                # print(f"Distancia ate waypoint: {distance}")

                return distance
            
    id_drone = drone.target_system
    print(f"Drone {id_drone}: sem telemetria de distancia até waypoint")

    return None

def goto_location_and_wait_fpv(drone, point, altitude, erro_ate_ponto=5, wait=True):
    # Desempacota as coordenadas do ponto (lat, lon)
    lat, lon = point

    id_drone = drone.target_system

    vel_x = 0
    vel_y = 0
    vel_z = 0

    graus_rate = 0
    yaw_rate = graus_rate * 2 * math.pi / 360

    # Obtém a altitude atual do drone
    # current_altitude = get_altitude(drone)
    
    print(f"Drone {id_drone}: Indo para lat: {lat}, lon: {lon}, altitude: {altitude}m...")

    # Bitmask to indicate which fields should be ignored by the vehicle (see POSITION_TARGET_TYPEMASK enum)
    # bit1:PosX, bit2:PosY, bit3:PosZ, bit4:VelX, bit5:VelY, bit6:VelZ, bit7:AccX, bit8:AccY, bit9:AccZ, bit11:yaw, bit12:yaw rate
    # When providing Pos, Vel and/or Accel all 3 axis must be provided. At least one of Pos, Vel and Accel must be provided (e.g. providing Yaw or YawRate alone is not supported)
    # Use Position : 0b110111111000 / 0x0DF8 / 3576 (decimal)
    # Use Velocity : 0b110111000111 / 0x0DC7 / 3527 (decimal)
    # Use Acceleration : 0b110000111000 / 0x0C38 / 3128 (decimal)
    # Use Pos+Vel : 0b110111000000 / 0x0DC0 / 3520 (decimal)
    # Use Pos+Vel+Accel : 0b110000000000 / 0x0C00 / 3072 (decimal)
    # Use Yaw : 0b100111111111 / 0x09FF / 2559 (decimal)
    # Use Yaw Rate : 0b010111111111 / 0x05FF / 1535 (decimal)

    	
    # Valid options are:
    # MAV_FRAME_GLOBAL (0): alt is in meters above sea level
    # MAV_FRAME_GLOBAL_INT (5): alt is in meters above sea level
    # MAV_FRAME_GLOBAL_RELATIVE_ALT (3): alt is in meters above home
    # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT (6): alt is in meters above home
    # MAV_FRAME_GLOBAL_TERRAIN_ALT (10): alt is in meters above terrain
    # MAV_FRAME_GLOBAL_TERRAIN_ALT_INT (11): alt is in meters above terrain

    tentativas_envio = 3

    for i in range(tentativas_envio):

        drone.mav.set_position_target_global_int_send(
            0, id_drone, drone.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            #   0b110111000000,  #Posição + velocidade
            # 0b110111000000 
            0b110111111000,  # Máscara para posição
            int(lat * 10**7), 
            int(lon * 10**7), 
            altitude,
            vel_x, 
            vel_y, 
            vel_z, 
            0, 
            0, 
            0, 
            0, 
            yaw_rate)
        
        time.sleep(0.2)


    
    while wait:

        distancia_ate_ponto = distancia_ate_fpv(drone, point)

        if distancia_ate_ponto < erro_ate_ponto :
            print(f"Drone {id_drone}: chegou")
            break

        # NAO RETIRE!!
        # sem o sleep esse while segura o lock de posicao
        time.sleep(0.1)

        # else:
        #     print(f"Drone {id_drone}: distancia até ponto {distancia_ate_ponto}")

def enviar_velocidade_visual(drone, velocidade_forward, velocidade_right):
    """
    Envia comando de velocidade para o drone.

    velocidade_forward:
        velocidade para frente do drone, em m/s

    velocidade_right:
        velocidade para a direita do drone, em m/s

    O eixo vertical é mantido em zero.
    """

    velocidade_down = 0.0

    drone.mav.set_position_target_local_ned_send(
        0,
        drone.target_system,
        drone.target_component,

        # Referencial preso ao drone:
        # X = frente
        # Y = direita
        # Z = baixo
        mavutil.mavlink.MAV_FRAME_BODY_NED,

        # Usar somente velocidade
        0b110111000111,

        # Posição ignorada
        0, 0, 0,

        # Velocidade
        velocidade_forward,
        velocidade_right,
        velocidade_down,

        # Aceleração ignorada
        0, 0, 0,

        # Yaw e yaw rate ignorados
        0,
        0
    )

def main():

    # https://ardupilot.org/copter/docs/common-mavlink-mission-command-messages-mav_cmd.html
    # https://ardupilot.org/dev/docs/copter-commands-in-guided-mode.html

    # rota = ler_waypoints_qgc_para_lista("circular2_ama.waypoints")
    rota = ler_waypoints_qgc_para_lista("trajetorias/busca_alvo_esefex__sem_intermediarios_6_passadas_INVERSA.waypoints")
    # rota = ler_waypoints_qgc_para_lista("trajetorias/busca_alvo_ama.waypoints")
    # print(rota)

    
    # port1 = "COM14"
    # baud1 = 57600
    port1 = "COM5"
    baud1 = 57600
    drone_lider = connect_drone(port1, baud1) #FPV
    # drone_lider = connect_drone_sitl(host="127.0.0.1", port=5762)
    print("Conectado ao drone 1")

    # port2 = "COM15" 
    # baud2 = 57600
    # port2 = "COM4" 
    # baud2 = 57600
    # drone_escolta_esquerda = connect_drone(port2, baud2) # Carga ESQUERDA
    # drone_escolta_esquerda = connect_drone_sitl(host="127.0.0.1", port=5772)
    # print("Conectado ao drone 2")

    # port3 = "COM13"
    # baud3 = 57600
    # drone3_escolta_direita = connect_drone_sitl(host="127.0.0.1", port=5782)
    # print("Conectado ao drone 3")

  
    drones = []
    drones.append(drone_lider)
    # drones.append(drone_escolta_esquerda)
    # drones.append(drone3_escolta_direita)


    altitudes = []
    altitudes.append(10)
    # altitudes.append(4)

    # drone = connect_drone_IP("tcp:192.168.0.103:5760")

    for drone in drones:
        set_flight_mode(drone, "GUIDED")

    print("----------")

    delay(2)

    for drone in drones:

        msg = drone.recv_match(type='STATUSTEXT', blocking=True, timeout=2)
        
        if msg:
            print(f"STATUSTEXT: {msg.text}")


    print("Cadastrando recebimento de telemetria de POSIÇÃO...")
    for drone in drones:

        tentativas = 3

        for i in range(tentativas):

            # Telemetria GLOBAL_POSITION_INT
            drone.mav.command_long_send(
                drone.target_system, drone.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                33,      # ID da mensagem GLOBAL_POSITION_INT
                1000000 / 15,  # intervalo em microssegundos → 25 Hz
                0, 0, 0, 0, 0)
            
            drone.mav.command_long_send(
            drone.target_system, drone.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            30, # ATTITUDE
            1000000 / 15,  # intervalo em microssegundos → 25 Hz
            0,0,0,0,0)
            
            time.sleep(0.2)
        
    # for drone in drones:
    threading.Thread(target=telemetria_thread, args=[drones[0]], daemon=True).start()
    
    threading.Thread(target=camera_thread, args=[], daemon=True).start()


    arm_threads = []

    delay(3)

    for drone, altitude  in zip(drones, altitudes):

        tolerancia_m: float = 0.5
        timeout_s: float = 40.0

        th = threading.Thread(target=drone_control.arm_takeoff_and_wait, args=(drone, altitude,tolerancia_m,timeout_s))

        arm_threads.append(th)
    
    for th in arm_threads:
        th.start()
    
    for th in arm_threads:
        th.join()
    
    print("Decolaram")

    threading.Thread(target=rota_automatica, args=(drones, rota, altitudes), daemon=True).start()
    threading.Thread(target=salvamento_automatico_deteccoes_thread, args=(), daemon=True).start()

    while True:
        frame = None
        with frame_lock:
            if frame_global is not None:
                frame = frame_global.copy()

        if frame is not None:
            cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()